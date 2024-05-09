import argparse
import cv2
import os
import time
from logger import fps_logger, average_fps_logger, average_inference_time_logger, average_precision_logger

from tflite_runtime.interpreter import Interpreter

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from logger import LoggedData

default_labels = 'coco_labels.txt'

def get_average_inference_score(objs): 
    total_score = sum(obj.score for obj in objs)
    return total_score/len(objs)

def load_all_model():
    models_fullpath = []
    models_dir_location = os.listdir("models")
    for model_dir in models_dir_location:
        model_dir_path = os.path.join("models",model_dir)
        models_location = os.listdir(model_dir_path)
        for model in models_location:
            if len(models_location) < 3:
                models_fullpath.append(os.path.join(model_dir_path, model))
            else:
                continue
    return models_fullpath        
    
def load_label(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}
        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}
        
def get_cmd():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--labels', help='label file path',
                        default=default_labels)
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--source', type=str, help='Index of which video source to use.', default="webcam")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')

    return parser.parse_args()

def main():
    # todo: redefine args.model
    args = get_cmd()

    models = load_all_model()
    
    all_models_data = []
    
    for model in models:
        print('Loading {} with {} labels as default.'.format(model, args.labels))
        labels = load_label(args.labels)   
        
        isEdgeTPUEnabled = True
             
        if "_edgetpu" in model: 
            interpreter = make_interpreter(model) # doesn't have to load delegate as make_interpreter already loaded it for us
        else:
            isEdgeTPUEnabled = False
            non_tpu_model = model.replace("_edgetpu", "")
            interpreter = Interpreter(non_tpu_model)   
            
        interpreter.allocate_tensors()    
        inference_size = input_size(interpreter) # width, height
            
        if args.source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.source)

        frame_count = 0
        start_time = time.time()
        
        fps_values = []
        inference_time = []
        times = []
        precisions = []

        while cap.isOpened():
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break

            cv2_im = frame

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB) 
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)        
        
            t1 = time.time()
            run_inference(interpreter, cv2_im_rgb.tobytes())
            t2 = time.time()
            inference_time.append(t2-t1)
            objs = get_objects(interpreter, args.threshold)[:args.top_k] # detected objects, top K mean the most positive scoring only
            if len(objs) > 0:
                precisions.append(get_average_inference_score(objs=objs))
            cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
            
            # get the current running time for the fps
            elapsed_time = time.time() - start_time
            fps_values.append(frame_count / elapsed_time)
            times.append(elapsed_time)
            
            cv2_im = cv2.putText(cv2_im, f"FPS: {frame_count / elapsed_time:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                loggedData = LoggedData(
                    testedModel=model.split('/')[1],
                    average_fps = round(sum(fps_values)/len(fps_values),5),
                    average_inference_time = sum(inference_time)/len(inference_time),
                    average_precision = sum(precisions)/len(precisions),
                    isEdgeTPUEnabled = isEdgeTPUEnabled
                )
                all_models_data.append(loggedData)
                average_fps_logger(all_models_data)
                average_inference_time_logger(all_models_data)
                average_precision_logger(all_models_data)
                break
        print("~~~~~~~~~~~~~~~~FINISH EVALUATING~~~~~~~~~~~~~~")
        loggedData = LoggedData(
            testedModel=model.split('/')[1],
            average_fps = round(sum(fps_values)/len(fps_values),5),
            average_inference_time = sum(inference_time)/len(inference_time),
            average_precision = sum(precisions)/len(precisions),
            isEdgeTPUEnabled = isEdgeTPUEnabled
        )
        all_models_data.append(loggedData)
    cap.release()
    cv2.destroyAllWindows()
    average_fps_logger(all_models_data)
    average_inference_time_logger(all_models_data)
    average_precision_logger(all_models_data)


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, _ = cv2_im.shape
    
    inference_width, inference_height = inference_size[0], inference_size[1]
    
    scale_x, scale_y = width / inference_width, height / inference_height
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()