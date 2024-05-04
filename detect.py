import argparse
import cv2
import os
import time
from logger import fps_logger

from tflite_runtime.interpreter import Interpreter

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

#these pycoral code are all based on tensor flow 

'''
# Run:
# 1) With out edgetpu:
# $ python3 detect.py --model <MODEL_LOCATION> --labels <LABEL_LOCATION>
# 2) With edgetpu:
# $ python3 detect.py --model <MODEL_LOCATION> --labels <LABEL_LOCATION> --edgetpu True

If not define -> uses pre-defined default models
'''

default_model_dir = 'models/tf2_mobilenet_v1'
default_model = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
default_labels = 'coco_labels.txt'

def start_logger(fps, times, testedModel, mode):
    fps_logger(
        fps=fps, 
        times=times, 
        testedModel=f"{testedModel} {'EdgeTPU Enable' if mode == 'True' else 'EdgeTPU Disable'}"
    )

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
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=default_labels)
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--source', type=str, help='Index of which video source to use.', default="webcam")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--edgetpu', help='EdgeTpu Flag', default=False)

    return parser.parse_args()

def main():
    args = get_cmd()
    print('Loading {} with {} labels as default.'.format(args.model, args.labels))

    labels = load_label(args.labels)
    
    if args.edgetpu == "True": 
        interpreter = make_interpreter(args.model) # doesn't have to load delegate as make_interpreter already loaded it for us
    else:
        non_tpu_model = args.model.replace("_edgetpu", "")
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
    times = []

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB) 
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)        
       
        t1 = cv2.getTickCount()
        run_inference(interpreter, cv2_im_rgb.tobytes())
        t2 = cv2.getTickCount()
        objs = get_objects(interpreter, args.threshold)[:args.top_k] # detected objects, top K mean the most positive scoring only
        print(objs, (t2-t1) / cv2.getTickFrequency(), "\n")

        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        
        # get the current running time for the fps
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time        
        
        fps_values.append(fps)
        times.append(elapsed_time)
        
        cv2_im = cv2.putText(cv2_im, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            start_logger(fps_values, times, args.model.split('/')[1] ,args.edgetpu)
            break

    cap.release()
    cv2.destroyAllWindows()
    start_logger(fps_values, times, args.model.split('/')[1] ,args.edgetpu)



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