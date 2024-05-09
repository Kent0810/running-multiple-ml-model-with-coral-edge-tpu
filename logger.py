import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class LoggedData:
    def __init__(self, average_fps, average_inference_time, average_precision, testedModel, isEdgeTPUEnabled):
        self.testedModel = testedModel
        self.average_fps = average_fps
        self.average_inference_time = average_inference_time
        self.average_precision = average_precision
        self.isEdgeTPUEnabled = isEdgeTPUEnabled
    
def fps_logger(self, times, testedModel):
    plt.title(f'FPS Over Time - {testedModel} - {"Coral EdgeTPU Enable" if self.isEdgeTPUEnable else "Coral EdgeTPU Disable"}')
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.plot(times, self.average_fps)
    plt.savefig(f"reports/{testedModel}_fps_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png")
    plt.show()

def average_fps_logger(all_models_data):
    labels = list(set([model.testedModel for model in all_models_data]))
    all_edgetpu_fps = [model.average_fps for model in all_models_data if model.isEdgeTPUEnabled == True]
    all_non_edgetpu_fps = [model.average_fps for model in all_models_data if model.isEdgeTPUEnabled == False]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax =plt.subplots()
    ax.bar(x - width/2, all_edgetpu_fps, width, label='Edge TPU Enable')
    ax.bar(x + width/2, all_non_edgetpu_fps, width, label='Edge TPU Disable')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Frame per second')
    ax.set_title("Average FPS")
    ax.set_xticks(x)   
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"reports/average_fps_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png", bbox_inches="tight")
    plt.show()

def average_inference_time_logger(all_models_data):
    labels = list(set([model.testedModel for model in all_models_data]))
    all_edgetpu_inference_time = [model.average_inference_time * 100 for model in all_models_data if model.isEdgeTPUEnabled == True]
    all_non_edgetpu_inference_time = [model.average_inference_time * 100 for model in all_models_data if model.isEdgeTPUEnabled == False]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax =plt.subplots()
    ax.bar(x - width/2, all_edgetpu_inference_time, width, label='Edge TPU Enable')
    ax.bar(x + width/2, all_non_edgetpu_inference_time, width, label='Edge TPU Disable')

    ax.set_xlabel('Models')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title("Average Inference Time")
    ax.set_xticks(x)   
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"reports/average_inference_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png", bbox_inches="tight")
    plt.show()

def average_precision_logger(all_models_data):
    labels = list(set([model.testedModel for model in all_models_data]))
    all_edgetpu_precision = [model.average_precision * 100 for model in all_models_data if model.isEdgeTPUEnabled == True]
    all_non_edgetpu_precision = [model.average_precision * 100 for model in all_models_data if model.isEdgeTPUEnabled == False]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax =plt.subplots()
    ax.bar(x - width/2, all_edgetpu_precision, width, label='Edge TPU Enable')
    ax.bar(x + width/2, all_non_edgetpu_precision, width, label='Edge TPU Disable')
 
    ax.set_ylabel('Precision')
    ax.set_title("Average Precision (%)")
    ax.set_xticks(x)   
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"reports/average_precision_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png", bbox_inches="tight")
    plt.show()