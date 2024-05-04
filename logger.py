import matplotlib.pyplot as plt
from datetime import datetime

def fps_logger (fps, times, testedModel):
    plt.title(f'FPS Over Time - {testedModel}')
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.plot(times, fps)
    plt.savefig(f"reports/{testedModel}_fps_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png")

    plt.show()
    
def average_fps_logger(average_fps, testedModel):
    plt.figure(figsize=(10, 6))
    plt.title(f'Average FPS')
    plt.xlabel('Models')
    plt.ylabel('Average FPS')
    plt.grid(True)
    plt.bar(x=testedModel, height=average_fps)
    plt.savefig(f"reports/average_fps_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png",bbox_inches="tight")
    plt.show()
    
def average_inference_time_logger(average_inference_time, testedModel):
    plt.figure(figsize=(10, 6))
    plt.title(f'Average Inference Time (ms)')
    plt.xlabel('Models')
    plt.ylabel('Average Inference Time (ms)')
    plt.grid(True)
    plt.bar(x=testedModel, height=[average_time * 100 for average_time in average_inference_time])
    plt.savefig(f"reports/average_fps_reports_{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.png",bbox_inches="tight")
    plt.show()