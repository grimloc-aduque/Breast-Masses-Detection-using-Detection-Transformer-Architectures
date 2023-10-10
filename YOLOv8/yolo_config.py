

import torch


class Config:
    ROOT = 'YOLOv8'
    DATASET = 'InBreast_YOLOv8'
    RUNS_DIR = 'runs_usfq_server_07_test'
    DEVICE = 'cpu'
    NUM_CLASSES = 1
    
    MODEL_SIZES = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    EPOCHS = 200
    BATCH_SIZE = 16
    FOLDS = range(1,11)    
    THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    IMG_SIZE = 800
    
    YAML_FILE = 'data_docker.yaml'
    METRICS_FILE = 'metrics.csv'
    LOCAL_ENV = False
    
    
    def set_local_settings():
        Config.MODEL_SIZES = ['yolov8n.pt']
        Config.EPOCHS = 1
        Config.BATCH_SIZE = 4
        Config.FOLDS = [1,2]
        Config.THRESHOLDS = [0.001, 0.5]
        Config.YAML_FILE = 'data.yaml'
        Config.LOCAL_ENV = True
        
    def set_gpu_settings():
        gpu_available = torch.cuda.is_available()
        Config.DEVICE = 0 if gpu_available else 'cpu'
