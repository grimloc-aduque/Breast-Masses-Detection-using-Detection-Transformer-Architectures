
import torch


class Config:
    
    DATASET = 'InBreast-COCO'
    # DATASET = 'DDSM-COCO'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    METRIC_PLOT = 'plot_metrics.png'
    NUM_CLASSES = 1
    ACCELERATOR = 'cpu'
    DEVICE = 'cpu'
    
    EPOCHS = 200
    BATCH_SIZE = 16
    FOLDS = 10
    THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    LOCAL_ENV = False
    
    
    def set_local_settings():
        Config.LOCAL_ENV = True
        Config.BATCH_SIZE = 12
        Config.FOLDS = 3
        Config.THRESHOLDS = [0.001, 0.1]
        Config.HYPERPARAMS = [
            ('DETR', 256, 100, 6),
            # ('D-DETR', 256, 100, 6),
        ]
        
    def set_gpu_settings():
        gpu_available = torch.cuda.is_available()
        Config.ACCELERATOR = 'gpu' if gpu_available else 'cpu'
        Config.DEVICE = 'cuda' if gpu_available else 'cpu'
        Config.HYPERPARAMS = [
            # ('D-DETR', 256, 100, 6),
            ('DETR', 256, 100, 6),
        ]