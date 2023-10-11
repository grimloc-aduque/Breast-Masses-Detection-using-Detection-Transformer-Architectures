
import torch


class Config:
    
    GPU_AVAILABLE = torch.cuda.is_available()
    # GPU_AVAILABLE = False
    DATASET = 'InBreast-COCO'
    # DATASET = 'DDSM-COCO'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    METRIC_PLOT = 'plot_metrics.png'
    NUM_CLASSES = 1
    
    def set_device():
        CUDA = torch.cuda.is_available()
        MPS = torch.backends.mps.is_available()
        if CUDA:
            Config.ACCELERATOR = 'gpu'
            Config.DEVICE = 'cuda'
        elif MPS:
            Config.ACCELERATOR = 'mps'
            Config.DEVICE = 'mps'
        else:
            Config.ACCELERATOR = 'cpu'
            Config.DEVICE = 'cpu'

    def set_local_settings():
        Config.set_device()
        Config.LOCAL_ENV = True
        Config.FOLDS = 2
        Config.BATCH_SIZE = 8
        Config.STEPS = 1
        Config.THRESHOLDS = [0.001, 0.1]
        Config.HYPERPARAMS = [
            ('DETR', 256, 100, 6),
        ]
        
    def set_benchmark_settings():
        Config.set_device()
        Config.LOCAL_ENV = False
        Config.FOLDS = 10
        Config.BATCH_SIZE = 16
        Config.EPOCHS = 200
        Config.THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        Config.HYPERPARAMS = [
            # Originales
            ('DETR', 256, 100, 6), 
            ('D-DETR', 256, 100, 6),
            
            ('DETR', 256, 100, 2),
            ('DETR', 256, 100, 4),
            ('DETR', 256, 100, 8),
            ('DETR', 256, 100, 10),
        
            ('D-DETR', 256, 100, 2),
            ('D-DETR', 256, 100, 4),
            ('D-DETR', 256, 100, 8),
            ('D-DETR', 256, 100, 10),
        ]


