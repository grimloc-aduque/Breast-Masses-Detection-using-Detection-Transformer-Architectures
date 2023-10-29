
import torch


class Config:
    
    DATASET = 'InBreast-COCO'
    # DATASET = 'DDSM-COCO'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    METRICS_PLOT = 'plot_metrics.png'
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
            # ('DETR', 'resnet50', 256, 100, 6),
            ('D-DETR', 'resnet18', 64, 50, 2),
        ]
    
    
    def set_benchmark_settings():
        Config.set_device()
        Config.LOCAL_ENV = False
        Config.FOLDS = 10
        Config.BATCH_SIZE = 16
        Config.EPOCHS = 200
        Config.THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        Config.HYPERPARAMS = [
            # ('DETR', 256, 100, 6),
            # ('D-DETR', 'resnet50', 256, 100, 6),
            
            # ('D-DETR', 'resnet50', 256, 100, 2),
            # ('D-DETR', 'resnet50', 256, 100, 4),
            # ('D-DETR', 'resnet50', 256, 100, 8),
            
            # ('D-DETR', 'resnet50', 256, 50, 6),
            # ('D-DETR', 'resnet50', 256, 75, 6),
            # ('D-DETR', 'resnet50', 256, 125, 6),
            
            # ('D-DETR', 'resnet50', 128, 100, 6),
            # ('D-DETR', 'resnet50', 192, 100, 6),
            # ('D-DETR', 'resnet50', 320, 100, 6),
            
            ('D-DETR', 'resnet18', 256, 100, 6),
            ('D-DETR', 'resnet26', 256, 100, 6),
            ('D-DETR', 'resnet34', 256, 100, 6),
        ]
