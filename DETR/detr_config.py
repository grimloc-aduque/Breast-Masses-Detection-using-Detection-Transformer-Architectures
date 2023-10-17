
import torch


class Config:
    
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
        Config.HYPERPARAMS = []
        Config.add_original_hyperparams()
    
    
    def set_benchmark_settings():
        Config.set_device()
        Config.LOCAL_ENV = False
        Config.FOLDS = 10
        Config.BATCH_SIZE = 16
        Config.EPOCHS = 200
        Config.THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        Config.HYPERPARAMS = []
        Config.add_original_hyperparams()
        # Config.add_layers_hyperparams()
        # Config.add_queries_hyperparams()
        # Config.add_dims_hyperparams()
        
    
    def add_original_hyperparams():
        Config.HYPERPARAMS.extend([            
            ('DETR', 256, 100, 6),
            ('D-DETR', 256, 100, 6),
        ])
    
    def add_layers_hyperparams():
        Config.HYPERPARAMS.extend([
            ('D-DETR', 256, 100, 2),
            ('D-DETR', 256, 100, 4),
            ('D-DETR', 256, 100, 8),
        ])

    def add_queries_hyperparams():
        Config.HYPERPARAMS.extend([
            ('D-DETR', 256, 50, 6),
            ('D-DETR', 256, 75, 6),
            ('D-DETR', 256, 125, 6),
        ])
        
    def add_dims_hyperparams():
        Config.HYPERPARAMS.extend([
            ('D-DETR', 128, 100, 6),
            ('D-DETR', 192, 100, 6),
            ('D-DETR', 320, 100, 6),
        ])


