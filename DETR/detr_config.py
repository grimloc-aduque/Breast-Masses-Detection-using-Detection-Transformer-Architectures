
import itertools

import torch


class Config:
    DATASET = 'InBreast_Coco'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    NUM_CLASSES = 1
    ACCELERATOR = 'cpu'
    DEVICE = 'cpu'

    # ARCHITECTURES = ['DETR', 'D-DETR']
    # NUM_QUERIES = [25, 50, 100]
    # TRANSFORMER_LAYERS = [2,4,6]
    # HYPERPARAMS = itertools.product(*[
    #     ARCHITECTURES,
    #     NUM_QUERIES,
    #     TRANSFORMER_LAYERS,
    # ])
    
    EPOCHS = 200
    BATCH_SIZE = 16
    FOLDS = 10
    THRESHOLDS = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    LOCAL_ENV = False
    
    
    def set_local_settings():
        Config.LOCAL_ENV = True
        Config.BATCH_SIZE = 6
        Config.FOLDS = 10
        Config.THRESHOLDS = [0.1]
        Config.HYPERPARAMS = [
            ('DETR', 100, 6),
            # ('D-DETR', 100, 6),
        ]
        
    def set_gpu_settings():
        gpu_available = torch.cuda.is_available()
        Config.ACCELERATOR = 'gpu' if gpu_available else 'cpu'
        Config.DEVICE = 'cuda' if gpu_available else 'cpu'
        Config.HYPERPARAMS = [
            # ('D-DETR', 100, 6),
            ('DETR', 100, 6),
        ]