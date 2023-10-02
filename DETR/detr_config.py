
import itertools

import torch


class Config:
    DATASET = 'InBreast_Coco'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    METRICS_DICT = 'metrics.json'
    NUM_CLASSES = 1
    ACCELERATOR = 'cpu'
    DEVICE = 'cpu'

    ARCHITECTURES = ['DETR', 'D-DETR']
    NUM_QUERIES = [25, 50, 100]
    TRANSFORMER_LAYERS = [2,4,6]
    HYPERPARAMS = itertools.product(*[
        ARCHITECTURES,
        NUM_QUERIES,
        TRANSFORMER_LAYERS,
    ])
    EPOCHS = 200
    BATCH_SIZE = 16
    FOLDS = range(1,11)
    THRESHOLDS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    LOCAL_ENV = False
    
    
    def set_local_settings():
        Config.LOCAL_ENV = True
        Config.BATCH_SIZE = 4
        Config.FOLDS = [1, 2]
        Config.THRESHOLDS = [0.01, 0.5]
        Config.HYPERPARAMS = [
            ('D-DETR', 300, 6),
            ('DETR', 100, 6),
        ]
        
    def set_gpu_settings():
        Config.NUM_QUERIES = [100]
        Config.TRANSFORMER_LAYERS = [6]
        Config.EPOCHS = 6
        gpu_available = torch.cuda.is_available()
        Config.ACCELERATOR = 'gpu' if gpu_available else 'cpu'
        Config.DEVICE = 'cuda' if gpu_available else 'cpu'