
import itertools

import torch


class Config:
    DATASET = 'InBreast_Coco'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    NUM_CLASSES = 1

    ARCHITECTURES = ['DETR', 'D-DETR']
    NUM_QUERIES = [25, 50, 100]
    TRANSFORMER_LAYERS = [2,4,6]
    HYPERPARAMS = itertools.product(*[
        ARCHITECTURES,
        NUM_QUERIES,
        TRANSFORMER_LAYERS,
    ])
    EPOCHS = 300
    BATCH_SIZE = 16
    THRESHOLD = 0.01
    
    
    def set_local_settings():
        Config.EPOCHS = 1
        Config.BATCH_SIZE = 4
        Config.ACCELERATOR = 'cpu'
        Config.DEVICE = 'cpu'
        Config.HYPERPARAMS = [
            ('DETR', 100, 6),
            # ('D-DETR', 100, 6)
        ]

    def set_cpu_settings():
        Config.ACCELERATOR = 'cpu'
        Config.DEVICE = 'cpu'
        
    def set_gpu_settings():
        gpu_available = torch.cuda.is_available()
        Config.ACCELERATOR = 'gpu' if gpu_available else 'cpu'
        Config.DEVICE = 'cuda' if gpu_available else 'cpu'