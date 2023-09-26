
import itertools

import torch


class Config:
    DATASET = 'InBreast_Coco'
    LOGS_DIR = 'lightning_logs'
    METRICS_FILE = 'metrics.csv'
    # ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ACCELERATOR = 'cpu'
    DEVICE = 'cpu'
    
    NUM_CLASSES = 1

    ARCHITECTURES = ['DETR', 'D-DETR']
    BACKBONES = [
        'resnet18.a1_in1k', # 11.17M
        'resnet34.a1_in1k', # 21.27M
        'resnet50.a1_in1k', # 23.45M
    ]
    NUM_QUERIES = [25, 50, 100]
    D_MODEL = [64, 128, 256]
    TRANSFORMER_LAYERS = [2,4,6]
    
    HYPERPARAMS = itertools.product(*[
        ARCHITECTURES,
        BACKBONES,
        NUM_QUERIES,
        D_MODEL,
        TRANSFORMER_LAYERS,
    ])
    
    
    EPOCHS = 300
    BATCH_SIZE = 16
    
    
