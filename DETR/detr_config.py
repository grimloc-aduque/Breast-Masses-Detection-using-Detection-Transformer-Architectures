
import os

import torch


class Config:
    DATASET = 'InBreast_Coco'
    TRAIN_DIRECTORY = os.path.join(DATASET, "train_valid")
    TEST_DIRECTORY = os.path.join(DATASET, "test")
    CHECKPOINT = "facebook/detr-resnet-50"
    # CHECKPOINT = "SenseTime/deformable-detr"

    EPOCHS = 100
    BATCH_SIZE = 16
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # NUM_QUERIES = [25, 50, 100]
    # D_MODEL = [64, 128, 256]
    # TRANSFORMER_LAYERS = [2,4,6]

    NUM_QUERIES = [100]
    D_MODEL = [256]
    TRANSFORMER_LAYERS = [6]

    BACKBONES = [
        'efficientnet_b0.ra_in1k', # 3.96M
        'efficientnet_b3.ra2_in1k', # 10.02
        'resnet18.a1_in1k', # 11.17M
        'resnet50.a1_in1k', # 23.45M
    ]
