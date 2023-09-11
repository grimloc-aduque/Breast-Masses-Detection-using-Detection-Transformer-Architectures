
import os

import torch


class Config:
    DATASET = 'InBreast_Coco'
    TRAIN_DIRECTORY = os.path.join(DATASET, "train")
    TEST_DIRECTORY = os.path.join(DATASET, "test")
    CHECKPOINT = "facebook/detr-resnet-50"
    EPOCHS = 100
    BATCH_SIZE = 16
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    NUM_QUERIES = [10, 50, 100]
    D_MODEL = [64, 128, 256]
    ENCODER_DECODER_LAYERS = [2,4,6]
