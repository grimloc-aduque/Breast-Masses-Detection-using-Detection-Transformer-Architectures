
import os
import torch


class Config:
    ROOT = 'YOLOv8'
    DATASET = 'InBreast_Yolov8'
    RUNS_DIR = 'runs_usfq_server'
    MODEL_SIZES = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    BATCH_SIZE = 16
    EPOCHS = 200
    YAML_FILE = 'data.yaml'
    # YAML_FILE = 'data_docker.yaml'