# %%
import os
import shutil

import pandas as pd
from detr_config import Config
from detr_dataset import get_dataset, get_dataloader
from detr_trainer import get_trainer
from detr_evaluation import load_best_model, get_metrics
from detr_model import DETRModel
from transformers import (DeformableDetrConfig,
                          DeformableDetrForObjectDetection,
                          DeformableDetrImageProcessor, DetrConfig,
                          DetrForObjectDetection, DetrImageProcessor)

# %%

# HyperParameters

hyperparameters = Config.HYPERPARAMS

hyperparameters = [
    ('DETR', 'resnet50', 100, 256, 6),
    # ('D-DETR', 'resnet50', 300, 256, 6),
]

# Hyperparameter Search

for architecture, backbone, num_queries, d_model, transformer_layers in hyperparameters:
    
    if architecture == 'DETR':
        IMG_PROCESSOR_CLASS = DetrImageProcessor
        DETR_CONFIG_CLASS = DetrConfig
        DETR_CLASS = DetrForObjectDetection
    else:
        IMG_PROCESSOR_CLASS = DeformableDetrImageProcessor
        DETR_CONFIG_CLASS = DeformableDetrConfig
        DETR_CLASS = DeformableDetrForObjectDetection    
    
    # Model Configuration

    config = DETR_CONFIG_CLASS(
        num_labels = Config.NUM_CLASSES,
        id2label = {0:'Mass'}, 
        label2id = {'Mass': 0},
        num_queries = num_queries,
        d_model = d_model,
        encoder_layers = transformer_layers,
        decoder_layers = transformer_layers,
        backbone=backbone
    )
    
    # Model Directory

    model_name = [
        'balloon_v2', # Remove this
        f'model={architecture}',
        f'backbone={backbone.split(".")[0]}',
        f'queries={num_queries}',
        f'dmodel={d_model}',
        f'layers={transformer_layers}'
    ]
    
    model_name = '_'.join(model_name)
    model_dir = os.path.join(Config.LOGS_DIR, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    print('\n-------- MODEL --------\n', model_name,
          '\n-----------------------\n')
    
    
    # K-fold Cross Validation 
    
    metrics_by_fold = []
    index = []

    for fold in range(1,11):
        
        # Model
        
        image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")        
        detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels = 1,
            id2label = {0:'Mass'}, 
            label2id = {'Mass': 0},
            ignore_mismatched_sizes=True
        )
        model = DETRModel(detr_model=detr_model)
        
        # Datasets
        
        fold_name = f'fold_{fold}'
        dataset_dir = os.path.join(Config.DATASET, fold_name)
        # dataset_dir = './balloon/' # Remove this
        
        train_dir = os.path.join(dataset_dir, 'train')
        train_dataset = get_dataset(train_dir, image_processor)
        train_loader = get_dataloader(train_dataset, image_processor)
        
        valid_dir = os.path.join(dataset_dir, 'valid')
        valid_dataset = get_dataset(valid_dir, image_processor)
        valid_loader = get_dataloader(valid_dataset, image_processor)

        # Training
        
        version = os.path.join(model_name, fold_name)
        trainer = get_trainer(version)
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        
        # Validation

        model = load_best_model(version)
        metrics_dict = get_metrics(model, valid_dataset, image_processor, threshold=0.001)
        metrics_by_fold.append(metrics_dict)
        index.append(fold_name)
        
        if fold != 1:
            print("Cleaning Checkpoints")
            # shutil.rmtree(checkpoints_dir)

        break # Fold
    
    
    # Aggregate Metrics
    
    metrics_by_fold = pd.DataFrame(metrics_by_fold, index=index)
    metrics_by_fold.loc['mean'] = metrics_by_fold.mean()
    metrics_path = os.path.join(Config.LOGS_DIR, model_name, Config.METRICS_FILE)
    metrics_by_fold.to_csv(metrics_path)
    
    break # Hyperparameter


# %%

# HyperParameters

hyperparameters = Config.HYPERPARAMS

hyperparameters = [
    ('DETR', 'resnet50', 100, 256, 6),
    # ('D-DETR', 'resnet50', 300, 256, 6),
]

# Hyperparameter Search

for architecture, backbone, num_queries, d_model, transformer_layers in hyperparameters:
    
    if architecture == 'DETR':
        IMG_PROCESSOR_CLASS = DetrImageProcessor
        DETR_CONFIG_CLASS = DetrConfig
        DETR_CLASS = DetrForObjectDetection
    else:
        IMG_PROCESSOR_CLASS = DeformableDetrImageProcessor
        DETR_CONFIG_CLASS = DeformableDetrConfig
        DETR_CLASS = DeformableDetrForObjectDetection    
    
    # Model Configuration

    config = DETR_CONFIG_CLASS(
        num_labels = Config.NUM_CLASSES,
        id2label = {0:'Mass'}, 
        label2id = {'Mass': 0},
        num_queries = num_queries,
        d_model = d_model,
        encoder_layers = transformer_layers,
        decoder_layers = transformer_layers,
        backbone=backbone
    )
    
    # Model Directory

    model_name = [
        'balloon', # Remove this
        f'model={architecture}',
        f'backbone={backbone.split(".")[0]}',
        f'queries={num_queries}',
        f'dmodel={d_model}',
        f'layers={transformer_layers}'
    ]
    
    model_name = '_'.join(model_name)
    model_dir = os.path.join(Config.LOGS_DIR, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    print('\n-------- MODEL --------\n', model_name,
          '\n-----------------------\n')
    
    
    # K-fold Cross Validation 
    
    metrics_by_fold = []
    index = []

    for fold in range(1,11):
        
        # Model
        
        image_processor = IMG_PROCESSOR_CLASS()        
        detr_model = DETR_CLASS(config=config)
        model = DETRModel(detr_model=detr_model)
        
        # Datasets
        
        fold_name = f'fold_{fold}'
        dataset_dir = os.path.join(Config.DATASET, fold_name)
        dataset_dir = './balloon/' # Remove this
        
        train_dir = os.path.join(dataset_dir, 'train')
        train_dataset = get_dataset(train_dir, image_processor)
        train_loader = get_dataloader(train_dataset, image_processor)
        
        valid_dir = os.path.join(dataset_dir, 'valid')
        valid_dataset = get_dataset(valid_dir, image_processor)
        valid_loader = get_dataloader(valid_dataset, image_processor)

        # Training
        
        version = os.path.join(model_name, fold_name)
        trainer = get_trainer(version)
        trainer.fit(model, train_dataloaders=train_loader,
                    val_dataloaders=valid_loader)
        
        # Validation

        model = load_best_model(version)
        metrics_dict = get_metrics(model, valid_dataset, image_processor, threshold=0.001)
        metrics_by_fold.append(metrics_dict)
        index.append(fold_name)
        
        if fold != 1:
            print("Cleaning Checkpoints")
            # shutil.rmtree(checkpoints_dir)

        break # Fold
    
    
    # Aggregate Metrics
    
    metrics_by_fold = pd.DataFrame(metrics_by_fold, index=index)
    metrics_by_fold.loc['mean'] = metrics_by_fold.mean()
    metrics_path = os.path.join(Config.LOGS_DIR, model_name, Config.METRICS_FILE)
    metrics_by_fold.to_csv(metrics_path)
    
    break # Hyperparameter


# %%



