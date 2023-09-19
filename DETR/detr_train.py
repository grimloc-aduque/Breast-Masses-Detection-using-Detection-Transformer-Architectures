# %%
import itertools
import os
import shutil
import sys
from io import StringIO

import pandas as pd
import pytorch_lightning as pl

from coco_eval import CocoEvaluator
from detr_config import Config
from detr_dataset import InBreastDataset, collate_fn
from detr_detection import prepare_for_coco_detection
from detr_model import DETRModel
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrConfig, DetrForObjectDetection
from transformers import DeformableDetrImageProcessor, DeformableDetrConfig, DeformableDetrForObjectDetection

STDOUT = sys.stdout

# %%

# HyperParameters

hyperparameters = itertools.product(*[
    Config.ARCHITECTURES,
    Config.BACKBONES,
    Config.NUM_QUERIES,
    Config.D_MODEL,
    Config.TRANSFORMER_LAYERS,
])

hyperparameters = itertools.product(*[
    ['DEFORMABLE-DETR'],
    [Config.BACKBONES[0]],
    [64],
    [128],
    [2],
])



# Hyperparameter Search

for architecture, backbone, num_queries, d_model, transformer_layers in hyperparameters:
    
    print('ARCHITECTURE: ', architecture,
          '\nBACKBONE: ', backbone,
          '\nNUM QUERIES: ', num_queries,
          '\nDIM MODEL: ', d_model,
          '\nENC-DEC LAYERS: ', transformer_layers)
    
    if architecture == 'DETR':
        IMG_PROCESSOR_CLASS = DetrImageProcessor
        DETR_CONFIG_CLASS = DetrConfig
        DETR_CLASS = DetrForObjectDetection
    else:
        IMG_PROCESSOR_CLASS = DeformableDetrImageProcessor
        DETR_CONFIG_CLASS = DeformableDetrConfig
        DETR_CLASS = DeformableDetrForObjectDetection
        
    
    image_processor = IMG_PROCESSOR_CLASS()
    
    
    # Model Configuration

    config = DETR_CONFIG_CLASS(
        num_labels=2,
        id2label = {0:'Mass', 1: 'No-Mass'}, 
        label2id = {'Mass': 0, 'No-Mass': 1},
        num_queries = num_queries,
        d_model = d_model,
        num_head = 8,
        encoder_layers = transformer_layers,
        decoder_layers = transformer_layers,
        backbone=backbone
    )
    
    # Model Directory

    model_name = [
        f'backbone={backbone.split(".")[0]}',
        f'queries={num_queries}',
        f'dmodel={d_model}',
        f'layers={transformer_layers}'
    ]
    
    model_name = '_'.join(model_name)
    
    model_dir = os.path.join(Config.LOGS_DIR, model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    
    metrics_by_fold = []
    index = []

    # K-fold Cross Validation 

    for fold in range(1,11):
        
        # Model
        
        detr_model = DETR_CLASS(
            config = config,
        )

        model = DETRModel(detr_model=detr_model)
        
        # Datasets
        
        fold_name = f'fold_{fold}'
        
        fold_dir = os.path.join(Config.DATASET, fold_name)
        
        train_dataset = InBreastDataset(
            images_path = os.path.join(fold_dir, 'train'),
            processor=image_processor
        )
        
        train_loader = DataLoader(
            dataset = train_dataset,
            batch_size = Config.BATCH_SIZE,
            collate_fn = collate_fn,
        )

        valid_dataset = InBreastDataset(
            images_path = os.path.join(fold_dir, 'valid'),
            processor=image_processor
        )

        valid_loader = DataLoader(
            dataset = valid_dataset,
            batch_size = Config.BATCH_SIZE,
            collate_fn = collate_fn,
        )
        
        # Training

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k = 1,
            save_last = True,
            monitor = "valid_loss",
            mode = "min"
        )

        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor = 'valid_loss',
            patience = 20
        )

        version = os.path.join(model_name, fold_name)

        logger = pl.loggers.TensorBoardLogger(
            save_dir = './',
            version = version
        )

        trainer = Trainer(
            max_epochs = Config.EPOCHS,
            log_every_n_steps = 5, 
            callbacks = [
                checkpoint_callback, 
                early_stopping_callback
            ],
            accelerator = Config.ACCELERATOR,
            logger = logger
        )
        
        trainer.fit(
            model, 
            train_dataloaders = train_loader, 
            val_dataloaders = valid_loader
        )
        
        
        # Validation
        
        checkpoints_dir = os.path.join(Config.LOGS_DIR, version, 'checkpoints')
        best_checkpoint = [f for f in os.listdir(checkpoints_dir) if 'last' not in f][0]
        checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
        
        model = DETRModel.load_from_checkpoint(checkpoint_path)

        evaluator = CocoEvaluator(
            coco_gt=valid_dataset.coco, 
            iou_types=["bbox"]
        )
        
        for batch in valid_loader:
            outputs = model(batch['pixel_values'])
            predictions = image_processor.post_process_object_detection(outputs, threshold=0.1)
            image_ids = [label['image_id'].item() for label in batch['labels']]
            predictions = {image_id:output for image_id, output in zip(image_ids, predictions)}
            predictions = prepare_for_coco_detection(predictions)
            evaluator.update(predictions)
            
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        
        # Metrics
        
        metrics_buffer = StringIO()
        sys.stdout = metrics_buffer
        evaluator.summarize()
        sys.stdout = STDOUT
        
        metrics = metrics_buffer.getvalue()
        metrics = metrics.split('\n')
        metrics = [m for m in metrics if 'Average' in m]
        metrics_dict = {}
        for metric in metrics:
            name, value = metric.split(' = ')
            metrics_dict[name[1:]] = float(value)
        
        metrics_by_fold.append(metrics_dict)
        index.append(fold_name)
        
        shutil.rmtree(checkpoints_dir)
          

        break # Fold
    
    break 
    
    # Aggregate Metrics
    
    metrics_by_fold = pd.DataFrame(metrics_by_fold, index=index)
    metrics_by_fold.loc['mean'] = metrics_by_fold.mean()
    
    metrics_path = os.path.join(
        Config.LOGS_DIR,
        model_name, 
        Config.METRICS_FILE
    )
    
    metrics_by_fold.to_csv(metrics_path)
    
    break # Hyperparameter


# %%



