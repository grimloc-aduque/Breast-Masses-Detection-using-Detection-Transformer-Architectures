# %%
import os
import shutil

import pandas as pd
from ultralytics import YOLO
from yolo_config import Config

# %%
for model_size in Config.MODEL_SIZES:
    
    model_dir = os.path.join(Config.RUNS_DIR, Config.DATASET, model_size)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        
    metrics_by_fold = []
    index = []
    
    for fold in range(1,11):

        # Dataset

        fold_name = f'fold_{fold}'
        yaml_path = os.path.join(Config.DATASET, fold_name, Config.YAML_FILE)
        project = os.path.join(model_dir, fold_name)


        # Train

        model = YOLO(f"./Weights/{model_size}")
        model.train(
            data = yaml_path,
            project = project,
            imgsz = Config.IMG_SIZE,
            # epochs = Config.EPOCHS,
            epochs = 1,
            batch = Config.BATCH_SIZE,
            device = Config.DEVICE
        )
        
        # Validate
        
        weights_dir = os.path.join(project, 'train', 'weights')
        
        best_weights = os.path.join(weights_dir, 'best.pt')
        
        model = YOLO(best_weights)
        
        val_metrics = model.val(
            data = yaml_path, 
            split = 'val', 
            project = project, 
        )
        
        confusion_matrix_keys = [f'CM({i},{j})' for i in range(3) for j in range(3)]
        confusion_matrix_values =  val_metrics.confusion_matrix.matrix.flatten()
        confusion_matrix_dict = {k:v for (k,v) in zip(confusion_matrix_keys, confusion_matrix_values)}

        metrics = val_metrics.results_dict
        metrics.update(confusion_matrix_dict)
        
        metrics_by_fold.append(metrics)
        index.append(fold_name)
        
        shutil.rmtree(weights_dir)
        
    
        # break # Fold
    
    # Aggregate Metrics
    
    metrics_by_fold = pd.DataFrame(metrics_by_fold, index=index)
    metrics_by_fold.loc['mean'] = metrics_by_fold.mean()
    
    metrics_path = os.path.join(model_dir, 'metrics.csv')
    metrics_by_fold.to_csv(metrics_path)
    
    # break # Model Size


# %%



