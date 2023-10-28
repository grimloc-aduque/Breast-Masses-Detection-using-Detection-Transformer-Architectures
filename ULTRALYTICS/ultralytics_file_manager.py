

import os
import shutil

from ultralytics_config import Config


class FileManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = os.path.join(Config.RUNS_DIR, Config.DATASET, model_name)
        self.pretrained_weights = os.path.join('Weights', self.model_name)
        self.csv_metrics_path = os.path.join(self.model_dir, Config.METRICS_FILE)
        
        
    def set_validation_setup(self, fold):
        self.fold = fold
        self.fold_name = f'fold_{fold}'
        self.yaml_path = os.path.join(Config.DATASET, self.fold_name, Config.YAML_FILE)
        self.project = os.path.join(self.model_dir, self.fold_name)
        self.weights_path = os.path.join(self.project, 'train', 'weights')
        
    def set_testing_setup(self):
        self.yaml_path = os.path.join(Config.DATASET, Config.YAML_FILE)
        self.project = os.path.join(self.model_dir, 'test')
        self.weights_path = os.path.join(self.project, 'train', 'weights')
    
    
    def get_fold_name(self):
        return self.fold_name
    
    # Clean
    
    def clean_model_runs(self):
        print("Cleaning Runs: ", self.model_dir)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
            
    def clean_weights(self):
        print("Cleaning Weights: ", self.weights_path)
        shutil.rmtree(self.weights_path)
        
    # Projects to log
        
    def get_train_project(self):
        return self.project
    
    def get_validation_project(self, threshold):
        threshold_name = f'conf={threshold}'
        validation_project = os.path.join(self.project, threshold_name)
        return validation_project
    
    def get_yaml_path(self):
        return self.yaml_path
    
    def get_csv_metrics_path(self):
        return self.csv_metrics_path
    
    # Weights
    
    def get_pretrained_weights(self):
        return self.pretrained_weights
    
    def get_best_weights(self):
        best_weights = os.path.join(self.weights_path,'best.pt')
        return best_weights
