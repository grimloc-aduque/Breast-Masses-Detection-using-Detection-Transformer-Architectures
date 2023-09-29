

import os
import shutil
from detr_config import Config

class FileManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset_dir = Config.DATASET
        self.logs_dir = Config.LOGS_DIR
        self.metrics_file = Config.METRICS_FILE
    
    # Logs
    
    def clean_model_logs(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if os.path.exists(model_logs_dir):
            shutil.rmtree(model_logs_dir)
    
    # Setup: Validation - Training
    
    def set_validation_fold_setup(self, fold):
        self.fold = fold
        self.fold_name = f'fold_{fold}'
        self.dataset_fold_dir = os.path.join(self.dataset_dir, self.fold_name)
        self.version = self.get_fold_version()
        self.checkpoints_dir = self.get_checkpoints_dir()
        
    def set_testing_setup(self):
        self.version = self.get_test_version()
        self.checkpoints_dir = self.get_checkpoints_dir()
    
    def get_fold_name(self):
        return self.fold_name

    # Dataset Directories
    
    def get_train_dir(self):
        return os.path.join(self.dataset_fold_dir, 'train')

    def get_valid_dir(self):
        return os.path.join(self.dataset_fold_dir, 'valid')
        
    def get_train_valid_dir(self):
        return os.path.join(self.dataset_dir, 'train_valid')
        
    def get_test_dir(self):
        return os.path.join(self.dataset_dir, 'test')

    # Versions for Training
    
    def get_fold_version(self):
        return os.path.join(self.model_name, self.fold_name)
        
    def get_test_version(self):
        return os.path.join(self.model_name, 'test')
    
    # Pretrained Checkpoints
        
    def get_checkpoints_dir(self):
        return os.path.join(self.logs_dir, self.version, 'checkpoints')
    
    def clean_checkpoints(self):
        shutil.rmtree(self.checkpoints_dir)
        
    # Metrics
    
    def get_metrics_path(self):
        return os.path.join(self.logs_dir, self.model_name, self.metrics_file)
    
