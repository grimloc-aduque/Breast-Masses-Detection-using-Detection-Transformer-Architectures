

import os
import shutil
from detr_config import Config
from detr_factory import DETRFactory

class FileManager:
    def __init__(self, detr_factory):
        self.model_name = detr_factory.get_model_name()
        self.dataset_dir = Config.DATASET
        self.logs_dir = Config.LOGS_DIR
        self.metrics_csv_file = Config.METRICS_FILE
    
    # Logs
    
    def clean_model_logs(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        print("Cleaning Logs: ", model_logs_dir)
        if os.path.exists(model_logs_dir):
            shutil.rmtree(model_logs_dir)
    
    # Setup: Validation - Testing
    
    def set_validation_setup(self, fold):
        self.validation_setup = True
        self.fold = fold
        self.fold_name = f'fold_{fold}'
        self.dataset_fold_dir = os.path.join(self.dataset_dir, self.fold_name)
        self.version = self.get_version()
        self.checkpoints_dir = self.get_checkpoints_dir()
        
    def set_testing_setup(self):
        self.validation_setup = False
        self.version = self.get_version()
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
    
    def _get_fold_version(self):
        return os.path.join(self.model_name, self.fold_name)
        
    def _get_test_version(self):
        return os.path.join(self.model_name, 'test')
    
    def get_version(self):
        if self.validation_setup:
            return self._get_fold_version()
        else:
            return self._get_test_version() 
    
    # Pretrained Checkpoints
        
    def get_checkpoints_dir(self):
        return os.path.join(self.logs_dir, self.version, 'checkpoints')
    
    def clean_checkpoints(self):
        print("Cleaning Checkpoints: ", self.checkpoints_dir)
        shutil.rmtree(self.checkpoints_dir)
        
    # Metrics
    
    def get_csv_metrics_path(self):
        return os.path.join(self.logs_dir, self.model_name, self.metrics_csv_file)
    
