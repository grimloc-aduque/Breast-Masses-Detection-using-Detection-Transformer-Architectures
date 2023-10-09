

import os
import shutil

from detr_config import Config


class FileManager:
    def __init__(self, detr_factory):
        self.model_name = detr_factory.get_model_name()
        self.dataset_dir = Config.DATASET
        self.logs_dir = Config.LOGS_DIR
        self.metrics_csv_file = Config.METRICS_FILE
        self.metrics_plot_name = Config.METRIC_PLOT
    
    # Logs
    
    def clean_model_logs(self):
        self.model_logs_dir = os.path.join(self.logs_dir, self.dataset_dir, self.model_name)
        print("Cleaning Logs: ", self.model_logs_dir)
        if os.path.exists(self.model_logs_dir):
            shutil.rmtree(self.model_logs_dir)
    
    # Setup: Validation - Testing
    
    def set_validation_setup(self, fold):
        self.fold_name = f'fold_{fold}'
        self.version = self._get_fold_version()
        self.version_dir = os.path.join(self.logs_dir, self.version)
        self.checkpoints_dir = self.get_checkpoints_dir()
        
    def set_testing_setup(self):
        self.version = self._get_test_version()
        self.version_dir = os.path.join(self.logs_dir, self.version)
        self.checkpoints_dir = self.get_checkpoints_dir()
    
    def get_fold_name(self):
        return self.fold_name

    # Dataset Directories

    def get_train_dir(self):
        return os.path.join(self.dataset_dir, 'train')
        
    def get_test_dir(self):
        return os.path.join(self.dataset_dir, 'test')

    # Logs Dir

    # Versions for Training
    
    def _get_fold_version(self):
        return os.path.join(self.dataset_dir, self.model_name, self.fold_name)
        
    def _get_test_version(self):
        return os.path.join(self.dataset_dir, self.model_name, 'test')
    
    def get_version(self):
        return self._get_fold_version() if self.validation_setup else self._get_test_version()
    
    # Pretrained Checkpoints
        
    def get_checkpoints_dir(self):
        return os.path.join(self.version_dir, 'checkpoints')
    
    def clean_checkpoints(self):
        print("Cleaning Checkpoints: ", self.checkpoints_dir)
        shutil.rmtree(self.checkpoints_dir)
        
    # Metrics and plots
    
    def get_detection_plot_path(self, threshold, batch_id):
        plots_threshold_dir = os.path.join(self.version_dir, f'threshold={threshold}')
        os.makedirs(plots_threshold_dir, exist_ok=True)
        file_name = f'batch_{batch_id}.png'
        return os.path.join(plots_threshold_dir, file_name)

    def get_metrics_plot_path(self):
        return os.path.join(self.model_logs_dir, self.metrics_plot_name)
    
    def get_csv_metrics_path(self):
        return os.path.join(self.model_logs_dir, self.metrics_csv_file)
    
