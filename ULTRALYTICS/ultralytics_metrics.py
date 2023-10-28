
import pandas as pd
from ultralytics_config import Config
from ultralytics_file_manager import FileManager

metrics_names = [
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)',
    'fitness'
]

class MetricsAggregator():
    
    def __init__(self, file_manager:FileManager):
        self.file_manager = file_manager
        self.metrics = pd.DataFrame(columns=metrics_names)
        self.metrics.index.name = 'threshold'
        self.metrics_by_threshold = {
            t:pd.DataFrame(columns=metrics_names) for t in Config.THRESHOLDS
        }
    
    def add_metrics(self, threshold, valid_metrics):
        threshold_metrics = self.metrics_by_threshold[threshold]
        fold_name = self.file_manager.get_fold_name()
        threshold_metrics.loc[fold_name] = pd.Series(valid_metrics)       
        
    def finish_validation(self):
        for threshold in Config.THRESHOLDS:
            threshold_metrics = self.metrics_by_threshold[threshold]
            mean_metrics = threshold_metrics.mean()
            self.metrics.loc[threshold] = mean_metrics
        
    def save_metrics(self):
        metrics_path = self.file_manager.get_csv_metrics_path()
        self.metrics.to_csv(metrics_path)
        
