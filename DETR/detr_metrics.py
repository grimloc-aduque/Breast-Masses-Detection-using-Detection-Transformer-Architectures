
import pandas as pd

from detr_config import Config
from detr_file_manager import FileManager

metrics_names =  [
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
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
        
    def load_metrics(self):
        metrics_path = self.file_manager.get_csv_metrics_path()
        return pd.read_csv(metrics_path, index_col='threshold')
