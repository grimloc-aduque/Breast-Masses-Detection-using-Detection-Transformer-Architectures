
import pandas as pd
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
        self.metrics_df = pd.DataFrame(columns=metrics_names)
        
    def add_valid_metrics(self, valid_metrics):
        fold_name = self.file_manager.get_fold_name()
        self.metrics_df.loc[fold_name] = pd.Series(valid_metrics)
        
    def calculate_valid_mean(self):
        self.metrics_df.loc['mean'] = self.metrics_df.mean()
    
    def add_test_metrics(self, test_metrics):
        self.metrics_df.loc['test'] =  pd.Series(test_metrics)
        
    def save_metrics(self):
        metrics_path = self.file_manager.get_metrics_path()
        self.metrics_df.to_csv(metrics_path)