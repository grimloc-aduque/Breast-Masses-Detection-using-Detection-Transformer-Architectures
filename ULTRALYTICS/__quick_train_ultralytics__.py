
import argparse

import pandas as pd
from colorama import Fore
from ultralytics_config import Config
from ultralytics_file_manager import FileManager
from ultralytics_metrics import MetricsAggregator, metrics_names
from ultralytics_model import UltralyticsModel

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true") 


if __name__ == '__main__':

    args = parser.parse_args()
    if args.local:
        print(Fore.YELLOW, "Loading Configuration: Local", Fore.WHITE)
        Config.set_local_settings()
    else:
        print(Fore.YELLOW, "Loading Configuration: Benchmark", Fore.WHITE)
        Config.set_benchmark_settings()
    print(Fore.YELLOW, "Runnig on: ", Config.DEVICE, Fore.WHITE)

    # -----------------------
    # K-fold Cross Validation
    # -----------------------

    for model_name in Config.MODEL_NAMES:
        
        file_manager = FileManager(model_name)
        model = UltralyticsModel(file_manager)
        metrics_aggregator = MetricsAggregator(file_manager)
        
        file_manager.clean_model_runs()
        
        # -----------------------
        # Testing
        # -----------------------
        
        file_manager.set_testing_setup()
        model.train()
        
        metrics_by_threshold = pd.DataFrame(columns=metrics_names)
        for threshold in Config.THRESHOLDS:
            metrics = model.validate(threshold)
            metrics_by_threshold.loc[threshold] = pd.Series(metrics)
            
        metrics_by_threshold.index.name = 'threshold'
        metrics_by_threshold.to_csv(file_manager.get_csv_metrics_path())
