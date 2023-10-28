
import argparse

from colorama import Fore
from ultralytics_config import Config
from ultralytics_file_manager import FileManager
from ultralytics_metrics import MetricsAggregator
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
        
        for fold in Config.FOLDS:

            # Dataset

            file_manager.set_validation_setup(fold)

            # Train

            model.train()
        
            # Validation - Threshold Optimization
            
            for threshold in Config.THRESHOLDS:
                valid_metrics = model.validate(threshold)
                metrics_aggregator.add_metrics(threshold, valid_metrics)

            file_manager.clean_weights()
        
        # Aggregate Metrics
        
        metrics_aggregator.finish_validation()
        metrics_aggregator.save_metrics()
        
        
        # -----------------------
        # Testing
        # -----------------------
        
        file_manager.set_testing_setup()
        model.train()
        
        for threshold in Config.THRESHOLDS:
            valid_metrics = model.validate(threshold)
            metrics_aggregator.add_metrics(threshold, valid_metrics)

