import argparse

from colorama import Fore
from faster_config import Config
from faster_data_source import DataSource
from faster_file_manager import FileManager
from faster_metrics import MetricsAggregator
from faster_model_evaluator import ModelEvaluator
from faster_model_loader import ModelLoader
from faster_plotter import Plotter
from faster_trainer import ModelTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true") 


if __name__ == '__main__':
    
    # Load Configuration
    
    args = parser.parse_args() 
    if args.local:
        print(Fore.YELLOW, "Loading Configuration: Local", Fore.WHITE)
        Config.set_local_settings()
    else:
        print(Fore.YELLOW, "Loading Configuration: Benchmark", Fore.WHITE)
        Config.set_benchmark_settings()
    print(Fore.YELLOW, "Runnig on: ", Config.DEVICE, Fore.WHITE)
        
    file_manager = FileManager()
    data_source = DataSource(file_manager)
    model_loader = ModelLoader(file_manager)
    model_trainer = ModelTrainer(file_manager)
    metrics_aggregator = MetricsAggregator(file_manager)
    plotter = Plotter(file_manager, metrics_aggregator)
    
    file_manager.clean_model_logs()

    # -----------------------
    # K-fold Cross Validation
    # -----------------------
    
    data_source.start_kfold()
    
    for _ in range(Config.FOLDS):

        # Model
        
        model = model_loader.new_pretrained_model()
        
        # Dataset
        datasets, dataloaders = data_source.next_fold()
        train_dataset, valid_dataset = datasets
        train_loader, valid_loader = dataloaders

        # Training

        model_trainer.fit(model, train_loader, valid_loader)
        
        # Validation - Threshold Optimization
        
        best_model = model_loader.load_best_model()
        model_evaluator = ModelEvaluator(best_model, plotter)

        for threshold in Config.THRESHOLDS:
            valid_metrics = model_evaluator.evaluate(valid_dataset, valid_loader, threshold, save_plots=False)
            metrics_aggregator.add_metrics(threshold, valid_metrics)
        
        file_manager.clean_checkpoints()        
        
    # Aggregate Validation Metrics
    
    metrics_aggregator.finish_validation()
    metrics_aggregator.save_metrics()
    plotter.plot_metrics()

    # -----------------------
    # Testing
    # -----------------------
    
    # Model
    
    model = model_loader.new_pretrained_model()

    # Dataset
    
    datasets, dataloaders = data_source.testing()
    train_dataset, test_dataset = datasets
    train_loader, test_loader = dataloaders
    
    # Training
    
    model_trainer.fit(model, train_loader, test_loader)
    
    # Testing
        
    best_model = model_loader.load_best_model()
    model_evaluator = ModelEvaluator(best_model, plotter)
    
    for threshold in Config.THRESHOLDS:
        model_evaluator.evaluate(test_dataset, test_loader, threshold, save_plots=True)


