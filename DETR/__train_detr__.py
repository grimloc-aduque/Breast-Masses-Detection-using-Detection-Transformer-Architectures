import argparse

from colorama import Fore
from detr_config import Config
from detr_data_source import DataSource
from detr_factory import DETRFactory
from detr_file_manager import FileManager
from detr_metrics import MetricsAggregator
from detr_model_evaluator import ModelEvaluator
from detr_model_loader import ModelLoader
from detr_model_trainer import ModelTrainer
from detr_plotter import Plotter

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

    # -----------------------
    # Hyperparameter Search
    # -----------------------
        
    for hyperparams in Config.HYPERPARAMS:
        detr_factory = DETRFactory(*hyperparams)    
        file_manager = FileManager(detr_factory)
        data_source = DataSource(detr_factory, file_manager)
        model_loader = ModelLoader(detr_factory, file_manager)
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
            model_evaluator = ModelEvaluator(best_model, detr_factory, plotter)

            for threshold in Config.THRESHOLDS:
                valid_metrics = model_evaluator.evaluate(valid_dataset, valid_loader, threshold)
                metrics_aggregator.add_metrics(threshold, valid_metrics)
            
            file_manager.clean_checkpoints()        
            
        
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
        model_evaluator = ModelEvaluator(best_model, detr_factory, plotter)
        
        for threshold in Config.THRESHOLDS:
            model_evaluator.evaluate(test_dataset, test_loader, threshold)


