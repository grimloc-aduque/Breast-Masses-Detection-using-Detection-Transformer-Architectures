from detr_config import Config
from detr_data_source import DataSource
from detr_factory import DETRFactory
from detr_file_manager import FileManager
from detr_metrics import MetricsAggregator
from detr_model_evaluator import ModelEvaluator
from detr_model_loader import ModelLoader
from detr_model_trainer import ModelTrainer
from detr_plotter import Plotter

if __name__ == '__main__':
    
    Config.set_local_settings()
    # Config.set_gpu_settings()
    
    Config.DATASET = 'InBreast-COCO'
    Config.LOGS_DIR = '../Otros/DETR_logs_data_augmentation'
    hyperparams = ('DETR', 256, 100, 6)
    detr_factory = DETRFactory(*hyperparams)    
    file_manager = FileManager(detr_factory)
    data_source = DataSource(detr_factory, file_manager)
    model_loader = ModelLoader(detr_factory, file_manager)
    model_trainer = ModelTrainer(file_manager)
    metrics_aggregator = MetricsAggregator(file_manager)
    plotter = Plotter(file_manager, metrics_aggregator)
        
    datasets, dataloaders = data_source.testing()
    train_dataset, test_dataset = datasets
    train_loader, test_loader = dataloaders        
    best_model = model_loader.load_best_model()
    model_evaluator = ModelEvaluator(best_model, detr_factory, plotter)
    
    # model_evaluator.evaluate(test_dataset, test_loader, threshold=0.1)
    # model_evaluator.evaluate(test_dataset, test_loader, threshold=0.4)
    
    print(len(test_loader))