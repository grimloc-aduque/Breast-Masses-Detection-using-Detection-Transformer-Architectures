# %%
from detr_config import Config
from detr_model_trainer import ModelTrainer
from detr_model_evaluator import ModelEvaluator
from detr_model_loader import ModelLoader
from detr_file_manager import FileManager
from detr_metrics import MetricsAggregator
from detr_model_loader import ModelLoader
from detr_factory import DETRFactory
from detr_data_source import DataSource

# %%
# Config.set_local_settings()
Config.set_gpu_settings()

for hyperparams in Config.HYPERPARAMS:
    architecture, num_queries, transformer_layers = hyperparams
    detr_factory = DETRFactory(architecture, num_queries, transformer_layers)    
    file_manager = FileManager(detr_factory)
    data_source = DataSource(detr_factory, file_manager)
    model_loader = ModelLoader(detr_factory, file_manager)
    model_trainer = ModelTrainer(file_manager)
    metrics_aggregator = MetricsAggregator(file_manager)
    
    file_manager.clean_model_logs()

    # -----------------------
    # K-fold Cross Validation
    # -----------------------

    for fold in Config.FOLDS:

        file_manager.set_validation_setup(fold)

        # Model
        
        model = model_loader.new_pretrained_model()
        
        # Dataset
                
        train_dataset, train_loader = data_source.get_train_dataset_dataloader()
        valid_dataset, valid_loader = data_source.get_valid_dataset_dataloader()

        # Training

        model_trainer.fit(model, train_loader, valid_loader)
        
        # Validation - Threshold Optimization
        
        best_model = model_loader.load_best_model()
        model_evaluator = ModelEvaluator(best_model, detr_factory)

        for threshold in Config.THRESHOLDS:
            valid_metrics = model_evaluator.evaluate(valid_dataset, valid_loader, threshold)
            metrics_aggregator.add_metrics(threshold, valid_metrics)
        
        best_model.to('cpu')
        file_manager.clean_checkpoints()        
        
    
    metrics_aggregator.finish_validation()
    metrics_aggregator.save_metrics()
    
    # -----------------------
    # Testing
    # -----------------------
    
    # Model
    
    model = model_loader.new_pretrained_model()

    # Dataset
    
    file_manager.set_testing_setup()

    train_valid_dataset, train_valid_loader = data_source.get_train_valid_dataset_dataloader()
    test_dataset, test_loader = data_source.get_test_dataset_dataloader()
    
    # Training
    
    model_trainer.fit(model, train_valid_loader, test_loader)
    

# %%



