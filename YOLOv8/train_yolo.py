# %%
from yolo_config import Config
from yolo_file_manager import FileManager
from yolo_model import YoloModel
from yolo_metrics import MetricsAggregator

# %%
# Config.set_local_settings()
Config.set_gpu_settings()

# -----------------------
# K-fold Cross Validation
# -----------------------

for model_size in Config.MODEL_SIZES:
    
    file_manager = FileManager(model_size)
    yolo_model = YoloModel(file_manager)
    metrics_aggregator = MetricsAggregator(file_manager)
    
    file_manager.clean_model_runs()
    
    for fold in Config.FOLDS:

        # Dataset

        file_manager.set_validation_setup(fold)

        # Train

        yolo_model.train()
    
        # Validation - Threshold Optimization
        
        for threshold in Config.THRESHOLDS:
            valid_metrics = yolo_model.validate(threshold)
            metrics_aggregator.add_metrics(threshold, valid_metrics)

        file_manager.clean_weights()
    
    # Aggregate Metrics
    
    metrics_aggregator.finish_validation()
    metrics_aggregator.save_metrics()
    
    
    # -----------------------
    # Testing
    # -----------------------
    
    file_manager.set_testing_setup()
    yolo_model.train()


# %%



