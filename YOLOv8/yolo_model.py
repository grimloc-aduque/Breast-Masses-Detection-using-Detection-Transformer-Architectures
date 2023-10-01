

from ultralytics import YOLO
from yolo_config import Config
from yolo_file_manager import FileManager


class YoloModel:
    def __init__(self, file_manager:FileManager):
        self.file_manager = file_manager
        pass
    
    def train(self):
        yaml_path = self.file_manager.get_yaml_path()
        project = self.file_manager.get_train_project()
        weights = self.file_manager.get_pretrained_weights()
        print("Loading Model: ", weights)
        print("Training on YAML: ", yaml_path)
        model = YOLO(weights)
        model.train(
            data = yaml_path,
            project = project,
            imgsz = Config.IMG_SIZE,
            epochs = Config.EPOCHS,
            batch = Config.BATCH_SIZE,
            device = Config.DEVICE
        )
        
        
    def validate(self, threshold):
        yaml_path = self.file_manager.get_yaml_path()
        project = self.file_manager.get_validation_project(threshold)
        weights = self.file_manager.get_best_weights()
        print("Loading Model: ", weights)
        print("Validating on YAML: ", yaml_path)
        print("Threshold: ", threshold)
        best_model = YOLO(weights)
        metrics = best_model.val(
            data = yaml_path, 
            split = 'val', 
            conf = threshold, 
            project = project,
        )
        return metrics.results_dict