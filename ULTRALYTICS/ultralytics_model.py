

from ultralytics_config import Config
from ultralytics_file_manager import FileManager

from ultralytics import NAS, RTDETR, YOLO


class UltralyticsModel:
    def __init__(self, file_manager:FileManager):
        self.file_manager = file_manager
        pass
    
    def load_model(self, weights):
        if 'yolov8' in weights or 'yolov5' in weights:
            model = YOLO(weights)
        elif 'rtdetr' in weights:
            model = RTDETR(weights)
        # elif 'yolo_nas' in weights:
        #     model = NAS(weights)
        return model
    
    def train(self):
        yaml_path = self.file_manager.get_yaml_path()
        project = self.file_manager.get_train_project()
        weights = self.file_manager.get_pretrained_weights()
        print("Loading Model: ", weights)
        print("Training on YAML: ", yaml_path)
        model = self.load_model(weights)
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
        # best_model = YOLO(weights)
        best_model = self.load_model(weights)
        metrics = best_model.val(
            data = yaml_path, 
            split = 'val', 
            conf = threshold, 
            project = project,
        )
        return metrics.results_dict