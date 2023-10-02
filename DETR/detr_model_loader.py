
import os

import torch
from detr_model import DETRModel
from detr_file_manager import FileManager
from detr_factory import DETRFactory
from detr_config import Config


class ModelLoader():
    
    def __init__(self, detr_factory:DETRFactory, file_manager:FileManager):
        self.detr_factory = detr_factory
        self.file_manager = file_manager
    
    def load_best_model(self):
        checkpoints_dir = self.file_manager.get_checkpoints_dir()
        best_checkpoint = [f for f in os.listdir(checkpoints_dir) if 'last' not in f][0]
        checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        print("Loading Model: ", checkpoint_path)
        
        detr = self.detr_factory.new_empty_model()
        model = DETRModel(detr=detr)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(Config.DEVICE)
        return model
    
    def new_pretrained_model(self):
        print('New Model: ', self.detr_factory.get_model_name())
        detr = self.detr_factory.new_pretrained_model()
        model = DETRModel(detr)
        return model
