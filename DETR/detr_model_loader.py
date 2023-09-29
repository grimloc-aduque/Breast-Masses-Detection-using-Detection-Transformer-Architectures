
import os

import torch
from detr_model import DETRModel
from detr_file_manager import FileManager
from detr_factory import DETRFactory


class ModelLoader():
    
    def __init__(self, detr_factory:DETRFactory, file_manager:FileManager):
        self.detr_factory = detr_factory
        self.file_manager = file_manager
        
    def _load_detr(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('detr.', ''):state_dict[k] for k in state_dict.keys()}
        detr = self.detr_factory.new_empty_model()
        detr.load_state_dict(state_dict)
        return detr
    
    def load_best_model(self):
        checkpoints_dir = self.file_manager.get_checkpoints_dir()
        best_checkpoint = [f for f in os.listdir(checkpoints_dir) if 'last' not in f][0]
        checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
        print("Loading: ", checkpoint_path)
        detr = self._load_detr(checkpoint_path)
        model = DETRModel(detr)
        return model
    
    def new_pretrained_model(self):
        print('New Model: ', self.detr_factory.get_model_name())
        detr = self.detr_factory.new_pretrained_model()
        model = DETRModel(detr)
        return model