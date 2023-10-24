
import os

import torch
from colorama import Fore
from faster_config import Config
from faster_file_manager import FileManager
from faster_model import FasterRCNNModel


class ModelLoader():
    
    def __init__(self, file_manager:FileManager):
        self.file_manager = file_manager
    
    def load_best_model(self):
        checkpoints_dir = self.file_manager.get_checkpoints_dir()
        best_checkpoint = [f for f in os.listdir(checkpoints_dir) if 'last' not in f][0]
        checkpoint_path = os.path.join(checkpoints_dir, best_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        print(Fore.LIGHTBLUE_EX, "Loading Model: ", checkpoint_path, Fore.WHITE)
        
        model = FasterRCNNModel()
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(Config.DEVICE)
        return model
    
    def new_pretrained_model(self):
        print(Fore.LIGHTBLUE_EX, 'New Model: ', Config.MODEL_NAME, Fore.WHITE)
        model = FasterRCNNModel()
        return model