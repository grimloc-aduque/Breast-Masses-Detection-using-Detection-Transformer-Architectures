# %%
from detr_config import Config
from detr_factory import DETRFactory
from detr_file_manager import FileManager
from detr_model_loader import ModelLoader
from detr_data_source import DataSource
from detr_model import DETRModel
import os
import torch

from transformers import (DeformableDetrConfig,
                          DeformableDetrForObjectDetection,
                          DeformableDetrImageProcessor, DetrConfig,
                          DetrForObjectDetection, DetrImageProcessor)


Config.set_local_settings()

# %%
hyperparameters = ('DETR', 100, 6)
detr_factory = DETRFactory(*hyperparameters)
file_manager = FileManager(detr_factory)
model_loader = ModelLoader(detr_factory, file_manager)
data_source = DataSource(detr_factory, file_manager)
file_manager.set_validation_setup(fold=1)
model = model_loader.load_best_model()


