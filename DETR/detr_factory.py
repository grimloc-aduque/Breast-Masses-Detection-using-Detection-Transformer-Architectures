
import pytorch_lightning as pl
import torch
from detr_config import Config
from transformers import (DeformableDetrConfig,
                          DeformableDetrForObjectDetection,
                          DeformableDetrImageProcessor, DetrConfig,
                          DetrForObjectDetection, DetrImageProcessor)

class DETRFactory:
    
    def __init__(self, architecture, num_queries, transformer_layers):        
        # DETR or Deformable DETR
        if architecture == 'DETR':
            self.IMG_PROCESSOR = DetrImageProcessor
            self.DETR_CONFIG = DetrConfig
            self.DETR = DetrForObjectDetection
            self.checkpoint = 'facebook/detr-resnet-50'
        elif architecture == 'D-DETR':
            self.IMG_PROCESSOR = DeformableDetrImageProcessor
            self.DETR_CONFIG = DeformableDetrConfig
            self.DETR = DeformableDetrForObjectDetection
            self.checkpoint = 'SenseTime/deformable-detr'
        else:
            raise Exception('Architecture not suported')
        
        self.architecture = architecture
        self.transformer_layers = transformer_layers
        self.num_queries = num_queries
        self.init_model_name()
        self.init_config()
        
    # Init
        
    def init_model_name(self):
        self.model_name = [
            f'model={self.architecture}',
            f'queries={self.num_queries}',
            f'layers={self.transformer_layers}'
        ]
        self.model_name = '_'.join(self.model_name)        
        
    def init_config(self):
        self.config = self.DETR_CONFIG(
            pretrained_model_name_or_path = self.checkpoint,
            num_labels = Config.NUM_CLASSES,
            id2label = {0:'Mass'}, 
            label2id = {'Mass': 0},
            encoder_layers = self.transformer_layers,
            decoder_layers = self.transformer_layers,
            num_queries = self.num_queries,
        )

    # DETR Factory Methods

    def new_pretrained_model(self):        
        detr = self.DETR.from_pretrained(
            pretrained_model_name_or_path = self.checkpoint,
            config = self.config,
            ignore_mismatched_sizes = True
        )
        return detr
    
    def new_empty_model(self):
        detr = self.DETR(self.config)
        return detr
    
    def new_image_processor(self):
        image_processor = self.IMG_PROCESSOR.from_pretrained(
            pretrained_model_name_or_path = self.checkpoint
        )
        return image_processor
    
    # Getters    
    
    def get_model_name(self):
        return self.model_name
    
    def get_config(self):
        return self.config
    
