
import pytorch_lightning as pl
import torch
from detr_config import Config
from transformers import (DeformableDetrConfig,
                          DeformableDetrForObjectDetection,
                          DeformableDetrImageProcessor, DetrConfig,
                          DetrForObjectDetection, DetrImageProcessor)


class DETRModel(pl.LightningModule):
    def __init__(self, detr):
        super().__init__()
        self.detr = detr
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        outputs = self.detr(
            pixel_values = pixel_values, 
            pixel_mask = pixel_mask
        )
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self.detr(
            pixel_values = batch["pixel_values"], 
            pixel_mask = batch["pixel_mask"], 
            labels = batch["labels"]
        )
        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train_{loss_name}", loss_value.item())
        return loss
     
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)  
        self.log("valid_loss", loss)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"valid_{loss_name}", loss_value.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            { 
                "params": [
                    param for name, param in self.named_parameters()
                    if "backbone" not in name and param.requires_grad
                ]
            },
            {
                "params": [
                    param for name, param in self.named_parameters() 
                    if "backbone" in name and param.requires_grad
                ],
                "lr": 1e-5
            }
        ]
        print(self.named_parameters())
        optimizer = torch.optim.AdamW(param_dicts, lr=1e-4)
        return optimizer
    


class ModelGenerator:
    
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

    def get_model(self):
        config = self.DETR_CONFIG.from_pretrained(
            pretrained_model_name_or_path = self.checkpoint,
            num_labels = Config.NUM_CLASSES,
            id2label = {0:'Mass'}, 
            label2id = {'Mass': 0},
            encoder_layers = self.transformer_layers,
            decoder_layers = self.transformer_layers,
            num_queries = self.num_queries,
        )
        
        detr = self.DETR.from_pretrained(
            pretrained_model_name_or_path = self.checkpoint,
            config = config,
            ignore_mismatched_sizes = True
        )
        
        model = DETRModel(detr=detr)
        return model
    
    
    def get_image_processor(self):
        image_processor = self.IMG_PROCESSOR.from_pretrained(
            pretrained_model_name_or_path = self.checkpoint
        )
        return image_processor
    
