
import pytorch_lightning as pl
import torch


class DETRModel(pl.LightningModule):
    def __init__(self, detr_model):
        super().__init__()
        self.detr_model = detr_model
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        outputs = self.detr_model(
            pixel_values = pixel_values, 
            pixel_mask = pixel_mask
        )
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self.detr_model(
            pixel_values = batch["pixel_values"], 
            pixel_mask = batch["pixel_mask"], 
            labels = batch["labels"]
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        for loss_name, loss in loss_dict.items():
            self.log(f"train_{loss_name}", loss.item())
        return loss
     
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)  
        self.log("valid_loss", loss)
        for loss_name, loss in loss_dict.items():
            self.log(f"valid_{loss_name}", loss.item())
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
    