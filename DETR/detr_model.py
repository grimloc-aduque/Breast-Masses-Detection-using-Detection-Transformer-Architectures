
import pytorch_lightning as pl
import torch


class DETRModel(pl.LightningModule):
    def __init__(self, detr_model):
        super().__init__()
        self.detr_model = detr_model
        self.save_hyperparameters()

    def forward(self, pixel_values):
        outputs = self.detr_model(pixel_values)
        return outputs

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch['labels']
        outputs = self.detr_model(pixel_values=pixel_values, labels=labels)
        self.log("train_loss", outputs.loss)
        for loss_name, loss in outputs.loss_dict.items():
            self.log(f"train_{loss_name}", loss.item())
        return outputs.loss
     
    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch['labels']
        outputs = self.detr_model(pixel_values=pixel_values, labels=labels)
        self.log("valid_loss", outputs.loss)
        for loss_name, loss in outputs.loss_dict.items():
            self.log(f"valid_{loss_name}", loss.item())
        return outputs.loss

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
    