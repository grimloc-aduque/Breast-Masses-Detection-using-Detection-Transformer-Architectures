
import pytorch_lightning as pl
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import (FasterRCNN,
                                                      FastRCNNPredictor)


class FasterRCNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 1)
        self.faster_rcnn = model

    def forward(self, pixel_values):
        self.faster_rcnn.eval()
        return self.faster_rcnn(pixel_values)

    def training_step(self, batch, batch_idx):
        self.faster_rcnn.train()
        loss_dict = self.faster_rcnn(batch['pixel_values'], batch['labels'])
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train_{loss_name}", loss_value.item())
        return loss
     
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.faster_rcnn.train()
            loss_dict = self.faster_rcnn(batch['pixel_values'], batch['labels'])
            loss = sum(loss for loss in loss_dict.values())
            self.log("valid_loss", loss)
            for loss_name, loss_value in loss_dict.items():
                self.log(f"valid_{loss_name}", loss_value.item())
            self.faster_rcnn.eval()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

