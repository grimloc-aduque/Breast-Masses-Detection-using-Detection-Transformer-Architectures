
import os

import torch
import torchvision


class InBreastDataset(torchvision.datasets.CocoDetection):
    def __init__(self, images_path: str, processor):
        annotation_file_path = os.path.join(images_path, '_annotations.coco.json')
        super().__init__(images_path, annotation_file_path)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        annotations  = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        return pixel_values, labels


def collate_fn(batch):
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = pixel_values
    batch['labels'] = labels
    return batch

