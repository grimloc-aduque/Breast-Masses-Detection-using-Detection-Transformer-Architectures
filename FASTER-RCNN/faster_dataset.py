
import os

import albumentations as A
import numpy as np
import torch
import torchvision
from colorama import Fore
from torchvision import transforms


class FasterRCNNDataset(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_dir, data_augmentation):
        annotation_file_path = os.path.join(dataset_dir, '_annotations.coco.json')
        print(Fore.GREEN, "Loading Annotations: ", annotation_file_path, Fore.WHITE)
        super().__init__(dataset_dir, annotation_file_path)
        anns = {}
        for ann_id, ann in self.coco.anns.items():
            ann['category_id'] = 1
            anns[ann_id] = ann
        self.coco.anns = anns
        self.data_augmentation = data_augmentation
        self.transform = A.Compose([
            # A.Normalize(),
            A.PixelDropout(),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.2),
            A.Affine(
                scale = (0.75, 1.25),
                translate_percent = {
                    'x': (-0.05, 0.05),
                    'y': (0, 0.2)
                },
                rotate = (-15, 15),
                shear = (-8, 8),
                p = 0.6)
            ],
            bbox_params = A.BboxParams(
                format = 'coco',
                label_fields = ['category_ids', 'annotations_ids'],
                min_visibility = 0.3
            ),
        )


    def image_transformation(self, image, annotations):
        image = np.flip(np.array(image), -1)
        bboxes = [ann['bbox'] for ann in annotations]
        annotations_ids = [ann['id'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]
        transformed = self.transform(
            image = image,
            bboxes = bboxes,
            annotations_ids = annotations_ids,
            category_ids = category_ids,
        )
        transformed_image = torch.tensor(transformed['image']).flip(-1).permute(2, 0, 1)
        
        transformed_annotations = []
        image_id = annotations[0]['image_id']
        num_annotations = len(transformed['bboxes'])
        for i in range(num_annotations):
            bbox = transformed['bboxes'][i]
            annotation_id = transformed['annotations_ids'][i]
            category_id = transformed['category_ids'][i]
            w, h = bbox[2], bbox[3]
            area = w * h
            transformed_annotations.append(
                {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox':  bbox,
                    'area': area
                }
            )
        return transformed_image, transformed_annotations
        

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx) # bbox: (x_min, y_min, w, h)
        if self.data_augmentation:
            image, annotations = self.image_transformation(image, annotations)
        
        image = np.array(image)/ 255.0
        image = torch.Tensor(image)
        if image.shape[0] != 3:
            image = image.permute(2,0,1)
            
        # bbox: (x_min, y_min, w, h)
        # boxes: (x_min, y_min, x_max, y_max)
        if len(annotations) == 0:
            target = {
                'boxes': torch.Tensor([[0,0,0.1,0.1]]),
                'labels': torch.Tensor([0]).type(torch.int64),
                'area': torch.Tensor([0.01]),
                'iscrowd': torch.Tensor([0]),
                'image_id': torch.Tensor([idx]).type(torch.int64)
            }
        else:
            target = {
                'boxes': torch.Tensor([
                    [
                        ann['bbox'][0], 
                        ann['bbox'][1],
                        (ann['bbox'][0] + ann['bbox'][2]),
                        (ann['bbox'][1] + ann['bbox'][3]), 
                    ]
                    for ann in annotations
                ]),
                'labels': torch.Tensor([ann['category_id'] for ann in annotations]).type(torch.int64),
                'area': torch.Tensor([ann['area'] for ann in annotations]),
                'iscrowd': torch.Tensor([0 for _ in annotations]),
                'image_id': torch.Tensor([idx]).type(torch.int64)
            }
        
        # target = {k:torch.Tensor(v) for k,v in target.items()}
        return image, target


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = labels
    return batch
