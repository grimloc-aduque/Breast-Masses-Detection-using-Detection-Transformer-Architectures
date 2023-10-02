
import os

import torchvision
import albumentations as A
import numpy as np
import torch

class InBreastDataset(torchvision.datasets.CocoDetection):
    
    def __init__(self, dataset_dir, processor):
        annotation_file_path = os.path.join(dataset_dir, '_annotations.coco.json')
        print("Loading Annotations from: ", annotation_file_path)
        super().__init__(dataset_dir, annotation_file_path)
        self.processor = processor
        self.transform = A.Compose([
            # A.Normalize(),
            A.PixelDropout(),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.2),
            A.Affine(
                scale = (0.8, 1.2),
                translate_percent = {
                    'y': (0, 0.2)
                },
                rotate = (-15, 15),
                shear = (-10, 10),
                p = 0.5
            )],
            bbox_params = A.BboxParams(
                format = 'coco',
                label_fields = ['category_ids'],
                min_visibility = 0.25
            )
        )


    def data_augmentation(self, image, annotations):
        image = np.flip(np.array(image), -1)
        bboxes = [ann['bbox'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]
        transformed = self.transform(
            image = image,
            bboxes = bboxes,
            category_ids = category_ids,
        )
        transformed_image = torch.tensor(transformed['image']).flip(-1).permute(2, 0, 1)
        transformed_annotations = [
            {
                'id': annotations[0]['id'],
                'image_id': annotations[0]['image_id'],
                'category_id': transformed['category_ids'][i],
                'bbox':  transformed['bboxes'][i],
                'area': transformed['bboxes'][i][2] * transformed['bboxes'][i][3],
                'segmentation': [],
                'iscrowd': 0
            }
            for i in range(len(transformed['bboxes']))
        ]
        return transformed_image, transformed_annotations
        

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        image, annotations = self.data_augmentation(image, annotations)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': annotations}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding['pixel_values'][0]
        target = encoding["labels"][0]
        return pixel_values, target


def collate_fn(batch, image_processor):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch
