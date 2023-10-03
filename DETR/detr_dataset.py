
import os

import albumentations as A
import numpy as np
import torch
import torchvision


class InBreastDataset(torchvision.datasets.CocoDetection):
    
    def __init__(self, dataset_dir, processor, data_augmentation):
        annotation_file_path = os.path.join(dataset_dir, '_annotations.coco.json')
        print("Loading Annotations from: ", annotation_file_path)
        super().__init__(dataset_dir, annotation_file_path)
        self.processor = processor
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
                    'area': area,
                    'segmentation': [],
                    'iscrowd': 0
                }
            )
        return transformed_image, transformed_annotations
        

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        if self.data_augmentation:
            image, annotations = self.image_transformation(image, annotations)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.processor(images=image, annotations=annotations, return_tensors="pt")
        image = encoding['pixel_values'][0]
        target = encoding["labels"][0]
        return image, target


def collate_fn(batch, image_processor):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch
