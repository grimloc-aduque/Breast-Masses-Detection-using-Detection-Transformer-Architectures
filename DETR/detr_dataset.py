
import os

import torchvision


class InBreastDataset(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_dir, processor):
        annotation_file_path = os.path.join(dataset_dir, '_annotations.coco.json')
        print("Loading Annotations from: ", annotation_file_path)
        super().__init__(dataset_dir, annotation_file_path)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
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
