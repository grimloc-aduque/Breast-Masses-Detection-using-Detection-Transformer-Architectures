
import os

import torchvision
from detr_config import Config
from torch.utils.data import DataLoader

from detr_file_manager import FileManager


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



class DataGenerator():
    
    def __init__(self, image_processor, file_manager:FileManager):
        self.image_processor = image_processor
        self.file_manager = file_manager
    
    
    def _get_dataset(self, dataset_dir):
        dataset = InBreastDataset(
            dataset_dir = dataset_dir,
            processor = self.image_processor
        )
        return dataset
            
    def _get_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = Config.BATCH_SIZE,
            collate_fn = lambda batch: 
                            collate_fn(batch, self.image_processor),
            shuffle = True
        )
        return dataloader
    
    def _get_dataset_dataloader(self, dataset_dir):
        dataset = self._get_dataset(dataset_dir)
        dataloader = self._get_dataloader(dataset)
        return dataset, dataloader
    
    
    def get_train_dataset_dataloader(self):
        train_dir = self.file_manager.get_train_dir()
        return self._get_dataset_dataloader(train_dir)
        
    def get_valid_dataset_dataloader(self):
        valid_dir = self.file_manager.get_valid_dir()
        return self._get_dataset_dataloader(valid_dir)
        
    def get_train_valid_dataset_dataloader(self):
        train_valid_dir = self.file_manager.get_train_valid_dir()
        return self._get_dataset_dataloader(train_valid_dir)
        
    def get_test_dataset_dataloader(self):
        test_dir = self.file_manager.get_test_dir()
        return self._get_dataset_dataloader(test_dir)
    
    
    
    
    
