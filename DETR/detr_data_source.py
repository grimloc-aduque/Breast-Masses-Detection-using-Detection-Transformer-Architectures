

from detr_config import Config
from detr_dataset import InBreastDataset, collate_fn
from detr_file_manager import FileManager
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy

class DataSource():
    
    def __init__(self, detr_factory, file_manager:FileManager):
        self.image_processor = detr_factory.new_image_processor()
        self.file_manager = file_manager
    
    
    def _get_dataset(self, dataset_dir, data_augmentation):
        dataset = InBreastDataset(
            dataset_dir = dataset_dir,
            processor = self.image_processor,
            data_augmentation = data_augmentation
        )
        return dataset
            
    def get_dataloader(self, dataset, dataset_ids=[], shuffle=False):
        if len(dataset_ids)>0:
            sampler = SubsetRandomSampler(indices=dataset_ids)
            shuffle = False
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=Config.BATCH_SIZE,
            collate_fn=lambda batch: 
                            collate_fn(batch, self.image_processor),
            shuffle=shuffle,
            sampler=sampler
        )
        return dataloader
    
    def get_train_dataset(self):
        dataset_dir = self.file_manager.get_train_dir()
        return self._get_dataset(dataset_dir, data_augmentation=True)
    
    def get_valid_dataset(self, train_dataset, valid_ids):
        valid_dataset = copy.deepcopy(train_dataset)
        valid_dataset.coco.imgs = {
            k:v for k,v in valid_dataset.coco.imgs.items() 
            if k in valid_ids
        }
        valid_dataset.data_augmentation = False
        return valid_dataset
    
    def get_test_dataset(self):
        dataset_dir = self.file_manager.get_test_dir()
        return self._get_dataset(dataset_dir, data_augmentation=False)
