

import copy

from detr_config import Config
from detr_dataset import DETRDataset, collate_fn
from detr_file_manager import FileManager
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DataSource():
    
    def __init__(self, detr_factory, file_manager:FileManager):
        self.image_processor = detr_factory.new_image_processor()
        self.file_manager = file_manager
        self.train_dataset = self._get_train_dataset()

    # Dataset and Dataloader

    def _get_dataset(self, dataset_dir, data_augmentation):
        dataset = DETRDataset(
            dataset_dir = dataset_dir,
            processor = self.image_processor,
            data_augmentation = data_augmentation
        )
        return dataset
            
    def _get_dataloader(self, dataset, indices=[], shuffle=False):
        if len(indices)>0:
            sampler = SubsetRandomSampler(indices=indices)
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=Config.BATCH_SIZE,
            collate_fn=lambda batch: 
                            collate_fn(batch, self.image_processor),
            shuffle=shuffle,
            sampler=sampler,
            # num_workers=Config.NUM_WORKERS, # Uncomment to run in server
        )
        return dataloader
    
    # Train and Test Datasets
    
    def _get_train_dataset(self):
        train_dir = self.file_manager.get_train_dir()
        dataset = self._get_dataset(
            dataset_dir=train_dir, 
            data_augmentation=True
        )
        return dataset
    
    def _get_test_dataset(self):
        test_dir = self.file_manager.get_test_dir()
        test_dataset = self._get_dataset(
            dataset_dir=test_dir, 
            data_augmentation=False
        )
        return test_dataset
    
    
    # KFOLD and Testing
    
    def start_kfold(self):
        self.kfold = KFold(
            n_splits=Config.FOLDS, 
            shuffle=True, 
            random_state=123456
        )
        self.kfold_split = self.kfold.split(self.train_dataset.ids)
        self.fold = 0
    
    def next_fold(self):
        self.fold += 1
        self.file_manager.set_validation_setup(fold=self.fold)
        train_ids, valid_ids = self.kfold_split.__next__()
        
        train_dataset = copy.deepcopy(self.train_dataset)
        train_loader = self._get_dataloader(
            dataset=train_dataset,
            indices=train_ids
        )
        
        valid_dataset = copy.deepcopy(self.train_dataset)
        valid_dataset.coco.imgs = {
            k:v for k,v in valid_dataset.coco.imgs.items() 
            if k in valid_ids
        }
        valid_dataset.data_augmentation = False
        
        valid_loader = self._get_dataloader(
            dataset=valid_dataset,
            indices=valid_ids
        )
        
        return (train_dataset, valid_dataset), (train_loader, valid_loader)
    
    
    def testing(self):
        self.file_manager.set_testing_setup()
        train_dataset = copy.deepcopy(self.train_dataset)
        train_loader = self._get_dataloader(
            dataset=train_dataset,
            shuffle=True
        )
        test_dataset = self._get_test_dataset()
        test_loader = self._get_dataloader(
            dataset=test_dataset
        )
        return (train_dataset, test_dataset), (train_loader, test_loader)
    

