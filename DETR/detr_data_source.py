

from detr_config import Config
from detr_dataset import InBreastDataset, collate_fn
from detr_file_manager import FileManager
from torch.utils.data import DataLoader


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
            
    def _get_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = Config.BATCH_SIZE,
            collate_fn = lambda batch: 
                            collate_fn(batch, self.image_processor),
            shuffle = True
        )
        return dataloader
    
    def _get_dataset_dataloader(self, dataset_dir, data_augmentation):
        dataset = self._get_dataset(dataset_dir, data_augmentation)
        dataloader = self._get_dataloader(dataset)
        return dataset, dataloader
    
    
    def get_train_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_train_dir()
        return self._get_dataset_dataloader(dataset_dir, data_augmentation=True)
    
    def get_valid_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_valid_dir()
        return self._get_dataset_dataloader(dataset_dir, data_augmentation=False)
    
    def get_train_valid_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_train_valid_dir()
        return self._get_dataset_dataloader(dataset_dir, data_augmentation=True)
    
    def get_test_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_test_dir()
        return self._get_dataset_dataloader(dataset_dir, data_augmentation=False)