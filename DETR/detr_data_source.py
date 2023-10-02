

from detr_config import Config
from detr_dataset import InBreastDataset, collate_fn
from detr_file_manager import FileManager
from torch.utils.data import DataLoader
from torchvision import datasets


from detr_factory import DETRFactory

class DataSource():
    
    def __init__(self, detr_factory, file_manager:FileManager):
        self.image_processor = detr_factory.new_image_processor()
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
        dataset_dir = self.file_manager.get_train_dir()
        return self._get_dataset_dataloader(dataset_dir)
    
    def get_valid_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_valid_dir()
        return self._get_dataset_dataloader(dataset_dir)
    
    def get_train_valid_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_train_valid_dir()
        return self._get_dataset_dataloader(dataset_dir)
    
    def get_test_dataset_dataloader(self):
        dataset_dir = self.file_manager.get_test_dir()
        return self._get_dataset_dataloader(dataset_dir)