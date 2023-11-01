
import pytorch_lightning as pl
from faster_config import Config
from faster_file_manager import FileManager
from pytorch_lightning import Trainer


class ModelTrainer():
    
    def __init__(self, file_manager:FileManager):
        self.file_manager = file_manager

    def fit(self, model, train_loader, valid_loader):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k = 1,
            save_last = True,
            monitor = "valid_loss",
            mode = "min"
        )

        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor = 'valid_loss',
            patience = Config.PATIENCE
        )

        version = self.file_manager.get_version()

        logger = pl.loggers.TensorBoardLogger(
            save_dir = './',
            version = version
        )

        if Config.LOCAL_ENV:
            trainer = Trainer(
                max_steps = Config.STEPS,
                logger = logger
            )
        else:
            trainer = Trainer(
                max_epochs = Config.EPOCHS,
                callbacks = [
                    checkpoint_callback, 
                    early_stopping_callback
                ],
                accelerator = Config.ACCELERATOR,
                logger = logger,
                log_every_n_steps = 5
            )
            
        trainer.fit(
            model, 
            train_dataloaders = train_loader, 
            val_dataloaders = valid_loader
        )
        