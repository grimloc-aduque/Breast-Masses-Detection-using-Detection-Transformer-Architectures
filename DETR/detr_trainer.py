
import pytorch_lightning as pl
from detr_config import Config
from pytorch_lightning import Trainer


def get_trainer(version):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k = 1,
        save_last = True,
        monitor = "valid_loss",
        mode = "min"
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor = 'valid_loss',
        patience = 50
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir = './',
        version = version
    )

    if Config.LOCAL_ENV:
        trainer = Trainer(
            max_steps = 1,
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
            logger = logger
        )
    
    return trainer