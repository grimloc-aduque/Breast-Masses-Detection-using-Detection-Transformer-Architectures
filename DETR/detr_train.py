
import itertools
import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import DetrConfig, DetrForObjectDetection

from detr_config import Config
from detr_dataset import collate_fn, get_train_dataset, get_test_dataset
from detr_model import DETRModel



if __name__ == '__main__':
    # Dataset
    train_dataset = get_train_dataset()
    test_dataset = get_test_dataset()

    
    # HyperParameters

    hyperparameters = itertools.product(*[
        Config.NUM_QUERIES,
        Config.D_MODEL,
        Config.ENCODER_DECODER_LAYERS
    ])


    # Hyperparameter Search

    for num_queries, d_model, encoder_decoder_layers in hyperparameters:
        print('(Num Queries, Dim model, Enc-Dec Layers): ', 
                f'({num_queries}, {d_model}, {encoder_decoder_layers})' )

        # Model Construction

        config = DetrConfig.from_pretrained(
            Config.CHECKPOINT,
            num_labels=1,
            id2label = {0:'Mass'}, 
            label2id={'Mass': 0},
            num_queries = num_queries,
            d_model = d_model,
            num_head = 8,
            encoder_layers = encoder_decoder_layers,
            decoder_layers = encoder_decoder_layers,
            position_embedding_type  = 'sine',
            decoder_ffn_dim = 2048,
            encoder_ffn_dim = 2048,
        )

        detr_model = DetrForObjectDetection.from_pretrained(
            Config.CHECKPOINT,
            config = config,
            ignore_mismatched_sizes=True
        )

        model = DETRModel(detr_model=detr_model)


        # Training with K-fold Cross Validation 

        k_fold = KFold(n_splits=10, shuffle=True, random_state=123456)

        for fold, (train_idx, valid_idx) in enumerate(k_fold.split(train_dataset)):
            print(f"Fold {fold + 1}")

            train_loader = DataLoader(
                dataset = train_dataset,
                batch_size = Config.BATCH_SIZE,
                collate_fn=collate_fn,
                sampler = SubsetRandomSampler(train_idx),
            )

            valid_loader = DataLoader(
                dataset = train_dataset,
                batch_size = Config.BATCH_SIZE,
                collate_fn=collate_fn,
                sampler = SubsetRandomSampler(valid_idx),
            )

            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                save_top_k = 1,
                save_last = True,
                monitor = "valid_loss",
                mode = "min"
            )

            early_stopping_callback = pl.callbacks.EarlyStopping(
                monitor = 'valid_loss',
                patience = 15
            )

            version = os.path.join(
                f'queries={num_queries}_dmodel={d_model}_layers={encoder_decoder_layers}',
                f'fold_{fold+1}'
            )

            logger = pl.loggers.TensorBoardLogger(
                save_dir = './',
                version = version
            )

            trainer = Trainer(
                max_epochs = Config.EPOCHS, 
                log_every_n_steps = 5, 
                callbacks = [
                    checkpoint_callback, 
                    early_stopping_callback
                ],
                accelerator = Config.ACCELERATOR,
                logger = logger
            )
            
            trainer.fit(
                model, 
                train_dataloaders = train_loader, 
                val_dataloaders = valid_loader
            )

            break # Fold
        break # Hyperparameter
