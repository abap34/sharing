import os

import pytorch_lightning as pl
import skimage.transform as transform
import timm
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Subset, TensorDataset

import pl_module
from dataset import *
from utils import info


def train(params):
    info('start training...')
    pl.seed_everything(0)
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    train_wave, train_meta, train_y = load_train(params["data"])
    if params["transform"]:
        transformer = lambda image : transform.resize(image, (params["size"], params["size"])) 
        dataset = TrainDataset(train_wave, train_meta, train_y, transformer=transformer)
    else:
        dataset = TrainDataset(train_wave, train_meta, train_y)

    info('fin setup.')
    for fold, (train_index, valid_index) in enumerate(kf.split(range(2000), train_y)):
        info('run fold', fold + 1)

        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, valid_index)

        datamodule = PlDataModule(train_dataset, val_dataset)
        info('load model.')
        core_model = timm.create_model(params["model_name"], pretrained=True, num_classes=0, in_chans=12)
        # core_model, sample_input, optimizer,  freeze
        model = pl_module.Model(core_model, train_dataset[1].unsqueeze(0), params["optimizer"], params["freeze"])
        
        early_stopping = EarlyStopping("val_auc", verbose=True, patience=5,  mode='max')
        info('log:', params["log"])
        if params["log"]:
            run_name = "fold-" + str(fold + 1)
            wandb.init(
                project="MedCon2021",
                reinit=True,
                name=run_name,
                tags=params["tags"],
                group=params["run-id"],
                save_code=True,
            )
            wandb_logger = WandbLogger()

            trainer = pl.Trainer(
                max_epochs=64, logger=wandb_logger, callbacks=[early_stopping]
            )
        else:
            trainer = pl.Trainer(
                max_epochs=64, callbacks=[early_stopping]
            )

        info('start fit.')
        trainer.fit(model=model, datamodule=datamodule)
        info('done.')
        path = './data/output/models/' + params["model_name"]
        if not os.path.exists(path):
            info('make dir:', path)
            os.makedirs(path)
        model_path = path + "/fold-" + str(fold + 1) + ".pth"
        torch.save(model.model.state_dict(), model_path)
        info('save model in ', model_path)




