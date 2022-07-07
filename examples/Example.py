import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append("./")
import pyhealth.datasets.datasets as datasets
import pyhealth.models.models as models
import pyhealth.utils as utils
import torch
print(torch.cuda.is_available())


print ('---load dataset and initialize model ---')
# load MIMIC-III
dataset = datasets.MIMIC_III(
    table_names=['med', 'prod', 'diag']
)

# initialize the model and build the dataloaders
model = models.RETAIN(
    voc_size=dataset.voc_size,
    ddi_adj=dataset.ddi_adj,
    emb_dim=64,
)

dataset.get_dataloader("RETAIN")



print ('--- train and test ---')
# set trainer with checkpoint
checkpoint_callback = ModelCheckpoint(dirpath='./model_cpt')
trainer = Trainer(
    gpus=1,
    max_epochs=5,
    progress_bar_refresh_rate=5,
    callbacks=[checkpoint_callback],
)

# train model
trainer.fit(
    model=model,
    train_dataloaders=dataset.train_loader,
    val_dataloaders=dataset.val_loader,
)

# test the best model
model.summary(
    test_dataloaders=dataset.test_loader,
    ckpt_path=checkpoint_callback.best_model_path,
)