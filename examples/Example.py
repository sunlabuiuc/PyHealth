# %% md

# 1. load/reload all dependency

from importlib import reload

from MedCode import CodeMapping
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import pyhealth.datasets.datasets as datasets
import pyhealth.models.models as models
import pyhealth.utils as utils

reload(datasets)
reload(models)
reload(utils)

# 2. get MIMIC-III dataset object

# choose a target code: ATC4, NDC or RXCUI
target_code = 'ATC4'
tool = CodeMapping('RxNorm', target_code)
tool.load_mapping()

# load dataset by passing arguments
dataset = datasets.MIMIC_III(
    table_names=['med', 'prod', 'diag'],
    code_map=tool.RxNorm_to_ATC4,
)

# 3. model train and test

# initialize RETAIN model and build the dataloaders
model = models.RETAIN(
    voc_size=dataset.voc_size,
    emb_size=64,
)
dataset.get_dataloader("RETAIN")

# set checkpoint and trainer
checkpoint_callback = ModelCheckpoint(dirpath='./model_cpt')
trainer = Trainer(
    gpus=1,
    max_epochs=20,
    progress_bar_refresh_rate=5,
    callbacks=[checkpoint_callback],
)

# train model
trainer.fit(
    model=model,
    train_dataloaders=dataset.train_loader,
    val_dataloaders=dataset.val_loader,
)

# load the best model for test
checkpoint_callback.best_model_path
model.summary(dataset.test_loader)

# %%
