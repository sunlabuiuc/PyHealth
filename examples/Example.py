import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys

sys.path.append("./home/chaoqiy2/github/PyHealth-OMOP")
import pyhealth.datasets.datasets as datasets
import pyhealth.models.models as models
import pyhealth.utils as utils
import torch
import os

root = "/home/chaoqiy2/github/PyHealth-OMOP/pyhealth-web/downloads"


def default_return(output_file):
    with open(output_file, "w") as outfile:
        print({"result": "None"}, file=outfile)
        return output_file


def run_healthcare_ml_job(run_id, trigger_time, config):

    # config load
    dataset = config["dataset"]
    task = config["task"]
    model = config["model"]

    output_file = os.path.join(
        root,
        "{}_{}_{}_{}_{}.json".format(
            run_id, dataset, task, model, trigger_time.timestamp()
        ),
    )

    # load MIMIC-III
    if dataset == "mimic_iii":
        dataset = datasets.MIMIC_III()
    else:
        default_return(output_file)

    # initialize the model and build the dataloaders
    if model == "safedrug":
        dataset.get_dataloader("SafeDrug")
        model = models.SafeDrug(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == "gamenet":
        dataset.get_dataloader("GAMENet")
        model = models.GAMENet(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == "micron":
        dataset.get_dataloader("MICRON")
        model = models.MICRON(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == "retain":
        dataset.get_dataloader("RETAIN")
        model = models.RETAIN(
            dataset=dataset,
            emb_dim=64,
        )
    else:
        default_return(output_file)

    if task == "drug_rec":
        # set trainer with checkpoint
        checkpoint_callback = ModelCheckpoint(dirpath="./model_cpt")
        trainer = Trainer(
            gpus=1,
            max_epochs=3,
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
            output_file=output_file,
            test_dataloaders=dataset.test_loader,
            ckpt_path=checkpoint_callback.best_model_path,
        )
    else:
        default_return(output_file)

    return output_file
