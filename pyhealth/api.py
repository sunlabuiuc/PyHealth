import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
import pyhealth.datasets.datasets as datasets
import pyhealth.models.models as models
import pyhealth.utils as utils
import torch
import os
import pickle

output_root = '/home/chaoqiy2/github/PyHealth-OMOP/pyhealth-web/downloads'
download_root = '/download/downloads'

def default_return(path):
    with open(path, 'w') as outfile:
        print ({"result": "None"}, file=outfile)

def run_healthcare_ml_job(run_id, trigger_time, config):

    # config load
    data = config['dataset']
    task = config['task']
    model = config['model']

    output_file = "{}_{}_{}_{}_{}.json".format(
        run_id,
        data,
        task,
        model,
        trigger_time.timestamp()
    )

    output_path = os.path.join(output_root, output_file)
    download_path = os.path.join(download_root, output_file)

    # load MIMIC-III
    if not os.path.exists('./data/{}.pkl'.format(data)):
        if data == 'mimic_iii':
            dataset = datasets.MIMIC_III()
        else:
            default_return(output_path)
            return download_path
        pickle.dump(dataset, open('./data/{}.pkl'.format(data), 'wb'))
    else:
        dataset = pickle.load(open('./data/{}.pkl'.format(data), 'rb'))

    # initialize the model and build the dataloaders
    if model == 'safedrug':
        dataset.get_dataloader("SafeDrug")
        model = models.SafeDrug(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == 'gamenet':
        dataset.get_dataloader("GAMENet")
        model = models.GAMENet(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == 'micron':
        dataset.get_dataloader("MICRON")
        model = models.MICRON(
            dataset=dataset,
            emb_dim=64,
        )
    elif model == 'retain':
        dataset.get_dataloader("RETAIN")
        model = models.RETAIN(
            dataset=dataset,
            emb_dim=64,
        )
    else:
        default_return(output_path)
        return download_path

    if task == 'drug_rec':
        # set trainer with checkpoint
        checkpoint_callback = ModelCheckpoint(dirpath='./model_cpt')
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
            output_path=output_path,
            test_dataloaders=dataset.test_loader,
            ckpt_path=checkpoint_callback.best_model_path,
        )
    else:
        default_return(output_path)
        return download_path

    return download_path