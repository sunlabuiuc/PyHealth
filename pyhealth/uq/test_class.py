import os

import pandas as pd
import persist_to_disk as ptd
import torch

from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
from pyhealth.models import ContraWR, SparcNet
from pyhealth.tasks import sleep_staging_isruc_fn
from pyhealth.trainer import Trainer


@ptd.persistf()
def get_dataset(dev=True, train_split=0.6, val_split=0.2):
    
    # step 1: load signal data
    dataset = ISRUCDataset(
        root="/srv/scratch1/data/ISRUC-I",
        dev=dev,
        refresh_cache=False,
        download=True,
    )

    print(dataset.stat())

    # step 2: set task
    sleep_staging_ds = dataset.set_task(sleep_staging_isruc_fn)
    sleep_staging_ds.stat()
    print(sleep_staging_ds.samples[0])

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sleep_staging_ds, [train_split, val_split, 1-train_split -val_split]
    )
    return sleep_staging_ds, {'val': val_dataset, 'test': test_dataset, 'train': train_dataset}


@ptd.persistf()
def get_get_trained_model(dev=True, epochs=5, train_split=0.6, val_split=0.2):
    sleep_staging_ds, datasets = get_dataset(dev=dev, train_split=train_split, val_split=val_split)

    train_dataloader = get_dataloader(datasets['train'], batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(datasets['val'], batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(f"""loader_size: {','.join([_+'|'+str(len(datasets[_])) for _ in ['train', 'val', 'test']])}""")

    # STEP 3: define model
    model = SparcNet(
        dataset=sleep_staging_ds,
        feature_keys=["signal"],
        label_key="label",
        mode="multiclass",
    )

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        monitor="accuracy",
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
    
    return model

def test_KCal(model, datasets, dev=False, split_by_patient=False, load_best_model_at_last=False):
    cal_model = uq.KCal(model, debug=dev, dim=32)
    if dev:
        cal_model.calibrate(cal_dataset=datasets['train'], train_dataset=datasets['train'], epochs=2, train_split_by_patient=split_by_patient)
        #cal_model.calibrate(cal_dataset=datasets['train'])
    else:
        #NOTE: The option to directly apply is commented out. Uncomment (and comment out the line below it) to test "Untrained"
        #cal_model.calibrate(cal_dataset=datasets['val']) 
        cal_model.calibrate(cal_dataset=datasets['val'], train_dataset=datasets['train'], 
                            train_split_by_patient=split_by_patient, load_best_model_at_last=load_best_model_at_last)
    print(Trainer(model=cal_model).evaluate(test_dataloader))
    print(split_by_patient, load_best_model_at_last)
    # iid training:
        # last: {'accuracy': 0.7303689172252193, 'f1_macro': 0.6723729303147306, 'f1_micro': 0.7303689172252193, 'brier_top1': 0.1624394356864779, 'ECE': 0.01857905730632738, 'ECE_adapt': 0.018753550572241425, 'cwECEt': 0.03351102548966962, 'cwECEt_adapt': 0.03324275630220515, 'loss': 0.7355402175110874}
        # best: {'accuracy': 0.7299511379942965, 'f1_macro': 0.674266314382974, 'f1_micro': 0.7299511379942965, 'brier_top1': 0.1640010064908008, 'ECE': 0.016736564971431108, 'ECE_adapt': 0.016472468704773945, 'cwECEt': 0.03285920705761686, 'cwECEt_adapt': 0.03276531069411252, 'loss': 0.7352885068813636}

    # cross-group training:
        # last: {'accuracy': 0.7234301491290211, 'f1_macro': 0.6704747419823732, 'f1_micro': 0.7234301491290211, 'brier_top1': 0.16744656293673485, 'ECE': 0.014152077469427982, 'ECE_adapt': 0.013806219407609724, 'cwECEt': 0.03320402350861207, 'cwECEt_adapt': 0.033243825050117234, 'loss': 0.750489552379226} 
        # best: {'accuracy': 0.7259913174577226, 'f1_macro': 0.6707387450082415, 'f1_micro': 0.7259913174577226, 'brier_top1': 0.16649657759376135, 'ECE': 0.013335821308244854, 'ECE_adapt': 0.013295454057411078, 'cwECEt': 0.03407399784974526, 'cwECEt_adapt': 0.03398056066755895, 'loss': 0.7413015165065893}

    # Untrained: {'accuracy': 0.7228125624398307, 'f1_macro': 0.6620750673184927, 'f1_micro': 0.7228125624398308, 'brier_top1': 0.16823867486854877, 'ECE': 0.0070479668872682425, 'ECE_adapt': 0.008330089265145252, 'cwECEt': 0.03842654176149014, 'cwECEt_adapt': 0.03817514330317887, 'loss': 0.7374457463807167}

def test_TemperatureScaling(model, datasets, dev=False):
    cal_model = uq.TemperatureScaling(model, debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'])
    print(Trainer(model=cal_model).evaluate(test_dataloader))
    # After: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.1690855287831884, 'ECE': 0.0133140537816558, 'ECE_adapt': 0.012904327771012886, 'cwECEt': 0.05194820100811948, 'cwECEt_adapt': 0.051673596521491505, 'loss': 0.747624546357088}

def test_SCRIB(model, datasets, dev=False):
    cal_model = uq.SCRIB(model, 0.1, 'overall', debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'])
    print(Trainer(model=cal_model).evaluate(test_dataloader))
    
def test_LABEL(model, datasets, dev=False):
    cal_model = uq.LABEL(model, 0.1, debug=dev)
    #cal_model = uq.LABEL(model, [0.1, 0.1, 0.1, 0.1, 0.1], debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'])
    print(Trainer(model=cal_model).evaluate(test_dataloader))
    


if __name__ == '__main__':
    from importlib import reload

    import pyhealth.uq as uq
    dev = True

    model = get_get_trained_model(epochs=10, dev=dev)
    _, datasets = get_dataset(dev)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=model).evaluate(test_dataloader))
    # Pre-calibrate: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.17428343458993806, 'ECE': 0.06710521236002231, 'ECE_adapt': 0.06692437927112259, 'cwECEt': 0.07640062884173958, 'cwECEt_adapt': 0.07623978359739776, 'loss': 0.7824779271569161}
    
    test_LABEL(model, datasets, dev)
