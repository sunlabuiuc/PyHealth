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

if __name__ == '__main__':
    from importlib import reload

    import pyhealth.uq as uq
    dev = False

    model = get_get_trained_model(epochs=10, dev=dev)
    _, datasets = get_dataset(dev)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=model).evaluate(test_dataloader))

    cal_model = uq.KCal(model, debug=dev, d=32)
    if dev:
        cal_model.fit(train_dataset=datasets['train'], val_dataset=datasets['train'], epochs=2)
        cal_model.calibrate(cal_dataset=datasets['train'])
    else:
        cal_model.fit(train_dataset=datasets['train'], val_dataset=datasets['val'])
        cal_model.calibrate(cal_dataset=datasets['val'])
    print(Trainer(model=cal_model).evaluate(test_dataloader))


# Pre-calibrate: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.17428343458993806, 'ECE': 0.06710521236002231, 'ECE_adapt': 0.06692437927112259, 'cwECEt': 0.07640062884173958, 'cwECEt_adapt': 0.07623978359739776, 'loss': 0.7824779271569161}
# Post-calibrate (group): {'accuracy': 0.7217408678909415, 'f1_macro': 0.664123893816841, 'f1_micro': 0.7217408678909415, 'brier_top1': 0.1692223981181552, 'ECE': 0.015064848566449641, 'ECE_adapt': 0.015708275340033162, 'cwECEt': 0.03792842117009698, 'cwECEt_adapt': 0.03788084028878816, 'loss': 0.7530207563952512}
# Post-calibrate (iid): {'accuracy': 0.7293880442482699, 'f1_macro': 0.67423321029707, 'f1_micro': 0.72938804424827, 'brier_top1': 0.16321523621889844, 'ECE': 0.015654685328345127, 'ECE_adapt': 0.016525406151089137, 'cwECEt': 0.032718917617014155, 'cwECEt_adapt': 0.03251704491053249, 'loss': 0.7395213880483656}