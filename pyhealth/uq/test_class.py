import pandas as pd
import persist_to_disk as ptd
import torch

from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
from pyhealth.models import ContraWR, SparcNet
from pyhealth.tasks import sleep_staging_isruc_fn
from pyhealth.trainer import Trainer


@ptd.persistf()
def get_get_trained_model(dev=True, epochs=5):
    
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
        sleep_staging_ds, [0.34, 0.33, 0.33]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
    print(
        "loader size: train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

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


@ptd.persistf()
def get_dataset(split='val', dev=True):
    
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
        sleep_staging_ds, [0.34, 0.33, 0.33]
    )
    return {'val': val_dataset, 'test': test_dataset, 'train': train_dataset}[split]



if __name__ == '__main__':
    from importlib import reload

    import pyhealth.uq.kcal as kcal
    reload(kcal)
    model = get_get_trained_model(dev=True)

    o = kcal.KCal(model, debug=True, d=32)
    o.fit(train_dataset=get_dataset('train'), bs_pred=5, bs_supp=5)
    o.calibrate(cal_dataset=get_dataset('val'))
    o.eval()
    test_loader = get_dataloader(get_dataset('test'), batch_size=3, shuffle=False)
    with torch.no_grad():
        for test_batch in test_loader:
            res = o.forward(**test_batch)
            break
    
    