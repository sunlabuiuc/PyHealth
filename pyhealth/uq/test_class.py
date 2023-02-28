import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="7"
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

def test_KCal(model, datasets, dev=False, split_by_patient=False, load_best_model_at_last=True):
    cal_model = uq.KCal(model, debug=dev, d=32)
    if dev:
        cal_model.fit(train_dataset=datasets['train'], val_dataset=datasets['train'], epochs=2, split_by_patient=split_by_patient)
        cal_model.calibrate(cal_dataset=datasets['train'])
    else:
        cal_model.fit(train_dataset=datasets['train'], val_dataset=datasets['val'], split_by_patient=split_by_patient, load_best_model_at_last=load_best_model_at_last)
        cal_model.calibrate(cal_dataset=datasets['val'])
    print(Trainer(model=cal_model).evaluate(test_dataloader))
    # Pre-calibrate: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.17428343458993806, 'ECE': 0.06710521236002231, 'ECE_adapt': 0.06692437927112259, 'cwECEt': 0.07640062884173958, 'cwECEt_adapt': 0.07623978359739776, 'loss': 0.7824779271569161}
    # last ckpt:
        # Group: {'accuracy': 0.7234301491290211, 'f1_macro': 0.6704747419823732, 'f1_micro': 0.7234301491290211, 'brier_top1': 0.16744656293673485, 'ECE': 0.014152077469427982, 'ECE_adapt': 0.013806219407609724, 'cwECEt': 0.03320402350861207, 'cwECEt_adapt': 0.033243825050117234, 'loss': 0.750489552379226} 
        # iid:  {'accuracy': 0.7291519081612264, 'f1_macro': 0.671548684223261, 'f1_micro': 0.7291519081612264, 'brier_top1': 0.1639172352517858, 'ECE': 0.019648808434688088, 'ECE_adapt': 0.019401786757314958, 'cwECEt': 0.03410393431255464, 'cwECEt_adapt': 0.033785869332806785, 'loss': 0.7421316428922814}
    # Best ckpt:
        # Group: {'accuracy': 0.7259913174577226, 'f1_macro': 0.6707387450082415, 'f1_micro': 0.7259913174577226, 'brier_top1': 0.16649657759376135, 'ECE': 0.013335821308244854, 'ECE_adapt': 0.013295454057411078, 'cwECEt': 0.03407399784974526, 'cwECEt_adapt': 0.03398056066755895, 'loss': 0.7413015165065893}
        # iid (group best): {'accuracy': 0.7299511379942965, 'f1_macro': 0.674266314382974, 'f1_micro': 0.7299511379942965, 'brier_top1': 0.1640010064908008, 'ECE': 0.016736564971431108, 'ECE_adapt': 0.016472468704773945, 'cwECEt': 0.03285920705761686, 'cwECEt_adapt': 0.03276531069411252, 'loss': 0.7352885068813636}
        # iid (iid best): {'accuracy': 0.729351715619494, 'f1_macro': 0.6745745524669647, 'f1_micro': 0.7293517156194939, 'brier_top1': 0.1643893485320108, 'ECE': 0.01778785605993927, 'ECE_adapt': 0.018174404035031965, 'cwECEt': 0.03434773876859663, 'cwECEt_adapt': 0.034825103930863056, 'loss': 0.7395044076941011}
    # Untrained: {'accuracy': 0.7228125624398307, 'f1_macro': 0.6620750673184927, 'f1_micro': 0.7228125624398308, 'brier_top1': 0.16823867486854877, 'ECE': 0.0070479668872682425, 'ECE_adapt': 0.008330089265145252, 'cwECEt': 0.03842654176149014, 'cwECEt_adapt': 0.03817514330317887, 'loss': 0.7374457463807167}


if __name__ == '__main__':
    from importlib import reload

    import pyhealth.uq as uq
    dev = False

    model = get_get_trained_model(epochs=10, dev=dev)
    _, datasets = get_dataset(dev)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=model).evaluate(test_dataloader))
    
    test_KCal(model, datasets, dev=dev)

    
