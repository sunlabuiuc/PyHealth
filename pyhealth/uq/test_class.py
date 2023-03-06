import os

import pandas as pd
import persist_to_disk as ptd
import torch

from pyhealth.datasets import (ISRUCDataset, MIMIC3Dataset, get_dataloader,
                               split_by_patient)
from pyhealth.models import RNN, ContraWR, SparcNet, Transformer
from pyhealth.tasks import (drug_recommendation_mimic3_fn,
                            length_of_stay_prediction_mimic3_fn,
                            readmission_prediction_mimic3_fn,
                            sleep_staging_isruc_fn)
from pyhealth.trainer import Trainer

metrics = {
    'multiclass': ["accuracy", "f1_macro", "f1_micro"] + ['brier_top1', 'ECE', 'ECE_adapt', 'cwECEt', 'cwECEt_adapt'],
    'multilabel': ['pr_auc_samples'] + ['cwECE', 'cwECE_adapt'],
    'binary': ["pr_auc", "roc_auc", "f1"] + ['ECE', 'ECE_adapt']
}

@ptd.persistf()
def get_dataset_mimic_drug(dev=True, train_split=0.6, val_split=0.2):
    base_dataset = MIMIC3Dataset(
        root="/srv/scratch1/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=dev,
        refresh_cache=False,
    )

    print(base_dataset.stat())

    # step 2: set task
    sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
    sample_dataset.stat()
    print(sample_dataset.samples[0])

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [train_split, val_split, 1-train_split -val_split]
    )
    return sample_dataset, {'val': val_dataset, 'test': test_dataset, 'train': train_dataset}

@ptd.persistf()
def get_dataset_mimic_binary(dev=True, train_split=0.6, val_split=0.2):
    base_dataset = MIMIC3Dataset(
        root="/srv/scratch1/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
        dev=dev,
        refresh_cache=False,
    )

    print(base_dataset.stat())

    # step 2: set task
    sample_dataset = base_dataset.set_task(readmission_prediction_mimic3_fn)
    sample_dataset.stat()
    print(sample_dataset.samples[0])

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [train_split, val_split, 1-train_split -val_split]
    )
    return sample_dataset, {'val': val_dataset, 'test': test_dataset, 'train': train_dataset}


@ptd.persistf()
def get_get_trained_model_drug(dev=True, epochs=50, train_split=0.6, val_split=0.2):
    sample_dataset, datasets = get_dataset_mimic_drug(dev=dev, train_split=train_split, val_split=val_split)

    train_dataloader = get_dataloader(datasets['train'], batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(datasets['val'], batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)

    # STEP 3: define model
    model = Transformer(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures"],
        label_key="drugs",
        mode="multilabel",
    )


    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        monitor="pr_auc_samples",
    )


    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))

    return model


@ptd.persistf()
def get_trained_model_binary(dev=True, epochs=50, train_split=0.6, val_split=0.2):
    sample_dataset, datasets = get_dataset_mimic_binary(dev=dev, train_split=train_split, val_split=val_split)

    train_dataloader = get_dataloader(datasets['train'], batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(datasets['val'], batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)

    # STEP 3: define model
    model = RNN(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"],
        label_key="label",
        mode="binary",
    )


    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        monitor="roc_auc",
    )

    return model


@ptd.persistf()
def get_dataset_multiclass(dev=True, train_split=0.6, val_split=0.2):
    base_dataset = MIMIC3Dataset(
        root="/srv/scratch1/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
        dev=dev,
        refresh_cache=False,
    )

    # step 2: set task
    sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic3_fn)
    sample_dataset.stat()

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [train_split, val_split, 1-train_split -val_split]
    )
    return sample_dataset, {'val': val_dataset, 'test': test_dataset, 'train': train_dataset}


@ptd.persistf()
def get_trained_model_multiclass(dev=True, epochs=50, train_split=0.6, val_split=0.2):
    sample_dataset, datasets = get_dataset_multiclass(dev=dev, train_split=train_split, val_split=val_split)

    train_dataloader = get_dataloader(datasets['train'], batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(datasets['val'], batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)

    # STEP 3: define model
    model = Transformer(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"],
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

    return model



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
def get_get_trained_model(dev=True, epochs=10, train_split=0.6, val_split=0.2):
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

def test_KCal(model, datasets, dev=False, split_by_patient=False, load_best_model_at_last=False, **kwargs):
    cal_model = uq.KCal(model, debug=dev, dim=32)
    if dev:
        #cal_model.calibrate(cal_dataset=datasets['train'], train_dataset=datasets['train'], epochs=2, train_split_by_patient=split_by_patient)
        cal_model.calibrate(cal_dataset=datasets['train'])
    else:
        #NOTE: The option to directly apply is commented out. Uncomment (and comment out the line below it) to test "Untrained"
        #cal_model.calibrate(cal_dataset=datasets['val'])
        cal_model.calibrate(cal_dataset=datasets['val'], train_dataset=datasets['train'],
                            train_split_by_patient=split_by_patient, load_best_model_at_last=load_best_model_at_last, **kwargs)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=cal_model, metrics=metrics[model.mode]).evaluate(test_dataloader))
    print(split_by_patient, load_best_model_at_last)
    # iid training:
        # last: {'accuracy': 0.7303689172252193, 'f1_macro': 0.6723729303147306, 'f1_micro': 0.7303689172252193, 'brier_top1': 0.1624394356864779, 'ECE': 0.01857905730632738, 'ECE_adapt': 0.018753550572241425, 'cwECEt': 0.03351102548966962, 'cwECEt_adapt': 0.03324275630220515, 'loss': 0.7355402175110874}
        # best: {'accuracy': 0.7299511379942965, 'f1_macro': 0.674266314382974, 'f1_micro': 0.7299511379942965, 'brier_top1': 0.1640010064908008, 'ECE': 0.016736564971431108, 'ECE_adapt': 0.016472468704773945, 'cwECEt': 0.03285920705761686, 'cwECEt_adapt': 0.03276531069411252, 'loss': 0.7352885068813636}

    # cross-group training:
        # last: {'accuracy': 0.7234301491290211, 'f1_macro': 0.6704747419823732, 'f1_micro': 0.7234301491290211, 'brier_top1': 0.16744656293673485, 'ECE': 0.014152077469427982, 'ECE_adapt': 0.013806219407609724, 'cwECEt': 0.03320402350861207, 'cwECEt_adapt': 0.033243825050117234, 'loss': 0.750489552379226}
        # best: {'accuracy': 0.7259913174577226, 'f1_macro': 0.6707387450082415, 'f1_micro': 0.7259913174577226, 'brier_top1': 0.16649657759376135, 'ECE': 0.013335821308244854, 'ECE_adapt': 0.013295454057411078, 'cwECEt': 0.03407399784974526, 'cwECEt_adapt': 0.03398056066755895, 'loss': 0.7413015165065893}

    # Untrained: {'accuracy': 0.7228125624398307, 'f1_macro': 0.6620750673184927, 'f1_micro': 0.7228125624398308, 'brier_top1': 0.16823867486854877, 'ECE': 0.0070479668872682425, 'ECE_adapt': 0.008330089265145252, 'cwECEt': 0.03842654176149014, 'cwECEt_adapt': 0.03817514330317887, 'loss': 0.7374457463807167}

def test_TemperatureScaling(model, datasets, dev=False, **kwargs):
    cal_model = uq.TemperatureScaling(model, debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'], **kwargs)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=cal_model, metrics=metrics[model.mode]).evaluate(test_dataloader))
    # After: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.1690855287831884, 'ECE': 0.0133140537816558, 'ECE_adapt': 0.012904327771012886, 'cwECEt': 0.05194820100811948, 'cwECEt_adapt': 0.051673596521491505, 'loss': 0.747624546357088}

def test_SCRIB(model, datasets, dev=False):
    cal_model = uq.SCRIB(model, 0.1, debug=dev)
    #cal_model = uq.SCRIB(model, [0.1] * 5, debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'])
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=cal_model, metrics=metrics[model.mode]).evaluate(test_dataloader))

def test_LABEL(model, datasets, dev=False):
    cal_model = uq.LABEL(model, 0.1, debug=dev)
    #cal_model = uq.LABEL(model, [0.1] * 5, debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'])
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=cal_model, metrics=metrics[model.mode]).evaluate(test_dataloader))


def test_HB(model, datasets, dev=False, **kwargs):
    cal_model = uq.HistogramBinning(model, debug=dev)
    cal_model.calibrate(cal_dataset=datasets['val'], **kwargs)
    test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
    print(Trainer(model=cal_model, metrics=metrics[model.mode]).evaluate(test_dataloader))
    # After: {'accuracy': 0.7189072348464207, 'f1_macro': 0.6576898642873795, 'f1_micro': 0.7189072348464207, 'brier_top1': 0.16840067132548106, 'ECE': 0.04350910698408489, 'ECE_adapt': 0.044153410776727735, 'cwECEt': 0.04391579116036477, 'cwECEt_adapt': 0.04455814993598299, 'loss': 0.7335764775031021}

if __name__ == '__main__':
    from importlib import reload

    import pyhealth.uq as uq
    dev = False
    if False: # multiclass
        _, datasets = get_dataset_multiclass(dev)
        model = get_trained_model_multiclass(dev=dev)
        test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
        print(Trainer(model=model, metrics=metrics[model.mode]).evaluate(test_dataloader))

        #test_KCal(model, datasets, dev, load_best_model_at_last=True, optimizer_params={'lr': 1e-4})
        # {'accuracy': 0.4401496259351621, 'f1_macro': 0.2877082476835488, 'f1_micro': 0.4401496259351621, 'brier_top1': 0.19963319080703884, 'ECE': 0.02726052619045594, 'ECE_adapt': 0.026895376073073643, 'cwECEt': 0.01591851540294672, 'cwECEt_adapt': 0.017954394785340017, 'loss': 1.4853952246299689}
        #test_TemperatureScaling(model, datasets, dev)
        # {'accuracy': 0.44321015642711403, 'f1_macro': 0.30895310531080705, 'f1_micro': 0.44321015642711403, 'brier_top1': 0.20031940694168032, 'ECE': 0.01941438210818589, 'ECE_adapt': 0.02330048461162809, 'cwECEt': 0.022519317838478676, 'cwECEt_adapt': 0.02455287158864352, 'loss': 1.4698933535727903}
        #test_SCRIB(model, datasets, dev)

        #test_HB(model, datasets, dev)
        # sum
        # {'accuracy': 0.4407163908410791, 'f1_macro': 0.2640266353953583, 'f1_micro': 0.4407163908410791, 'brier_top1': 0.19667096873346457, 'ECE': 0.032893533392561666, 'ECE_adapt': 0.033776329521607494, 'cwECEt': 0.01616424215561377, 'cwECEt_adapt': 0.02126586349394122, 'loss': 1.4877435789592024}
        # None
        # {'accuracy': 0.4407163908410791, 'f1_macro': 0.2640266353953583, 'f1_micro': 0.4407163908410791, 'brier_top1': 0.20056810416311954, 'ECE': 0.027721811747218536, 'ECE_adapt': 0.022718262078539896, 'cwECEt': 0.009997820777836595, 'cwECEt_adapt': 0.01568430235516669, 'loss': 1.49660099848457}

    if False: # multiclass (ISRUC)
        _, datasets = get_dataset(dev)
        model = get_get_trained_model(epochs=10, dev=dev)
        test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
        print(Trainer(model=model, metrics=metrics[model.mode]).evaluate(test_dataloader))
        # Pre-calibrate: {'accuracy': 0.709843241966832, 'f1_macro': 0.6511024300262231, 'f1_micro': 0.709843241966832, 'brier_top1': 0.17428343458993806, 'ECE': 0.06710521236002231, 'ECE_adapt': 0.06692437927112259, 'cwECEt': 0.07640062884173958, 'cwECEt_adapt': 0.07623978359739776, 'loss': 0.7824779271569161}

        #test_KCal(model, datasets, dev)
        #test_TemperatureScaling(model, datasets, dev)
        test_HB(model, datasets, dev)
    if False: # multilabel
        _, datasets = get_dataset_mimic_drug(dev)
        model = get_get_trained_model_drug(dev=dev)
        test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
        print(Trainer(model=model, metrics=metrics[model.mode]).evaluate(test_dataloader))
        test_TemperatureScaling(model, datasets, dev)
        # {'pr_auc_samples': 0.761561145402367, 'cwECE': 0.028285707645770656, 'cwECE_adapt': 0.028208456076028652, 'loss': 0.2162835333454475}
        test_HB(model, datasets, dev)
        # {'pr_auc_samples': 0.7571941712990254, 'cwECE': 0.011628713390838414, 'cwECE_adapt': 0.01729795687946442, 'loss': 0.223203511720293}

    if False: # binary
        _, datasets = get_dataset_mimic_binary(dev)
        model = get_trained_model_binary(dev=dev)
        test_dataloader = get_dataloader(datasets['test'], batch_size=32, shuffle=False)
        print(Trainer(model=model, metrics=metrics[model.mode]).evaluate(test_dataloader))
        test_TemperatureScaling(model, datasets, dev)
        # {'pr_auc': 0.6582015807143802, 'roc_auc': 0.6330558925968278, 'f1': 0.67216673903604, 'ECE': 0.026914832605028666, 'ECE_adapt': 0.03994572116134691, 'loss': 0.6561976915500203}
        #test_HB(model, datasets, dev)
        # {'pr_auc': 0.6373137820731617, 'roc_auc': 0.6228630007245175, 'f1': 0.6524886877828054, 'ECE': 0.05897163179350966, 'ECE_adapt': 0.06523191008292568, 'loss': 0.667492433161032}