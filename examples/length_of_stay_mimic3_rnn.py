from pyhealth.calib import calibration, predictionset
from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import Transformer
from pyhealth.tasks import length_of_stay_prediction_mimic3_fn
from pyhealth.trainer import Trainer, get_metrics_fn

# STEP 1: load data
base_dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
    dev=False,
    refresh_cache=True,
)
base_dataset.stat()

# STEP 2: set task
sample_dataset = base_dataset.set_task(length_of_stay_prediction_mimic3_fn)
sample_dataset.stat()

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)
train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

# STEP 3: define modedl
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
    epochs=50,
    monitor="accuracy",
)

# STEP 5: evaluate
metrics = ['accuracy', 'f1_macro', 'f1_micro'] + ['ECE_adapt', 'cwECEt_adapt']
y_true_all, y_prob_all = trainer.inference(test_dataloader)[:2]
print(get_metrics_fn(model.mode)(y_true_all, y_prob_all, metrics=metrics))

# STEP 6: calibrate the model
cal_model = calibration.HistogramBinning(model, debug=True)
cal_model.calibrate(cal_dataset=val_dataset)
y_true_all, y_prob_all = Trainer(model=cal_model).inference(test_dataloader)[:2]
print(get_metrics_fn(cal_model.mode)(y_true_all, y_prob_all, metrics=metrics))


# STEP 7: Construct prediction set, controlling overall miscoverage rate (<0.1)
# Note that if you use calibrated model the coverate rate cannot be controlled, because
# with repect to the calibrated model (which was trained on the calibration set), the
# test set and calibration set is not i.i.d
ps_model = predictionset.LABEL(model, 0.1, debug=True)
ps_model.calibrate(cal_dataset=val_dataset)
y_true_all, y_prob_all, _, extra_output = Trainer(model=ps_model).inference(test_dataloader, additional_outputs=['y_predset'])
print(get_metrics_fn(ps_model.mode)(y_true_all, y_prob_all,
                                     metrics=metrics + ['miscoverage_overall_ps', 'rejection_rate'],
                                     y_predset=extra_output['y_predset']))
