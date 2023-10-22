import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pyhealth.datasets import SleepEDFDataset
from tasks.sleep_staging import multi_epoch_multi_modal_sleep_staging_sleepedf_fn
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader
from models import Seq_Cross_Modal_Transformer_PyHealth
from pyhealth.trainer import Trainer
from pyhealth.metrics.multiclass import multiclass_metrics_fn
from models.cross_modal_transformer import interpret_cmt, interactive_plot_cmt

'''
Example Colab tutorial can be found at:
https://colab.research.google.com/drive/1jwD5NX8cR47MRtnOW_drKIS6lP1tbWFc?usp=sharing
'''


# step 1: load signal data
sleepedf_ds = SleepEDFDataset(
    root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",#'/home/jp65/pyhealth_hackathon/storage.googleapis.com/pyhealth/sleepedf-sample',#"/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
    # dev=True,
    refresh_cache=False,
)


# step 2: set task
modality = ['EEG Fpz-Cz','EOG horizontal']
num_epoch_seq = 15 # Defalut value is 15
sleepedf_task_ds = sleepedf_ds.set_task(lambda x: multi_epoch_multi_modal_sleep_staging_sleepedf_fn(x,modality = modality,num_epoch_seq = num_epoch_seq))


# split dataset
# data split
train_dataset, val_dataset, test_dataset = split_by_patient(sleepedf_task_ds, [0.8, 0.1, 0.1])

# create dataloaders (they are <torch.data.DataLoader> object)
batch_size = 64
train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

# STEP 3: define model
model = Seq_Cross_Modal_Transformer_PyHealth( dataset= sleepedf_task_ds, 
                                            feature_keys= ['signal'], 
                                            label_key= ['label'], 
                                            mode= 'multiclass', 
                                            d_model = 128,
                                            num_epoch_seq = num_epoch_seq,
                                            dim_feedforward=512,
                                            window_size = 50,
                                            num_classes = 6,).to(device)#("cuda:1")

# STEP 4: define trainer
trainer = Trainer(model=model,metrics=["cohen_kappa", "accuracy"])
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=100,
    monitor="cohen_kappa",
    monitor_criterion="max"
)


# STEP 5: evaluate
score = trainer.evaluate(test_loader)
print (score)


# STEP 6: interpret
# test_data = interpret_cmt(trainer,test_loader,num_epoch_seq,num_classes=6)
# interactive_plot_cmt(test_data)