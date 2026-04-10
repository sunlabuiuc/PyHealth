import os
from pyedflib import highlevel
import pandas as pd
from scipy import signal as sci_sig
import torch
import numpy as np
from itertools import groupby
import pickle
import random

from pathlib import Path

def get_label_type(label_type):
  if label_type == "tse" or label_type == 'csv':
      return {'bckg':0,'cpsz':1,'mysz':2,'gnsz':3,'fnsz':4,'tnsz':5,'tcsz':6,'spsz':7,'absz':8}
  else:
      return {'bckg':0,'seiz':1}

def set_selected_diseases(disease_type, disease_labels):
  return [str(disease_labels[i]) for i in disease_type]

def set_target_dir(disease_type, disease_labels):
  target_dictionary = {0:0}
  for idx, i in enumerate(disease_type):
    target_dictionary[disease_labels[i]] = idx + 1
  return target_dictionary

def search_walk(full_path, extension):
    return [
        ('%s/%s' % (path, filename))
        for path, _, files in os.walk(full_path)
        for filename in files
        if Path(path).suffix == extension
    ]

def read_file(file):
    if LABEL_TYPE == 'tse':
      return pd.read_csv(
        file,
        comment="#",
        sep=r"\s+",
        names=["start_time", "stop_time", "label", "confidence"]
      )
    return pd.read_csv(
      file,
      comment="#",
      sep=","
    )

def get_patientwise_dir(file):
  patient_wise_dir = "/".join(file.split("/")[:-2])
  return patient_wise_dir

class TuhSignalHeader:
  def __init__(self, signal_headers):
    self.signal_headers = self.__extractlabels__(signal_headers)
    self.signal_sample_rate = int(self.signal_headers[0]['sample_frequency'])

  def __extractlabels__(self, signal_headers):
    for sh in signal_headers:
      sh['label'] = sh['label'].split("-")[0]
    return signal_headers

  def get_signal_sample_rate(self):
    return self.signal_sample_rate

  def __getitem__(self, key):
    return self.signal_headers[key]
  

SAMPLE_RATE            = 200
FEATURE_SAMPLE_RATE    = 50
CPU_NUM                = 32
DATASET                = 'tuh'
DATA_TYPE              = 'dev'
TASK_TYPE              = 'binary'
LABEL_TYPE             = 'csv' # 'tse'
MIN_BINARY_SLICELENGTH = 30
MIN_BINARY_EDGE_SEIZ   = 3
BINARY_TARGET1         = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}
BINARY_TARGET2         = {0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5}
DISEASE_TYPE           = ['gnsz','fnsz','spsz','cpsz','absz','tnsz','tcsz','mysz'] # ['seiz']
DISEASE_LABELS         = get_label_type(LABEL_TYPE)
SELECTED_DISEASES      = set_selected_diseases(DISEASE_TYPE, DISEASE_LABELS)
TARGET_DIR             = set_target_dir(DISEASE_TYPE, DISEASE_LABELS)
SAVE_DIR               = f"/content/tuh_eeg_preprocessed/{DATA_TYPE}"
EEG_DATA_DIR           = f"tuh_eeg_v2.0.5/{DATA_TYPE}"
LABEL_LIST = [
  'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
  'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
  'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
]

OUTPUT_DIM          = 8
BINARY_SAMPLER_TYPE = '30types'
DATA_TYPE           = 'training dataset'
DEV_BCKG_NUM        = 0
IS_TRAINING_SET     = DATA_TYPE == 'training dataset'
SEIZURE_TO_NUM      = {'gnsz': '0', 'fnsz': '1', 'spsz': '2', 'cpsz': '3', 'absz': '4', 'tnsz': '5', 'tcsz': '6', 'mysz': '7'}
LABEL_GROUP         = 'LABEL1'
EEG_TYPE            = 'uni_bipolar'
BATCH_SIZE          = 32

def create_bipolar_signals(signals):
    """
    Convert raw EEG signals into bipolar montage.
    Each channel becomes the difference between two electrodes.
    """
    electrode_pairs = [
        (0, 4), (1, 5), (4, 9), (5, 10),
        (9, 15), (10, 16), (15, 13), (16, 14),
        (9, 6), (7, 10), (6, 8), (8, 7),
        (0, 2), (1, 3), (2, 6), (3, 7),
        (6, 11), (7, 12), (11, 13), (12, 14)
    ]

    return [signals[a] - signals[b] for a, b in electrode_pairs]


def skip_process(type1, patient_dev_dict, patient_id):
  if type1 == "8":
    return OUTPUT_DIM == 8 or BINARY_SAMPLER_TYPE == "30types"
  if IS_TRAINING_SET:
    return False
  if (type1 == "0") and (patient_dev_dict[patient_id][0] >= DEV_BCKG_NUM):
    return True
  if (type1 != "0") and (patient_dev_dict[patient_id][2] >= DEV_BCKG_NUM):
    return True
  return False

def extract_label(parts):
  if BINARY_SAMPLER_TYPE == "6types":
    return parts[-1].split(".")[0]
  elif BINARY_SAMPLER_TYPE == "30types":
    return "_".join(parts[-2:]).split(".")[0]
  else:
    raise ValueError("Invalid sampler type")
  
def init_dev_patient_dict(patient_dev_dict, patient_id):
  if not IS_TRAINING_SET:
    patient_dev_dict[patient_id] = [0, 0, 0]
  return patient_dev_dict

def update_dev_patient_dict(patient_dev_dict, patient_id, type1, path):
  if IS_TRAINING_SET:
    return patient_dev_dict

  if type1 == "0":
    patient_dev_dict[patient_id][0] += 1
  elif "middle" in path:
    patient_dev_dict[patient_id][2] += 1
  else:
    patient_dev_dict[patient_id][1] += 1

  return patient_dev_dict

def print_summary(unique_label_list, type_detail1, type_detail2, patient_dev_dict):
  print("########## Summary of {} ##########".format(DATA_TYPE))
  print("Types of types for sampler: ", unique_label_list)
  print("Number of types for sampler: ", len(unique_label_list))
  print("--- Normal Slices Info ---")
  print("Patient normal slices size: ", type_detail1.count("0_patT"))
  print("Non-Patient normal slices size: ", type_detail1.count("0_patF"))
  print("Total normal slices size: ", type_detail2.count("0"))
  print("--- Seizure Slices Info ---")
  total_seiz_slices_num = 0
  for idx, seizure in enumerate(args.seiz_classes):
    seiz_num = SEIZURE_TO_NUM[seizure]
    beg_slice_num = type_detail1.count(seiz_num + "_beg")
    middle_slice_num = type_detail1.count(seiz_num + "_middle")
    end_slice_num = type_detail1.count(seiz_num + "_end")
    whole_slice_num = type_detail1.count(seiz_num + "_whole")
    total_seiz_num = type_detail2.count(seiz_num)
    total_seiz_slices_num += total_seiz_num
    print("Number of {} slices: total:{} - beg:{}, middle:{}, end:{}, whole:{}".format(seizure, str(total_seiz_num), str(beg_slice_num), str(middle_slice_num), str(end_slice_num), str(whole_slice_num)))
  print("Total seizure slices: ", str(total_seiz_slices_num))
  print("Dataset Prepared...\n")

  if "training dataset" != DATA_TYPE:
    print("Number of patients: ", len(patient_dev_dict))
    for pat_info in patient_dev_dict:
      pat_normal, pat_seiz, pat_middle = patient_dev_dict[pat_info]
      print("(Non-)Patient: {} has normals:{}, seizures:{}, mid_seizures:{}".format(pat_info, str(pat_normal), str(pat_seiz), str(pat_middle)))
      num_normals += pat_normal
      num_seizures_boundaries += pat_seiz
      num_seizures_middles += pat_middle

      print("Total normals:{}, seizures with boundaries:{}, seizures with middles:{}".format(str(num_normals), str(num_seizures_boundaries), str(num_seizures_middles)))

def create_weighted_sampler(unique_label_list, label_idx_in_unique_list):
    class_counts = np.bincount(label_idx_in_unique_list)
    weights = 1.0 / class_counts

    # 6types did nothing in the original script
    if BINARY_SAMPLER_TYPE == "30types":
        if "0_patT" in unique_label_list:
            idx = unique_label_list.index("0_patT")
            weights[idx] *= 7
        if "0_patF" in unique_label_list:
            idx = unique_label_list.index("0_patF")
            weights[idx] *= 7
    else:
        print("No control on sampler rate")

    sample_weights = weights[label_idx_in_unique_list]
    sample_weights = torch.from_numpy(sample_weights).double()

    return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

def eeg_collate_fn(file_list):
    batch = []

    for file_path in file_list:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        signals = data['RAW_DATA'][0]
        labels = data[LABEL_GROUP][0]

        if EEG_TYPE == "bipolar":
            signals = torch.stack(create_bipolar_signals(signals))

        elif EEG_TYPE == "uni_bipolar":
            bipolar = torch.stack(create_bipolar_signals(signals))
            signals = torch.cat((signals, bipolar))

        batch.append((signals, labels, file_path))

    batch_size = len(batch)
    max_seq_len = max(s[0].shape[1] for s in batch)
    num_channels = batch[0][0].shape[0]
    max_label_len = max(len(s[1]) for s in batch)

    seqs = torch.zeros(batch_size, max_seq_len, num_channels)
    targets = torch.zeros(batch_size, max_label_len, dtype=torch.long)

    seq_lengths = []
    target_lengths = []
    filenames = []

    for i, (signals, labels, fname) in enumerate(batch):
        seq_len = signals.shape[1]

        signals = signals.permute(1, 0)

        seqs[i, :seq_len] = signals
        targets[i, :len(labels)] = torch.tensor(labels)

        seq_lengths.append(seq_len)
        target_lengths.append(len(labels))
        filenames.append(os.path.basename(fname))

    return seqs, targets, seq_lengths, target_lengths, filenames


def process_files(files):
  files = []
  labels = []
  unique_label_list = []              # type_type = []    # unique label list (order preserved)
  label_idx_in_unique_list = []      # type_indices = [] # numeric labels
  _type_detail1 = []
  _type_detail2 = []
  patient_dev_dict = {}

  for path in files:
    parts = path.split("_")
    type1, type2 = parts[-2], parts[-1].split(".")[0]

    # get patient id and add to dictionary
    filename = os.path.basename(path)
    patient_id = filename.split("_")[0]
    patient_dev_dict = init_dev_patient_dict(patient_dev_dict, patient_id)

    # 1. skip file
    if skip_process(type1, patient_dev_dict, patient_id):
      continue

    # 2. extract label
    label = extract_label(parts)

    # 3. update dev dictionary
    patient_dev_dict = update_dev_patient_dict(patient_dev_dict, patient_id, type1, path)

    # 4. update label and index list
    if label not in unique_label_list:
      unique_label_list.append(label)
    label_idx_in_unique_list.append(unique_label_list.index(label))
    files.append(path)
    labels.append(label)

    # (optional) print summary for debugging
    _type_detail1.append(f"{type1}_{type2}")
    _type_detail2.append(type1)
    print_summary(unique_label_list, _type_detail1, _type_detail2, patient_dev_dict)

    return files, labels, unique_label_list, label_idx_in_unique_list


    #########################
    # main process
    #########################

    # PyHealth workflow:
    #     1) dataset = TuszDataset(root=<dataset_dir>, tables=<table_list>)
    #        dataset.get_patient('1')
    #     2) task = TuszTask()
    #        dataset = dataset.set_task(task)
    #     3) train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.1, 0.2], seed=SEED)
    #     4) train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn_dict)

    # Following steps are from original github
    # NOTES
    #     Step 0, 1, 3 & 4 should be translated into 1), 2), and 3) in PyHealth
    #     Step 2 & 5 correspond to 4) in PyHealth
    #     1), 2), and 3) in PyHealth include all substeps from Process 1 and Process 2



    # 0. load directories
    train_data_path = f"{SAVE_DIR}/train"
    # dev_data_path = f"{SAVE_DIR}/dev"
    # test_data_path = f"{SAVE_DIR}/test"

    # 1. get files for train/dev/test
    train_dir = search_walk(train_data_path, ".pkl")
    # dev_dir = search_walk({"path": dev_data_path, "extension": ".pkl"})

    # 2. shuffle for train/dev/test
    random.shuffle(train_dir)
    # random.shuffle(dev_dir)

    # 3. process files in train/dev/test
    files, labels, unique_label_list, label_indx_in_unique_list = process_files(train_dir)
    # files, labels, unique_label_list, label_indx_in_unique_list = process_files(dev_dir)

    # 4. weight labels
    sampler = create_weighted_sampler(unique_label_list, label_indx_in_unique_list)

    # 5. create dataloader
    #     originally: github code has Detector_Dataset class, which inherits from torch.utils.data.Dataset:
    #         train_data = Detector_Dataset(...)
    #     now: train_data will become TuszDataset class
    #         references of similar data from the Temple University:
    #             PyHealth/pyhealth/datasets/tuab.py: class TUABDataset(BaseDataset)
    #             PyHealth/pyhealth/datasets/tuev.py: class TUEVDataset(BaseDataset)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        sampler=sampler, # only in train dataset
        collate_fn=eeg_collate_fn,
        shuffle = True, # True in train and val/dev dataset
    )
