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
  
  # filter based on info in edf file
def skip_file(file_name, signal_headers):
  signal_sample_rate = signal_headers.get_signal_sample_rate()
  if SAMPLE_RATE > signal_sample_rate:
    print(f"[WARN] [{file_name}] Signal sample rate is smaller than required.")
    return True

  label_list_c = [
    sh['label'] for sh in signal_headers
  ]
  if not all(elem in label_list_c for elem in LABEL_LIST):
    print(f"[WARN] [{file_name}] label_list_c: {label_list_c}")
    print(f"[WARN] [{file_name}] Not all labels in is in label list.")
    return True
  return False

# extract labels from tse/csv file
def label_sampling_tuh(df):
  y_target = ""
  remained = 0
  feature_intv = 1 / float(FEATURE_SAMPLE_RATE)

  for _, row in df.iterrows():
    begin = row['start_time']
    end = row['stop_time']
    label = row['label']

    intv_count, remained = divmod(
        float(end) - float(begin) + remained,
        feature_intv
    )
    y_target += int(intv_count) * str(DISEASE_LABELS[label])
  return y_target

def process_label(file_name):
  df = read_file(f"{file_name}.{LABEL_TYPE}")
  return label_sampling_tuh(df)

# extract labels from bi file
def is_seizure_patient(file):
  patient_wise_dir = get_patientwise_dir(file)
  edf_list = search_walk(patient_wise_dir, f".{LABEL_TYPE}_bi")

  for bi_file in edf_list:
    df = read_file(bi_file)
    for label in df["label"]:
      if label != 'bckg':
          return True

  return False

# resample from edf file
def resample(signals, signal_headers):
  signal_sample_rate = signal_headers.get_signal_sample_rate()

  signal_list = []
  signal_label_list = []

  for idx, signal in enumerate(signals):
    label = signal_headers[idx]['label']

    if label not in LABEL_LIST:
      continue

    if int(signal_headers[idx]['sample_frequency']) > SAMPLE_RATE:
      secs = len(signal) / float(signal_sample_rate)
      samps = int(secs * SAMPLE_RATE)
      signal = sci_sig.resample(signal, samps)

    signal_list.append(signal)
    signal_label_list.append(label)

  if len(signal_label_list) != len(LABEL_LIST):
    return

  return [ signal_list[signal_label_list.index(lead)] for lead in LABEL_LIST ]

def transform_labels_with_resampled_signals(signals, y_sampled):
  new_length = len(signals[0]) * (FEATURE_SAMPLE_RATE / SAMPLE_RATE)
  if len(y_sampled) > new_length:
    y_sampled = y_sampled[:int(new_length)]
  else:
    y_sampled += y_sampled[-1] * int(new_length - len(y_sampled))

  y_sampled = [
      "0" if l not in SELECTED_DISEASES else l
      for l in y_sampled
  ]

  if any(l in SELECTED_DISEASES for l in y_sampled):
      y_sampled = [
          str(TARGET_DIR[int(l)]) if l in SELECTED_DISEASES else l
          for l in y_sampled
      ]
  return y_sampled

def get_unique_labels(seq):
  return [x[0] for x in groupby(seq)]

def get_event_label(labels):
  return str(max(map(int, labels)))

def segment_signals(y_sampled, signals, is_seiz_patient):
  raw_data = torch.Tensor(signals).permute(1,0).to(torch.float16)

  min_seg_len_label = MIN_BINARY_SLICELENGTH * FEATURE_SAMPLE_RATE
  min_seg_len_raw = MIN_BINARY_SLICELENGTH * SAMPLE_RATE
  min_edge_label = MIN_BINARY_EDGE_SEIZ * FEATURE_SAMPLE_RATE
  min_edge_raw = MIN_BINARY_EDGE_SEIZ * SAMPLE_RATE

  if len(y_sampled) < min_seg_len_label:
    return None

  def slice_data(y, raw):
    return y[:min_seg_len_label], raw[:min_seg_len_raw].permute(1,0)

  def next_seg(y, raw):
    return y[min_seg_len_label:], raw[min_seg_len_raw:]

  sliced_raws = []
  sliced_labels = []
  label_names = []

  original_y = list(y_sampled)
  original_raw = raw_data
  is_middle_segment = False

  while len(y_sampled) >= min_seg_len_label:
    sliced_y, sliced_raw = slice_data(y_sampled, raw_data)
    unique_labels = get_unique_labels(sliced_y)
    start, end = sliced_y[0], sliced_y[-1]

    if len(unique_labels) == 1 and start == '0':
      label = "0_patT" if is_seiz_patient else "0_patF"

    elif start == '0' and end != '0':
      reversed_y = list(reversed(sliced_y))
      boundary_seizlen = reversed_y.index("0") + 1

      if boundary_seizlen < min_edge_label and len(y_sampled) > min_seg_len_label + min_edge_label:
        sliced_y = y_sampled[min_edge_label:min_seg_len_label+min_edge_label]
        sliced_raw = raw_data[min_edge_raw:min_seg_len_raw+min_edge_raw].permute(1,0)

      label = get_event_label(unique_labels) + "_beg"
      is_middle_segment = True

    elif start != '0' and end != '0':
      label = get_event_label(unique_labels)
      if len(unique_labels) == 1:
        label = label + "_middle"
      else:
        label = label + "_whole"
      is_middle_segment = True

    elif start != '0' and end == '0':
      label = get_event_label(unique_labels) + "_end"

    elif start == '0' and end == '0':
      label = get_event_label(unique_labels)+ "_whole"

    else:
      raise ValueError("Unexpected case encountered")

    y_sampled, raw_data = next_seg(y_sampled, raw_data)

    sliced_raws.append(sliced_raw)
    sliced_labels.append(sliced_y)
    label_names.append(label)

  # edge case: signal ends with seizure
  if is_middle_segment == True:
    sliced_y = original_y[-min_seg_len_label:]
    sliced_raw = original_raw[-min_seg_len_raw:].permute(1,0)

    if sliced_y[-1] == '0':
      label = get_event_label(get_unique_labels(sliced_y)) + "_end"
      sliced_raws.append(sliced_raw)
      sliced_labels.append(sliced_y)
      label_names.append(label)

  return sliced_raws, sliced_labels, label_names

# save file
def save_file(data_file_name, sliced_raws, sliced_labels, label_names):
  # clean up
  pkl_list = search_walk(SAVE_DIR, ".pkl")
  if pkl_list:
    for pkl_file in pkl_list:
      if data_file_name in pkl_file:
        os.remove(pkl_file)

  # create pkl file
  for i, (raw, labels) in enumerate(zip(sliced_raws, sliced_labels)):
    y_map = list(map(int, labels))
    y1 = torch.tensor(y_map).byte()

    def make_binary(target):
        return torch.tensor([target[j] for j in y_map]).byte() if target is not None else None

    y2 = make_binary(BINARY_TARGET1)
    y3 = make_binary(BINARY_TARGET2)

    new_data = {
        'RAW_DATA': [raw],
        'LABEL1': [y1],
        'LABEL2': [y2],
        'LABEL3': [y3],
    }

    label = label_names[i]
    path = f"{SAVE_DIR}/{data_file_name}_c{i}_label_{label}.pkl"

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(new_data, f)

  print(f"Saved {len(sliced_raws)} files for [{data_file_name}].")

  def Convert2PklFile(file):
    # 0. set variables
    signals, signal_headers, _ = highlevel.read_edf(file)
    file_name = ".".join(file.split(".")[:-1])
    data_file_name = file_name.split("/")[-1]
    signal_headers = TuhSignalHeader(signal_headers)

    # 1. skip certain conditions
    if skip_file(file_name, signal_headers):
        print(f"Skipping {file_name}.")
        return

    # 2. get labels and patient status
    y_sampled = process_label(file_name)
    is_seiz_patient = is_seizure_patient(file)

    # 3. resample signals
    signal_final_list_raw = resample(signals, signal_headers)
    if signal_final_list_raw is None:
        print(f"Skipping {file_name}.")
        return

    # 4. transform labels with resampled signals
    y_sampled = transform_labels_with_resampled_signals(signal_final_list_raw, y_sampled)

    # 5. segment signals
    sliced_raws, sliced_labels, label_names = segment_signals(y_sampled, signal_final_list_raw, is_seiz_patient)

    # 6. save file
    save_file(data_file_name, sliced_raws, sliced_labels, label_names)

    print(f"Preprocessing {file_name} successfully completed.")


    #########################
    # main process
    #########################
    # process multiple files in parallel
    edf_list = search_walk(EEG_DATA_DIR, ".edf")
    print(f"Total number of files: {len(edf_list)}")

    # Convert2PklFile(edf_list[0])
    # Convert2PklFile(edf_list[1])
    Convert2PklFile(edf_list[2])
    # Convert2PklFile(edf_list[3])
    # Convert2PklFile(edf_list[4])

    # run_multi_process(
    #     Convert2PklFile,
    #     edf_list,
    #     n_processes=CPU_NUM
    # )