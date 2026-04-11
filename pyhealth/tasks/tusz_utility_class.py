import logging
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import signal as sci_sig
from itertools import groupby

logger = logging.getLogger(__name__)

LOG_INFO = 'info'
LOG_WARN = 'warn'
LOG_ERR = 'err'

class TUSZHelper:
    def __init__(self,
          sample_rate,
          feature_sample_rate,
          label_type,
          eeg_type,
          min_binary_slicelength,
          min_binary_edge_seiz,
    ):
        self.sample_rate            = sample_rate
        self.feature_sample_rate    = feature_sample_rate
        self.label_type             = label_type
        self.eeg_type               = eeg_type
        self.min_binary_slicelength = min_binary_slicelength
        self.min_binary_edge_seiz   = min_binary_edge_seiz
        self.binary_target1         = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}
        self.binary_target2         = {0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5}
        self.disease_type           = ['gnsz','fnsz','spsz','cpsz','absz','tnsz','tcsz','mysz'] # ['seiz']
        self.disease_labels         = self.__get_label_type()
        self.selected_diseases      = self.__set_selected_diseases()
        self.target_dir             = self.__set_target_dir()
        self.label_list             = [
            'EEG FP1','EEG FP2','EEG F3','EEG F4','EEG F7','EEG F8',
            'EEG C3','EEG C4','EEG CZ','EEG T3','EEG T4',
            'EEG P3','EEG P4','EEG O1','EEG O2','EEG T5','EEG T6','EEG PZ','EEG FZ'
        ]

    ##################################################
    # substeps
    ##################################################
    def skip_file(self, file_name, signal_headers):
        signal_sample_rate = signal_headers.get_signal_sample_rate()
        if self.sample_rate > signal_sample_rate:
            self.log(LOG_WARN, file_name, "*** signal sample rate is smaller than required.")
            return True

        label_list_c = [
            sh['label'] for sh in signal_headers
        ]
        if not all(elem in label_list_c for elem in self.label_list):
            self.log(LOG_WARN, file_name, f"*** label_list_c: {label_list_c}")
            self.log(LOG_WARN, file_name, "*** not all labels in is in label list.")
            return True
        return False

    def is_seizure_patient(self, file):
        patient_wise_dir = self.__get_patientwise_dir(file)
        edf_list = self.__search_walk(patient_wise_dir, f".{self.label_type}_bi")

        for bi_file in edf_list:
            df = self.__read_file(bi_file)
            for label in df["label"]:
                if label != 'bckg':
                    return True

        return False

    def process_label(self, file_name):
        df = self.__read_file(f"{file_name}.{self.label_type}")
        y_target = ""
        remained = 0
        feature_intv = 1 / float(self.feature_sample_rate)

        for _, row in df.iterrows():
            begin = row['start_time']
            end = row['stop_time']
            label = row['label']
            intv_count, remained = divmod(
                float(end) - float(begin) + remained,
                feature_intv
            )
            y_target += int(intv_count) * str(self.disease_labels[label])
        return y_target
    
    # resample from edf file
    def resample(self, file_name, signals, signal_headers):
        signal_sample_rate = signal_headers.get_signal_sample_rate()

        signal_list = []
        signal_label_list = []

        for idx, signal in enumerate(signals):
            label = signal_headers[idx]['label']

            if label not in self.label_list:
                continue

            if int(signal_headers[idx]['sample_frequency']) > self.sample_rate:
                secs = len(signal) / float(signal_sample_rate)
                samps = int(secs * self.sample_rate)
                signal = sci_sig.resample(signal, samps)

            signal_list.append(signal)
            signal_label_list.append(label)

        if len(signal_label_list) != len(self.label_list):
            self.log(LOG_WARN, file_name, f"not enough labels: {signal_label_list}")
            return []

        return [ signal_list[signal_label_list.index(lead)] for lead in self.label_list ]

    def transform_labels_with_resampled_signals(self, signals, y_sampled):
        new_length = len(signals[0]) * (self.feature_sample_rate / self.sample_rate)
        if len(y_sampled) > new_length:
            y_sampled = y_sampled[:int(new_length)]
        else:
            y_sampled += y_sampled[-1] * int(new_length - len(y_sampled))

        y_sampled = [
            "0" if label not in self.selected_diseases else label
            for label in y_sampled
        ]

        if any(label in self.selected_diseases for label in y_sampled):
            y_sampled = [
                str(self.target_dir[int(label)]) if label in self.selected_diseases else label
                for label in y_sampled
            ]
        return y_sampled

    def segment_signals(self, y_sampled, signals, is_seiz_patient):
        raw_data = torch.Tensor(np.array(signals)).permute(1,0).to(torch.float16)

        min_seg_len_label = self.min_binary_slicelength * self.feature_sample_rate
        min_seg_len_raw = self.min_binary_slicelength * self.sample_rate
        min_edge_label = self.min_binary_edge_seiz * self.feature_sample_rate
        min_edge_raw = self.min_binary_edge_seiz * self.sample_rate

        if len(y_sampled) < min_seg_len_label:
            return [], [], []

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
            unique_labels = self.__get_unique_labels(sliced_y)
            start, end = sliced_y[0], sliced_y[-1]

            if len(unique_labels) == 1 and start == '0':
                label = "0_patT" if is_seiz_patient else "0_patF"

            elif start == '0' and end != '0':
                reversed_y = list(reversed(sliced_y))
                boundary_seizlen = reversed_y.index("0") + 1

                if boundary_seizlen < min_edge_label and (
                    len(y_sampled) > min_seg_len_label + min_edge_label
                ):
                    sliced_y = y_sampled[min_edge_label:min_seg_len_label+min_edge_label]
                    sliced_raw = raw_data[min_edge_raw:min_seg_len_raw+min_edge_raw].permute(1,0)

                label = self.__get_event_label(unique_labels) + "_beg"
                is_middle_segment = True

            elif start != '0' and end != '0':
                label = self.__get_event_label(unique_labels)
                if len(unique_labels) == 1:
                    label = label + "_middle"
                else:
                    label = label + "_whole"
                is_middle_segment = True

            elif start != '0' and end == '0':
                label = self.__get_event_label(unique_labels) + "_end"

            elif start == '0' and end == '0':
                label = self.__get_event_label(unique_labels)+ "_whole"

            else:
                raise ValueError("Unexpected case encountered")

            y_sampled, raw_data = next_seg(y_sampled, raw_data)

            sliced_raws.append(sliced_raw)
            sliced_labels.append(sliced_y)
            label_names.append(label)

        # edge case: signal ends with seizure
        if is_middle_segment:
            sliced_y = original_y[-min_seg_len_label:]
            sliced_raw = original_raw[-min_seg_len_raw:].permute(1,0)

            if sliced_y[-1] == '0':
                label = self.__get_event_label(self.__get_unique_labels(sliced_y)) + "_end"
                sliced_raws.append(sliced_raw)
                sliced_labels.append(sliced_y)
                label_names.append(label)

        # sliced_raws: [(19, 6000), (19, 6000)] shape=(2, 19, 6000)
        # sliced_labels: shape=(2, 1500)
        # label_names: shape=(2, 1)
        return sliced_raws, sliced_labels, label_names

    def convert_labels(self, sliced_labels):
        y1, y2, y3 = [], [], []
        for i, labels in enumerate(sliced_labels):
            y_map = list(map(int, labels))
            y1.append(torch.tensor(y_map).byte())

            def make_binary(target):
                return torch.tensor([target[j] for j in y_map]).byte() if target is not None else None

            y2.append(make_binary(self.binary_target1))
            y3.append(make_binary(self.binary_target2))

        return y1, y2, y3
    
    def create_bipolar_signals(self, signals):
        electrode_pairs = [
            (0, 4), (1, 5), (4, 9), (5, 10),
            (9, 15), (10, 16), (15, 13), (16, 14),
            (9, 6), (7, 10), (6, 8), (8, 7),
            (0, 2), (1, 3), (2, 6), (3, 7),
            (6, 11), (7, 12), (11, 13), (12, 14)
        ]
        stacked_signals = torch.stack([
            signals[a] - signals[b] for a, b in electrode_pairs
        ])
        if self.eeg_type == "uni_bipolar":
            signals = torch.cat((signals, stacked_signals))

        return stacked_signals

    ##################################################
    # utilities
    ##################################################
    def __read_file(self, file):
        if self.label_type == 'tse':
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
    
    def __get_patientwise_dir(self, file):
        patient_wise_dir = "/".join(file.split("/")[:-2])
        return patient_wise_dir
    
    def __search_walk(self, full_path, extension):
        return [
            ('%s/%s' % (path, filename))
            for path, _, files in os.walk(full_path)
            for filename in files
            if Path(path).suffix == extension
        ]

    def __get_unique_labels(self, seq):
        return [x[0] for x in groupby(seq)]

    def __get_event_label(self, labels):
        return str(max(map(int, labels)))

    def log(self, level, file_name, msg):
        if level == LOG_INFO:
            logger.info(f"[{file_name}] {msg}")
        elif level == LOG_WARN:
            logger.warning(f"[{file_name}] {msg}")
        elif level == LOG_ERR:
            logger.error(f"[{file_name}] {msg}")


    ##################################################
    # variable functions
    ##################################################
    def __get_label_type(self):
        if self.label_type == "tse" or self.label_type == 'csv':
            return {'bckg':0,'cpsz':1,'mysz':2,'gnsz':3,'fnsz':4,'tnsz':5,'tcsz':6,'spsz':7,'absz':8}
        else:
            return {'bckg':0,'seiz':1}

    def __set_selected_diseases(self):
        return [str(self.disease_labels[i]) for i in self.disease_type]

    def __set_target_dir(self):
        target_dictionary = {0:0}
        for idx, i in enumerate(self.disease_type):
            target_dictionary[self.disease_labels[i]] = idx + 1
        return target_dictionary


class TUSZSignalHeader:
    def __init__(self, signal_headers):
        self.signal_headers = self.__extract_labels__(signal_headers)
        self.signal_sample_rate = int(self.signal_headers[0]['sample_frequency'])

    def __extract_labels__(self, signal_headers):
        for sh in signal_headers:
            sh['label'] = sh['label'].split("-")[0]
        return signal_headers

    def get_signal_sample_rate(self):
        return self.signal_sample_rate

    def __getitem__(self, key):
        return self.signal_headers[key]
