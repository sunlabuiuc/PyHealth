"""
Rohit Rao - Processing TUH Dataset

This code was adapted from the preprocessing code given in:
"Real-Time Seizure Detection using EEG: A Comprehensive Comparison of Recent Approaches under a Realistic Setting"
"""
import os
import argparse
import pickle
import random
import glob
from itertools import groupby
from multiprocessing import Pool

import numpy as np
import torch
from scipy import signal as sci_sig
from pyedflib import highlevel
from tqdm import tqdm

# Global dictionary to hold configuration
GLOBAL_DATA = {}

def search_walk(root_path, extension):
    """Recursively search for files with specific extension."""
    searched_list = []
    if not os.path.exists(root_path):
        print(f"Warning: Path does not exist: {root_path}")
        return []
    
    for (path, dir, files) in os.walk(root_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == extension:
                list_file = os.path.join(path, filename)
                searched_list.append(list_file)
    return searched_list

def run_multi_process(func, file_list, n_processes=40):
    """Run a function in parallel over a list of files."""
    if not file_list:
        return []
    
    n_processes = min(n_processes, len(file_list))
    print(f"Using {n_processes} processes for {len(file_list)} files")

    results = list()
    with Pool(processes=n_processes) as pool:
        for r in tqdm(pool.imap_unordered(func, file_list), total=len(file_list), ncols=75):
            results.append(r)
    return results

def label_sampling_tuh(labels, feature_samplerate):
    """Sample labels based on feature sample rate."""
    y_target = ""
    remained = 0
    feature_intv = 1/float(feature_samplerate)
    for i in labels:
        parts = i.split(" ")
        begin, end = parts[0], parts[1]
        label = parts[2]

        if GLOBAL_DATA['label_type'] == 'tse_bi' and label not in GLOBAL_DATA['disease_labels']:
            if label != 'bckg':
                label = 'seiz'
        
        if label not in GLOBAL_DATA['disease_labels']:
            continue

        intv_count, remained = divmod(float(end) - float(begin) + remained, feature_intv)
        y_target += int(intv_count) * str(GLOBAL_DATA['disease_labels'][label])
    return y_target

def read_label_file(file_name, label_type):
    """Read annotation file (.tse, .tse_bi, or .csv_bi)."""
    label_file_path = file_name + "." + label_type
    if not os.path.exists(label_file_path) and label_type == 'tse_bi':
        label_file_path = file_name + ".csv_bi"
    
    try:
        with open(label_file_path, 'r') as label_file:
            y = label_file.readlines()
    except FileNotFoundError:
        return None, None
    
    # Skip header
    y = list(y[2:])
    
    if label_file_path.endswith('.csv_bi'):
        y = [line for line in y if line.strip() and not line.strip().startswith('#')]
        if y and y[0].startswith('channel,'):
            y = y[1:]
        y_labels = list(set([line.split(",")[3].strip() for line in y if line.strip()]))
        # Convert CSV format to TSE-like format for processing
        y = [f"{line.split(',')[1]} {line.split(',')[2]} {line.split(',')[3]}" for line in y if line.strip()]
    else:
        y_labels = list(set([i.split(" ")[2] for i in y if len(i.split(" ")) > 2]))
    
    return y, y_labels

def generate_training_data_leadwise_tuh_train_final(file):
    """Process a single EDF file for training."""
    try:
        sample_rate = GLOBAL_DATA['sample_rate']
        file_name = os.path.splitext(file)[0]
        data_file_name = os.path.basename(file_name)
        
        signals, signal_headers, header = highlevel.read_edf(file)
        
        label_list_c = []
        for idx, signal in enumerate(signals):
            label_noref = signal_headers[idx]['label'].split("-")[0]
            label_list_c.append(label_noref)   

        y, y_labels = read_label_file(file_name, GLOBAL_DATA['label_type'])
        if y is None:
            return

        signal_sample_rate = int(signal_headers[0]['sample_frequency'])
        if sample_rate > signal_sample_rate:
            return
        if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']):
            return

        y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

        # Determine if patient has history
        patient_bool = False
        patient_wise_dir = os.path.dirname(os.path.dirname(file_name))
        edf_list = search_walk(patient_wise_dir, ".tse_bi")
        if not edf_list:
            edf_list = search_walk(patient_wise_dir, ".csv_bi")
        
        if edf_list:
            for label_file_path in edf_list:
                y_hist, _ = read_label_file(os.path.splitext(label_file_path)[0], GLOBAL_DATA['label_type'])
                if y_hist is None:
                    continue
                for line in y_hist:
                    parts = line.split(" ")
                    if len(parts) > 2 and parts[2] != 'bckg':
                        patient_bool = True
                        break
                if patient_bool:
                    break
        
        signal_list = []
        signal_label_list = []
        signal_final_list_raw = []

        for idx, signal in enumerate(signals):
            label = signal_headers[idx]['label'].split("-")[0]
            if label not in GLOBAL_DATA['label_list']:
                continue

            if int(signal_headers[idx]['sample_frequency']) > sample_rate:
                secs = len(signal)/float(signal_sample_rate)
                samps = int(secs*sample_rate)
                x = sci_sig.resample(signal, samps)
                signal_list.append(x)
                signal_label_list.append(label)
            else:
                signal_list.append(signal)
                signal_label_list.append(label)
        
        for lead_signal in GLOBAL_DATA['label_list']:
            signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

        new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
        
        if len(y_sampled) > new_length:
            y_sampled = y_sampled[:int(new_length)]
        elif len(y_sampled) < new_length:
            diff = int(new_length - len(y_sampled))
            if len(y_sampled) > 0:
                y_sampled += y_sampled[-1] * diff

        y_sampled_np = np.array(list(map(int, y_sampled)))

        # Map labels
        y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]
        if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
            y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

        new_data = {}
        raw_data = torch.from_numpy(np.array(signal_final_list_raw)).permute(1,0)
        raw_data = raw_data.type(torch.float16)
        
        min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
        min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
        min_binary_edge_seiz_label = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['feature_sample_rate']
        min_binary_edge_seiz_raw = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['sample_rate']

        sliced_raws = []
        sliced_labels = []
        label_list_for_filename = []
        
        if len(y_sampled) < min_seg_len_label:
            return
        
        label_count = {}
        y_sampled_2nd = list(y_sampled)
        raw_data_2nd = raw_data

        while len(y_sampled) >= min_seg_len_label:
            is_at_middle = False
            sliced_y = y_sampled[:min_seg_len_label]
            labels = [x[0] for x in groupby(sliced_y)]
                
            if len(labels) == 1 and "0" in labels:
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                label = "0_patT" if patient_bool else "0_patF"
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)
                
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] != '0'):
                temp_sliced_y = list(sliced_y)
                temp_sliced_y.reverse()
                try:
                    boundary_seizlen = temp_sliced_y.index("0") + 1
                except ValueError:
                    boundary_seizlen = 0 # Should not happen based on logic

                if boundary_seizlen < min_binary_edge_seiz_label:
                    if len(y_sampled) > (min_seg_len_label + min_binary_edge_seiz_label):
                        sliced_y = y_sampled[min_binary_edge_seiz_label:min_seg_len_label+min_binary_edge_seiz_label]
                        sliced_raw_data = raw_data[min_binary_edge_seiz_raw:min_seg_len_raw+min_binary_edge_seiz_raw].permute(1,0)
                    else:
                        sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                else:
                    sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)

                y_sampled = y_sampled[min_seg_len_label:]
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_beg"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) != 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_whole"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) == 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_middle"
                label_list_for_filename.append(label)
                is_at_middle = True
            
            elif len(labels) != 1 and (sliced_y[0] != '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_end"
                label_list_for_filename.append(label)
            
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_whole"
                label_list_for_filename.append(label)
            
            else:
                # Fallback
                y_sampled = y_sampled[min_seg_len_label:]
                raw_data = raw_data[min_seg_len_raw:]

        if is_at_middle:
            sliced_y = y_sampled_2nd[-min_seg_len_label:]
            sliced_raw_data = raw_data_2nd[-min_seg_len_raw:].permute(1,0)
            
            if sliced_y[-1] == '0':
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label = label + "_end"
                label_list_for_filename.append(label)

        # Save to disk
        for data_idx in range(len(sliced_raws)):
            sliced_raw = sliced_raws[data_idx]
            sliced_y = sliced_labels[data_idx]
            sliced_y_map = list(map(int,sliced_y))
            sliced_y = torch.Tensor(sliced_y_map).byte()

            sliced_y2 = None
            if GLOBAL_DATA['binary_target1'] is not None:
                sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()

            sliced_y3 = None
            if GLOBAL_DATA['binary_target2'] is not None:
                sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()

            new_data['RAW_DATA'] = [sliced_raw.cpu().numpy().astype(np.float16)]
            # Convert sliced_y (list of strings) to numpy array
            sliced_y_array = np.array([int(x) for x in sliced_y], dtype=np.uint8)
            new_data['LABEL1'] = [sliced_y_array]
            new_data['LABEL2'] = [sliced_y2.cpu().numpy().astype(np.uint8) if sliced_y2 is not None else None]
            new_data['LABEL3'] = [sliced_y3.cpu().numpy().astype(np.uint8) if sliced_y3 is not None else None]

            label = label_list_for_filename[data_idx]
            
            save_path = os.path.join(GLOBAL_DATA['data_file_directory'], 
                                     f"{data_file_name}_c{data_idx}_label_{label}.pkl")
            with open(save_path, 'wb') as _f:
                pickle.dump(new_data, _f, protocol=pickle.HIGHEST_PROTOCOL)      
            new_data = {}
    except Exception as e:
        print(f"Error processing {file}: {e}")

def generate_training_data_leadwise_tuh_dev(file):
    """Process a single EDF file for development/test."""
    try:
        sample_rate = GLOBAL_DATA['sample_rate']
        file_name = os.path.splitext(file)[0]
        data_file_name = os.path.basename(file_name)
        
        signals, signal_headers, header = highlevel.read_edf(file)
        
        label_list_c = []
        for idx, signal in enumerate(signals):
            label_noref = signal_headers[idx]['label'].split("-")[0]
            label_list_c.append(label_noref)   

        y, y_labels = read_label_file(file_name, GLOBAL_DATA['label_type'])
        if y is None:
            print(f"Warning: Could not read label file for {file_name}")
            return

        signal_sample_rate = int(signal_headers[0]['sample_frequency'])
        if sample_rate > signal_sample_rate:
            print(f"Warning: Sample rate {sample_rate} > signal sample rate {signal_sample_rate} for {file_name}")
            return
        if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']):
            missing = [elem for elem in GLOBAL_DATA['label_list'] if elem not in label_list_c]
            print(f"Warning: Missing channels {missing} in {file_name}. Found: {label_list_c}")
            return
        
        y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

        signal_list = []
        signal_label_list = []
        signal_final_list_raw = []

        for idx, signal in enumerate(signals):
            label = signal_headers[idx]['label'].split("-")[0]
            if label not in GLOBAL_DATA['label_list']:
                continue

            if int(signal_headers[idx]['sample_frequency']) > sample_rate:
                secs = len(signal)/float(signal_sample_rate)
                samps = int(secs*sample_rate)
                x = sci_sig.resample(signal, samps)
                signal_list.append(x)
                signal_label_list.append(label)
            else:
                signal_list.append(signal)
                signal_label_list.append(label)
        
        for lead_signal in GLOBAL_DATA['label_list']:
            signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

        new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
        
        if len(y_sampled) > new_length:
            y_sampled = y_sampled[:int(new_length)]
        elif len(y_sampled) < new_length:
            diff = int(new_length - len(y_sampled))
            if len(y_sampled) > 0:
                y_sampled += y_sampled[-1] * diff

        # Convert string to list of strings for processing (convert chars to strings)
        y_sampled = [str(c) for c in y_sampled]
        y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]
        if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
            y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

        new_data = {}
        raw_data = torch.from_numpy(np.array(signal_final_list_raw)).permute(1,0)
        raw_data = raw_data.type(torch.float16)
        
        slice_end_margin_length = GLOBAL_DATA.get('slice_end_margin_length', 5)
        min_end_margin_label = slice_end_margin_length * GLOBAL_DATA['feature_sample_rate']

        min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
        min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
        
        sliced_raws = []
        sliced_labels = []
        label_list_for_filename = []

        if len(y_sampled) < min_seg_len_label:
            print(f"Warning: y_sampled length {len(y_sampled)} < min_seg_len_label {min_seg_len_label} for {file_name}")
            return
        
        while len(y_sampled) >= min_seg_len_label:
            one_left_slice = False
            sliced_y = y_sampled[:min_seg_len_label]
                
            if (sliced_y[-1] == '0'):
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                y_sampled = y_sampled[min_seg_len_label:]

                labels = [x[0] for x in groupby(sliced_y)]
                if (len(labels) == 1) and (labels[0] == '0'):
                    label = "0"
                else:
                    label = ("".join(labels)).replace("0", "")[0]
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            else:
                if '0' in y_sampled[min_seg_len_label:]:
                    end_1 = y_sampled[min_seg_len_label:].index('0')
                    temp_y_sampled = list(y_sampled[min_seg_len_label+end_1:])
                    temp_y_sampled_order = [x[0] for x in groupby(temp_y_sampled)]

                    if len(list(set(temp_y_sampled))) == 1:
                        end_2 = len(temp_y_sampled)
                        one_left_slice = True
                    else:
                        end_2 = temp_y_sampled.index(temp_y_sampled_order[1])

                    if end_2 >= min_end_margin_label:
                        temp_sec = random.randint(1, slice_end_margin_length)
                        temp_seg_len_label = int(min_seg_len_label + (temp_sec * GLOBAL_DATA['feature_sample_rate']) + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_sec * GLOBAL_DATA['sample_rate']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))
                    else:
                        if one_left_slice:
                            temp_label = end_2
                        else:
                            temp_label = end_2 // 2

                        temp_seg_len_label = int(min_seg_len_label + temp_label + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_label * GLOBAL_DATA['fsr_sr_ratio']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))

                    sliced_y = y_sampled[:temp_seg_len_label]
                    sliced_raw_data = raw_data[:temp_seg_len_raw].permute(1,0)
                    raw_data = raw_data[temp_seg_len_raw:]
                    y_sampled = y_sampled[temp_seg_len_label:]

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
                else:
                    sliced_y = y_sampled[:]
                    sliced_raw_data = raw_data[:].permute(1,0)
                    raw_data = []
                    y_sampled = []

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
            
        for data_idx in range(len(sliced_raws)):
            sliced_raw = sliced_raws[data_idx]
            sliced_y = sliced_labels[data_idx]
            sliced_y_map = list(map(int,sliced_y))

            sliced_y2 = None
            if GLOBAL_DATA['binary_target1'] is not None:
                sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()

            sliced_y3 = None
            if GLOBAL_DATA['binary_target2'] is not None:
                sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()

            new_data['RAW_DATA'] = [sliced_raw.cpu().numpy().astype(np.float16)]
            # Convert sliced_y (list of strings) to numpy array
            sliced_y_array = np.array([int(x) for x in sliced_y], dtype=np.uint8)
            new_data['LABEL1'] = [sliced_y_array]
            new_data['LABEL2'] = [sliced_y2.cpu().numpy().astype(np.uint8) if sliced_y2 is not None else None]
            new_data['LABEL3'] = [sliced_y3.cpu().numpy().astype(np.uint8) if sliced_y3 is not None else None]

            label = label_list_for_filename[data_idx]
            
            save_path = os.path.join(GLOBAL_DATA['data_file_directory'], 
                                     f"{data_file_name}_c{data_idx}_len{len(sliced_y)}_label_{label}.pkl")
            
            with open(save_path, 'wb') as _f:
                pickle.dump(new_data, _f, protocol=pickle.HIGHEST_PROTOCOL)      
            new_data = {}
    except Exception as e:
        print(f"Error processing {file}: {e}")

def main(args):
    if not os.path.exists(args.data_folder):
        raise ValueError(f"Data folder does not exist: {args.data_folder}")
    
    save_directory = args.save_directory
    data_type = args.data_type
    dataset = args.dataset
    label_type = args.label_type
    sample_rate = args.samplerate
    cpu_num = args.cpu_num
    feature_type = args.feature_type
    feature_sample_rate = args.feature_sample_rate
    task_type = args.task_type
    data_file_directory = os.path.join(save_directory, f"dataset-{dataset}_task-{task_type}_datatype-{data_type}_v6")
    
    labels = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',  
                    'EEG C3', 'EEG C4', 'EEG CZ', 'EEG T3', 'EEG T4', 
                    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    eeg_data_directory = args.data_folder
    
    if label_type == "tse":
        disease_labels =  {'bckg': 0, 'cpsz': 1, 'mysz': 2, 'gnsz': 3, 'fnsz': 4, 'tnsz': 5, 'tcsz': 6, 'spsz': 7, 'absz': 8}
    elif label_type == "tse_bi":
        disease_labels =  {'bckg': 0, 'seiz': 1}
        # Explicitly set disease_type if we are in binary mode and it wasn't customized
        if 'seiz' not in args.disease_type:
            args.disease_type = ['seiz']
    
    disease_labels_inv = {v: k for k, v in disease_labels.items()}
    
    edf_list1 = search_walk(eeg_data_directory, ".edf")
    edf_list2 = search_walk(eeg_data_directory, ".EDF")
    edf_list = edf_list1 + edf_list2
    
    if not edf_list:
        raise ValueError(f"No EDF files found in {eeg_data_directory}")

    if os.path.exists(data_file_directory):
        import shutil
        shutil.rmtree(data_file_directory)
    os.makedirs(data_file_directory, exist_ok=True)

    GLOBAL_DATA['label_list'] = labels
    GLOBAL_DATA['disease_labels'] = disease_labels
    GLOBAL_DATA['disease_labels_inv'] = disease_labels_inv
    GLOBAL_DATA['data_file_directory'] = data_file_directory
    GLOBAL_DATA['label_type'] = label_type
    GLOBAL_DATA['feature_type'] = feature_type
    GLOBAL_DATA['feature_sample_rate'] = feature_sample_rate
    GLOBAL_DATA['sample_rate'] = sample_rate
    GLOBAL_DATA['fsr_sr_ratio'] = (sample_rate // feature_sample_rate)
    GLOBAL_DATA['min_binary_slicelength'] = args.min_binary_slicelength
    GLOBAL_DATA['min_binary_edge_seiz'] = args.min_binary_edge_seiz
    GLOBAL_DATA['slice_end_margin_length'] = args.slice_end_margin_length

    target_dictionary = {0:0}
    selected_diseases = []
    
    # Handle list input for disease_type properly
    if isinstance(args.disease_type, list):
        d_types = args.disease_type
    else:
        d_types = args.disease_type.split()

    for idx, i in enumerate(d_types):
        if i in disease_labels:
            selected_diseases.append(str(disease_labels[i]))
            target_dictionary[disease_labels[i]] = idx + 1
    
    GLOBAL_DATA['disease_type'] = d_types
    GLOBAL_DATA['target_dictionary'] = target_dictionary
    GLOBAL_DATA['selected_diseases'] = selected_diseases
    
    # Parse binary targets if passed as string (from CLI sometimes comes as string)
    if isinstance(args.binary_target1, str):
        import ast
        GLOBAL_DATA['binary_target1'] = ast.literal_eval(args.binary_target1)
    else:
        GLOBAL_DATA['binary_target1'] = args.binary_target1
        
    if isinstance(args.binary_target2, str):
        import ast
        GLOBAL_DATA['binary_target2'] = ast.literal_eval(args.binary_target2)
    else:
        GLOBAL_DATA['binary_target2'] = args.binary_target2

    print("########## Preprocessor Setting Information ##########")
    print("Data folder: ", eeg_data_directory)
    print("Number of EDF files: ", len(edf_list))
    
    with open(os.path.join(data_file_directory, 'preprocess_info.infopkl'), 'wb') as pkl:
        pickle.dump(GLOBAL_DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("################ Preprocess begins... ################\n")
    
    if (task_type == "binary") and (data_type == "train"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
    elif (task_type == "binary") and (data_type == "dev"):
        use_dev_function = getattr(args, 'use_dev_function', False)
        if use_dev_function:
            run_multi_process(generate_training_data_leadwise_tuh_dev, edf_list, n_processes=cpu_num)
        else:
            run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
    else:
        print(f"Warning: Unsupported task_type={task_type} and data_type={data_type} combination")
        
    print("################ Preprocess completed! ################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process TUH EEG dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_folder', '-df', type=str, required=True,
                        help='Path to the TUH dataset folder (train or dev)')
    parser.add_argument('--data_type', '-dt', type=str,
                        default='train',
                        choices=['train', 'dev'],
                        help='Dataset type: train or dev')
    parser.add_argument('--save_directory', '-sp', type=str, required=True,
                        help='Path to save processed data')
    parser.add_argument('--samplerate', '-sr', type=int, default=200,
                        help='Sample Rate (Hz)')
    parser.add_argument('--label_type', '-lt', type=str,
                        default='tse_bi',
                        choices=['tse', 'tse_bi'],
                        help='Label type')
    parser.add_argument('--cpu_num', '-cn', type=int, default=1,
                        help='Number of CPU processes to use')
    parser.add_argument('--feature_type', '-ft', type=str,
                        default='rawsignal',
                        help='Feature type')
    parser.add_argument('--feature_sample_rate', '-fsr', type=int, default=50,
                        help='Feature sample rate (Hz)')
    parser.add_argument('--dataset', '-st', type=str,
                        default='tuh',
                        choices=['tuh'],
                        help='Dataset name')
    parser.add_argument('--task_type', '-tt', type=str,
                        default='binary',
                        choices=['anomaly', 'multiclassification', 'binary'],
                        help='Task type')
    parser.add_argument('--seed', '-sd', type=int, default=1004,
                        help='Random seed number')
    parser.add_argument('--disease_type', type=str, nargs='+',
                        default=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'],
                        help='List of disease types to include')
    parser.add_argument('--binary_target1', type=str,
                        default="{0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}",
                        help='Binary target mapping 1 (pass as dict string)')
    parser.add_argument('--binary_target2', type=str,
                        default="{0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5}",
                        help='Binary target mapping 2 (pass as dict string)')
    parser.add_argument('--min_binary_slicelength', type=int, default=30,
                        help='Minimum binary slice length (seconds)')
    parser.add_argument('--min_binary_edge_seiz', type=int, default=3,
                        help='Minimum binary edge seizure length (seconds)')
    parser.add_argument('--slice_end_margin_length', type=int, default=5,
                        help='Slice end margin length (seconds)')
    parser.add_argument('--use_dev_function', action='store_true',
                        help='Use dev-specific processing function')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)