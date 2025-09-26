"""
Jake Cumberland (jakepc3)
Swaroop Potdar (spotd)

Based on the dataset object from Interpretation of Intracardiac
Electrograms Through Textual Representations: https://arxiv.org/abs/2402.01115

Dataset for the Intracardiac Atrial Fibrillation database on Physionet,
meant to unpack data folder into a single dataset.

Data available here:  
https://physionet.org/static/published-projects/iafdb/intracardiac-atrial-fibrillation-database-1.0.0.zip
"""
import os
#Necessary import for working with PhysioNet waveform data
import wfdb
import numpy as np
from typing import Dict, List, Optional

from .base_dataset import BaseDataset

class IAFDBDataset(BaseDataset):
    def __init__ (
            self, 
            root: str,  
            dataset_name: Optional[str] = "IAFDBD", 
            config_path: Optional[str] = None, 
            dev: bool = False,
            segment_length: int = 1000,
            step_size: Optional[int] = None):
        #The table isn't used due to the data being signal only, but it's needed to satisfy BaseDataset.
        tables = [
        ]
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            dev=dev
        )
        self.seg_length = segment_length
        self.samples = self.process()
        
    def __getitem__(self, index: int) -> Dict:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
    
    """
    process handles the reading of preprocessing of data 
    from the intracardiac-atrial-fibrillation-database-1.0.0 folder.

    returns:
    - segmented_data, a dictionary of egm signals
    The key is (current electrode, current placement, current segment, 1)
    The value the key returns is the entire segment at the current electrode/channel and placement

    For example:

    segmented_data[2, 1, 3, 1] would return the entire 3rd segment for the 2nd channel on the 1st placement.
    """
    def process(self) -> Dict:
            #Read in the signals from the path
            egm_signals = self.read_from_path(self.root)
            #Z_score normalized the signals
            signals_normalized = self.z_score_normalization(egm_signals)
            #Split the normalized signals into smaller segments, 1000 is the default because it represents 1 second with this data.
            segmented_data = self.segment_signal(signals_normalized, 1000)

            self.samples = []
            n_segments, segment_length, n_electrodes, n_placements = segmented_data.shape
            segmented_data_dict = {}
            n_segments, segment_length, n_electrodes, n_placements = segmented_data.shape
            for i in range(n_electrodes):
                for j in range(n_placements):
                    for k in range(n_segments):
                        key = (i, j, k, 1)
                        segmented_data_dict[key] = segmented_data[k, :, i, j]
            return segmented_data_dict
    """
    Stat prints some statistics of the dataset such as:

    -Number of channels

    -Segment length
    """
    def stat(self):
        print("dataset statistics:")
        
        shape_of_data = self.samples.shape
        #Data has 8 patients total
        print("Number of patients:", 8)
        #Data has 4 different placements per patient
        print("Number of placements:", 4)

        print("Total samples:", shape_of_data[3])

        print("Number of channels:", shape_of_data[2])

        print("Segment length:", shape_of_data[1])

        print("Number of segments:", shape_of_data[0])

        print("Shape of data:", shape_of_data)
    """
    read_from_path reads in the EGM signals from the specified path

    args:
    -path: a string representing path to the intracardiac-atrial-fibrillation-database-1.0.0

    returns:
    -stacked_array: an ndarray of shape [timesteps, electrodes, placements]

    For use in process()
    """
    def read_from_path(self, path : str) -> np.ndarray:
        all_signals = []
        #Iterate through the dataset folder, if qrs file, record
        for i in os.listdir(path):
            if 'qrs' in i:
                file_name = i.split('.')[0]
                record = wfdb.rdrecord(path + '/' + f'{file_name}')
                egm_signals = record.p_signal[:, 3:]
                all_signals.append(egm_signals)
        min_shape = min(array.shape[0] for array in all_signals)
        sliced_arrays = [array[:min_shape] for array in all_signals]

        stacked_array = np.stack(sliced_arrays, axis=-1)
        return stacked_array
    """
    z_score_normalization applies z_score normalization to the EGM signals array we created in read_from_path

    args:
    -egm_signals: an ndarray of shape [timesteps, electrodes, placements]

    returns:
    -normalized_data: data normalized with the z-score

    For use in process(), after reading the data in.
    """
    def z_score_normalization(self, egm_signals : np.ndarray) -> np.ndarray:
        #We take the average/standard deviation across all timesteps and channels for each patient.
        mean_val = np.nanmean(egm_signals, axis=(0, 1), keepdims=True)
        std_val = np.nanstd(egm_signals, axis=(0, 1), keepdims=True)
        #add this in case our standard deviation at an element is 0.
        std_val[std_val == 0] = 1.0
        normalized_data = (egm_signals - mean_val) / std_val
        return normalized_data

    """
    segment_signal splits our normalized signals into smaller segments, the default segment length is 1000 (1 second)

    args:
    -data: ndarray of data normalized from z_score_normalization with shape [timesteps, electrodes, placements]
    -segment_length: integer, length of each segment, default 1000 which represents 1 second.
    -step_size (optional): integer, how far we slide the window each segmentation, default is None which results in non-overlapping segmentation.

    returns:
    -segmented_data: an ndarray of shape [n_segments, segment_length, n_electrodes, n_placements]

    For use in process, after normalizing the data.
    """
    def segment_signal(self, data, segment_length=1000, step_size = None) -> np.ndarray:
    
        n_time_points, n_electrodes, n_placements = data.shape
        
        if step_size != None:
            n_segments = 1 + (n_time_points - segment_length) // step_size
            segmented_data = np.zeros((n_segments, segment_length, n_electrodes, n_placements))

            for i in range(n_segments):
                start_idx = i * step_size
                end_idx = start_idx + segment_length
                segmented_data[i] = data[start_idx:end_idx, :, :]
                
        elif step_size == None:
            n_segments = n_time_points // segment_length
            truncated_data = data[:n_segments * segment_length]
            segmented_data = truncated_data.reshape(n_segments, segment_length, data.shape[1], data.shape[2])
        
        return segmented_data

testDataset = IAFDBDataset("./intracardiac-atrial-fibrillation-database-1.0.0")