"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link:
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
import logging
import torch
import numpy as np
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

TYPES_6 = "6types"
TYPES_30 = "30types"

class TUSZSampler():
    """Sampler for TUSZ Dataset implements how samples are chosen.

    Dataset is available at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml.
    
    Args:
        dataset        : an instance of BaseDataset.
        is_training_set: is dataset a training set.
    """
    def __init__(self, 
        dataset: List[Dict[str, Any]],
        is_training_set: Optional[bool] = True,
        **kwargs
    ) -> None:
        self.dataset = dataset
        self.is_training_set = is_training_set
        self.output_dim = kwargs['output_dim'] if 'output_dim' in kwargs else 8
        self.binary_sampler_type = kwargs['binary_sampler_type'] if 'binary_sampler_type' in kwargs else '6types'
        self.dev_bckg_num = kwargs['dev_bckg_num'] if 'dev_bckg_num' in kwargs else 3
        self.unique_labels = []             # unique label list (order preserved)
        self.unique_label_indices = []      # numeric labels


    def get_weights(self) -> torch.Tensor:
        """Returns weights for each label."""
        self.__get_unique_labels()
        
        class_counts = np.bincount(self.unique_label_indices)
        weights = 1.0 / class_counts

        # 6types does nothing
        if self.binary_sampler_type == TYPES_30:
            if "0_patT" in self.unique_labels:
                idx = self.unique_labels.index("0_patT")
                weights[idx] *= 7
            if "0_patF" in self.unique_labels:
                idx = self.unique_labels.index("0_patF")
                weights[idx] *= 7
        else:
            logger.warning("No control on sampler rate")

        sample_weights = weights[self.unique_label_indices]
        sample_weights = torch.from_numpy(sample_weights).double()

        return sample_weights
    
    def __get_unique_labels(self) -> None:
        """Returns a unique set of labels."""
        patient_dev_dict = {}
        for sample in self.dataset:
            patient_id = sample['patient_id']
            label_name = sample['label_name']
            patient_dev_dict = self.__init_dev_patient_dict(patient_dev_dict, patient_id)

            if self.__skip_process(label_name, patient_dev_dict, patient_id):
                continue

            label = self.__extract_label(label_name)
                                
            patient_dev_dict = self.__update_dev_patient_dict(patient_dev_dict, patient_id, label_name)

            if label not in self.unique_labels:
                self.unique_labels.append(label)
            self.unique_label_indices.append(self.unique_labels.index(label))

    def __skip_process(self,
        label_name: str,
        patient_dev_dict: dict[str, list[int]],
        patient_id: str,
    ) -> bool:
        """Skips patients for certain conditions."""
        if label_name == "8":
            return self.output_dim == 8 or self.binary_sampler_type == TYPES_30
        if self.is_training_set:
            return False
        if (label_name == "0") and (patient_dev_dict[patient_id][0] >= self.dev_bckg_num):
            return True
        if (label_name != "0") and (patient_dev_dict[patient_id][2] >= self.dev_bckg_num):
            return True
        return False

    def __init_dev_patient_dict(self,
        patient_dev_dict: dict[str, list[int]],
        patient_id: str,
    ) -> dict[str, list[int]]:
        """Initializes dictionary for patient labels."""
        if not self.is_training_set:
            patient_dev_dict[patient_id] = [0, 0, 0]
        return patient_dev_dict

    def __extract_label(self, label_name: str) -> str:
        """Returns processed label name."""
        if self.binary_sampler_type == TYPES_6:
            return label_name
        elif self.binary_sampler_type == TYPES_30:
            return f"label_{label_name}"
        else:
            raise ValueError("Invalid sampler type")

    def __update_dev_patient_dict(self,
        patient_dev_dict: dict[str, list[int]],
        patient_id: str,
        label_name: str,
    ) -> dict[str, list[int]]:
        """Updates patient-label dictionary according to the label name."""
        if self.is_training_set:
            return patient_dev_dict

        if label_name == "0":
            patient_dev_dict[patient_id][0] += 1
        elif "middle" in label_name:
            patient_dev_dict[patient_id][2] += 1
        else:
            patient_dev_dict[patient_id][1] += 1

        return patient_dev_dict
