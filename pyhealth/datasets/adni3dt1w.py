"""
ADNI 3D MRI T1w dataset loader.
Team NetID: lu93, mina6
Paper: On the design of convolutional neural networks for automatic detection of Alzheimer’s disease
Link: https://proceedings.mlr.press/v116/liu20a

The dataset used in the paper is the Alzheimer’s Disease
Neuroimaging Initiative (ADNI) dataset (adn 2008), which
is a widely used public dataset for Alzheimer’s disease
research. The authors also validate their model on the 
Australian Imaging, Biomarkers and Lifestyle (AIBL) study
dataset (Ellis et al. 2009).

Preprocessing is required before using this dataset class.
Please follow the following detailed instructions to download and preprocess the data. 
https://github.com/luzhangyi319/cs598-DLH?tab=readme-ov-file#data-preparation

"""

import os
import numpy as np
import polars as pl
import pandas as pd
import nibabel as nib
import random
import pickle
import scipy
import logging
from typing import Optional, List, Tuple, Dict
from pyhealth.datasets import BaseDataset
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)

class ADNI3DT1W(BaseDataset):
    """Dataset for ADNI 3D MRI T1w dataset.
    
    The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a longitudinal study
    designed to develop clinical, imaging, genetic, and biochemical biomarkers 
    for the early detection and tracking of Alzheimer's disease.
    
    This dataset loader handles the T1-weighted MRI volume data from ADNI.
    
    Args:
        dataset_name: name of the dataset.
        root: root directory of the image data. The following is the folder structure.
            You can find more details from https://aramislab.paris.inria.fr/clinica/docs/public/dev/CAPS/Specifications/#t1-volume-pipeline---volume-based-processing-of-t1-weighted-mr-images
                dataset_description.json
                subjects/
                └─ <participant_id>/
                └─ <session_id>/
                    └─ t1/
                        └─ spm/
                            └─ segmentation/
                            ├─ normalized_space/
                            │  ├─ <source_file>_target-Ixi549Space_transformation-{inverse|forward}_deformation.nii.gz
                            │  ├─ <source_file>_segm-<segm>_space-Ixi549Space_modulated-{on|off}_probability.nii.gz
                            │  └─ <source_file>_space-Ixi549Space_T1w.nii.gz
                            ├─ native_space/
                            │  └─ <source_file>_segm-<segm>_probability.nii.gz
                            └─ dartel_input/
                                └─ <source_file>_segm-<segm>_dartelinput.nii.gz
        metadata_tsv: the directory contains Train/Val/Test metadata tsv files
            Under metadata_tsv folder, there should be six files named as below,
                1. Test_ADNI.tsv
                2. Test_diagnosis_ADNI.tsv
                3. Train_ADNI.tsv
                4. Train_diagnosis_ADNI.tsv
                5. Val_ADNI.tsv
                6. Val_diagnosis_ADNI.tsv
            Header of ADNI.tsv files is:
                (base) Zhangyis-MBP:sample_200 zhangyi$ head Train_ADNI.tsv
                    participant_id	session_id
                    sub-ADNI023S0926	ses-M060
                    sub-ADNI033S0920	ses-M072
            Header of *_diagnoisis file is:
                (base) Zhangyis-MBP:sample_200 zhangyi$ head Train_diagnosis_ADNI.tsv
                    participant_id	session_id	diagnosis	mmse	cdr	cdr_sb	age	examination_date	earliest_time	age_rounded
                    sub-ADNI023S0926	ses-M060	CN	29.0	3.0	4.0	76.4	2007-03-01	2007-03-01	76.5
                    sub-ADNI033S0920	ses-M072	CN	30.0	4.0	4.0	86.0	2007-03-01	2007-03-01	86.0
                    sub-ADNI006S4346	ses-M012	MCI	29.0			72.3	2007-03-01	2007-03-01	72.5
                    sub-ADNI002S1268	ses-M048	MCI	28.0	4.0	4.0	86.7	2007-03-01	2007-03-01	86.5
                    sub-ADNI029S0999	ses-M000	AD	25.0			70.8	2007-03-01	2007-03-01	71.0
        mode: Train/Val/Test
        n_label: the number of labels
        dev: whether to enable dev mode (only use a small subset of the data).
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated.
    
    Usage example:
        from pyhealth.datasets import ADNI3DT1W
        dataset = ADNI3DT1W(
                "ADNI3D",
                "/Users/zhangyi/Downloads/ADNI_processed_200/subjects",
                "/Users/zhangyi/Downloads/ADNI_converted_meta_all/sample_200",
                )
        dataset.stat()
    """
    
    def __init__(
        self,
        dataset_name: str,
        root: str,
        metadata_tsv: str,
        mode = 'Train', 
        n_label = 3,
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root=root,
            dev=dev,
            refresh_cache=refresh_cache,
        )
        self.metadata_tsv = metadata_tsv
        self.mode = mode
        self.n_label = n_label

        if n_label == 3:
            LABEL_MAPPING = ["CN", "MCI", "AD"]
        elif n_label == 2:
            LABEL_MAPPING = ["CN", "AD"]
        self.LABEL_MAPPING = LABEL_MAPPING    

    def parse_data(self):
        try:
            subject_tsv = pd.io.parsers.read_csv(os.path.join(self.metadata_tsv, self.mode + '_diagnosis_ADNI.tsv'), sep='\t')
            all_data = []
            idx = 0
            for i in range(len(subject_tsv)):
                if (subject_tsv.iloc[i].diagnosis not in self.LABEL_MAPPING):
                    continue
                participant_id = subject_tsv.iloc[i].participant_id
                session_id = subject_tsv.iloc[i].session_id
                path = os.path.join(participant_id, session_id,'t1/spm/segmentation/normalized_space')
                all_segs = list(os.listdir(path))
                if subject_tsv.iloc[i].diagnosis == 'CN':
                    label = 0
                elif subject_tsv.iloc[i].diagnosis == 'MCI':
                    label = 1
                elif subject_tsv.iloc[i].diagnosis == 'AD':
                    if self.LABEL_MAPPING == ["CN", "AD"]:
                        label = 1
                    else:
                        label = 2
                else:
                    print('WRONG LABEL VALUE!!!')
                    label = -100
                mmse = subject_tsv.iloc[i].mmse
                cdr_sub = subject_tsv.iloc[i].cdr
                age = list(np.arange(0.0,120.0,0.5)).index(self.subject_tsv.iloc[i].age_rounded) 

                for seg_name in all_segs:
                    if 'Space_T1w' in seg_name:
                        # image = nib.load(os.path.join(path,seg_name)).get_data().squeeze()
                        image = np.asanyarray(nib.load(os.path.join(path,seg_name)).dataobj).squeeze()
                        image[np.isnan(image)] = 0.0
                        image = (image - image.min())/(image.max() - image.min() + 1e-6)
            
                        if self.mode == 'Train':
                            image = self.augment_image(image)

                image = np.expand_dims(image,axis =0)

                if self.mode == 'Train':
                    image = self.randomCrop(image,96,96,96)
                else:
                    image = self.centerCrop(image,96,96,96)
                
                all_data.append({
                    "patient_id": participant_id,
                    "visit_id": session_id,
                    "image": image,
                    "label": label,
                    "idx": idx,
                    "mmse": mmse,
                    "cdr": cdr_sub,
                    "age": age
                })
                idx += 1
            return all_data
        
        except Exception as e:
            logger.error(f"Failed to load #{i}: {path}")
            logger.error(f"Errors encountered: {e}")
            return []

    def load_data(self) -> pl.LazyFrame:
        all_data = self.parse_data()
        try:
            df = pl.DataFrame(all_data, schema={
                "patient_id": pl.Utf8,
                "visit_id": pl.Utf8,
                "image": pl.Object,
                "label": pl.Int64,
                "idx": pl.Int64,
                "mmse": pl.Float32,
                "cdr": pl.Float32,
                "age": pl.Float32
            })
        except Exception as e:
            logger.error(f"Failed to create Polars DataFrame: {e}")

        return df     
        
    def centerCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height

        x = img.shape[1]//2 - length//2
        y = img.shape[2]//2 - width//2
        z = img.shape[3]//2 - height//2
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img

    def randomCrop(self, img, length, width, height):
        assert img.shape[1] >= length
        assert img.shape[2] >= width
        assert img.shape[3] >= height
        x = random.randint(0, img.shape[1] - length)
        y = random.randint(0, img.shape[2] - width)
        z = random.randint(0, img.shape[3] - height )
        img = img[:,x:x+length, y:y+width, z:z+height]
        return img
    
    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        image = scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
        return image

    def unpickling(self, path):
       file_return=pickle.load(open(path,'rb'))
       return file_return
    

if __name__ == "__main__":
    dataset = ADNI3DT1W(
        "ADNI3D",
        "/Users/zhangyi/Downloads/ADNI_processed_200/subjects",
        "/Users/zhangyi/Downloads/ADNI_converted_meta_all/sample_200",
        )
    dataset.stat()