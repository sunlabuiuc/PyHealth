import os
import pickle
import pkg_resources
from scipy.io import loadmat
import pandas as pd
import numpy as np


def cardiology_isAR_fn(record, epoch_sec=10, shift=5):
    """Processes a single patient for the Arrhythmias symptom in cardiology on the CardiologyDataset

    Cardiology symptoms can be divided into six categories. The task focuses on Arrhythmias and is defined as a binary classification.

    Args:
        record: a singleton list of one subject from the CardiologyDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_sec: how long will each epoch be (in seconds). 
        shift: the step size for the sampling window (with a width of epoch_sec)
        

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Sex": gender, "Age": age, Y": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> isAR = CardiologyDataset(
        ...         root="physionet.org/files/challenge-2020/1.0.2/training",
                    chosen_dataset=[1,1,1,1,1,1], 
        ...     )
        >>> from pyhealth.tasks import cardiology_isAR_fn
        >>> cardiology_ds = isAR.set_task(cardiology_isAR_fn)
        >>> cardiology_ds.samples[0]
        {
            'patient_id': '0_0',
            'visit_id': 'A0033',
            'record_id': 1,
            'Sex': ['Female'],
            'Age': ['34'],
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/46c18f2a1a18803b4707a934a577a331/0_0-0.pkl',
            'label': '0'
        }
    """

    # these are the AR diseases codes
    AR_space = list(
        map(
            str,
            [
                164889003,
                164890007,
                426627000,
                284470004,
                427172004,
                427393009,
                426177001,
                427084000,
                63593006,
                17338001,
            ],
        )
    )
    
    samples = []
    for visit in record:
        root, pid, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )
        
        # X load
        X = loadmat(os.path.join(root, signal))["val"]
        label_content =  open(os.path.join(root, label), "r").readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), \
                label_content[-5].split(" ")[-1][:-1].split(","), \
                label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(AR_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-AR-{index}.pkl")
            
                pickle.dump(
                    {"signal": X[:, (500 * shift) * index : (500 * shift) * index + 5000], "label": y},
                    open(save_file_path, "wb"),
                )
                
                samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": signal.split(".")[0],
                        "record_id": len(samples) + 1,
                        "Sex": Sex,
                        "Age": Age,
                        "epoch_path": save_file_path,
                        "label": y,
                    }
                )
    return samples

def cardiology_isBBBFB_fn(record, epoch_sec=10, shift=5):
    """Processes a single patient for the Bundle branch blocks and fascicular blocks symptom in cardiology on the CardiologyDataset

    Cardiology symptoms can be divided into six categories. The task focuses on Bundle branch blocks and fascicular blocks and is defined as a binary classification.

    Args:
        record: a singleton list of one subject from the CardiologyDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_sec: how long will each epoch be (in seconds). 
        shift: the step size for the sampling window (with a width of epoch_sec)
        

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Sex": gender, "Age": age, Y": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> isBBBFB = CardiologyDataset(
        ...         root="physionet.org/files/challenge-2020/1.0.2/training",
                    chosen_dataset=[1,1,1,1,1,1], 
        ...     )
        >>> from pyhealth.tasks import cardiology_isBBBFB_fn
        >>> cardiology_ds = isBBBFB.set_task(cardiology_isBBBFB_fn)
        >>> cardiology_ds.samples[0]
        {
            'patient_id': '0_0',
            'visit_id': 'A0033',
            'record_id': 1,
            'Sex': ['Female'],
            'Age': ['34'],
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/46c18f2a1a18803b4707a934a577a331/0_0-0.pkl',
            'label': '0'
        }
    """

    # these are the diseases codes for Bundle branch blocks and fascicular blocks symptom
    BBBFB_space = list(
        map(
            str,
            [
                713427006,
                713426002,
                445118002,
                164909002,
                59118001,
            ],
        )
    )
    
    samples = []
    for visit in record:
        root, pid, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )
        
        # X load
        X = loadmat(os.path.join(root, signal))["val"]
        label_content =  open(os.path.join(root, label), "r").readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(BBBFB_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-BBBFB-{index}.pkl")
            
                pickle.dump(
                    {"signal": X[:, (500 * shift) * index : (500 * shift) * index + 5000], "label": y},
                    open(save_file_path, "wb"),
                )
                
                samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": signal.split(".")[0],
                        "record_id": len(samples) + 1,
                        "Sex": Sex,
                        "Age": Age,
                        "epoch_path": save_file_path,
                        "label": y,
                    }
                )
                
    return samples


def cardiology_isAD_fn(record, epoch_sec=10, shift=5):
    """Processes a single patient for the Axis deviations symptom in cardiology on the CardiologyDataset

    Cardiology symptoms can be divided into six categories. The task focuses on Axis deviations and is defined as a binary classification.

    Args:
        record: a singleton list of one subject from the CardiologyDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_sec: how long will each epoch be (in seconds). 
        shift: the step size for the sampling window (with a width of epoch_sec)
        

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Sex": gender, "Age": age, Y": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> isAD = CardiologyDataset(
        ...         root="physionet.org/files/challenge-2020/1.0.2/training",
                    chosen_dataset=[1,1,1,1,1,1], 
        ...     )
        >>> from pyhealth.tasks import cardiology_isAD_fn
        >>> cardiology_ds = isAD.set_task(cardiology_isAD_fn)
        >>> cardiology_ds.samples[0]
        {
            'patient_id': '0_0',
            'visit_id': 'A0033',
            'record_id': 1,
            'Sex': ['Female'],
            'Age': ['34'],
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/46c18f2a1a18803b4707a934a577a331/0_0-0.pkl',
            'label': '0'
        }
    """
    
    
    # these are the diseases codes for Axis deviations symptom 
    AD_space = list(
        map(
            str,
            [
                39732003,
                47665007,
            ],
        )
    )
    
    samples = []
    for visit in record:
        root, pid, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )
        
        # X load
        X = loadmat(os.path.join(root, signal))["val"]
        label_content =  open(os.path.join(root, label), "r").readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(AD_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-AD-{index}.pkl")
            
                pickle.dump(
                    {"signal": X[:, (500 * shift) * index : (500 * shift) * index + 5000], "label": y},
                    open(save_file_path, "wb"),
                )
                
                samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": signal.split(".")[0],
                        "record_id": len(samples) + 1,
                        "Sex": Sex,
                        "Age": Age,
                        "epoch_path": save_file_path,
                        "label": y,
                    }
                )

    return samples

def cardiology_isCD_fn(record, epoch_sec=10, shift=5):
    """Processes a single patient for the Conduction delays symptom in cardiology on the CardiologyDataset

    Cardiology symptoms can be divided into six categories. The task focuses on Conduction delays and is defined as a binary classification.

    Args:
        record: a singleton list of one subject from the CardiologyDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_sec: how long will each epoch be (in seconds). 
        shift: the step size for the sampling window (with a width of epoch_sec)
        

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Sex": gender, "Age": age, Y": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> isCD = CardiologyDataset(
        ...         root="physionet.org/files/challenge-2020/1.0.2/training",
                    chosen_dataset=[1,1,1,1,1,1], 
        ...     )
        >>> from pyhealth.tasks import cardiology_isCD_fn
        >>> cardiology_ds = isCD.set_task(cardiology_isCD_fn)
        >>> cardiology_ds.samples[0]
        {
            'patient_id': '0_0',
            'visit_id': 'A0033',
            'record_id': 1,
            'Sex': ['Female'],
            'Age': ['34'],
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/46c18f2a1a18803b4707a934a577a331/0_0-0.pkl',
            'label': '0'
        }
    """

    # these are the diseases codes for Conduction delays symptom
    CD_space = list(
        map(
            str,
            [
                270492004,
                698252002,
                164947007,
                111975006,
            ],
        )
    )
    
    samples = []
    for visit in record:
        root, pid, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )
        
        # X load
        X = loadmat(os.path.join(root, signal))["val"]
        label_content =  open(os.path.join(root, label), "r").readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(CD_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-CD-{index}.pkl")
            
                pickle.dump(
                    {"signal": X[:, (500 * shift) * index : (500 * shift) * index + 5000], "label": y},
                    open(save_file_path, "wb"),
                )
                
                samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": signal.split(".")[0],
                        "record_id": len(samples) + 1,
                        "Sex": Sex,
                        "Age": Age,
                        "epoch_path": save_file_path,
                        "label": y,
                    }
                )

    return samples


def cardiology_isWA_fn(record, epoch_sec=10, shift=5):
    """Processes a single patient for the Wave abnormalities symptom in cardiology on the CardiologyDataset

    Cardiology symptoms can be divided into six categories. The task focuses on Wave abnormalities and is defined as a binary classification.

    Args:
        record: a singleton list of one subject from the CardiologyDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_sec: how long will each epoch be (in seconds). 
        shift: the step size for the sampling window (with a width of epoch_sec)
        

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Sex": gender, "Age": age, Y": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> isWA = CardiologyDataset(
        ...         root="physionet.org/files/challenge-2020/1.0.2/training",
                    chosen_dataset=[1,1,1,1,1,1], 
        ...     )
        >>> from pyhealth.tasks import cardiology_isWA_fn
        >>> cardiology_ds = isWA.set_task(cardiology_isWA_fn)
        >>> cardiology_ds.samples[0]
        {
            'patient_id': '0_0',
            'visit_id': 'A0033',
            'record_id': 1,
            'Sex': ['Female'],
            'Age': ['34'],
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/46c18f2a1a18803b4707a934a577a331/0_0-0.pkl',
            'label': '0'
        }
    """

    # these are the diseases codes for Wave abnormalities symptom
    WA_space = list(
        map(
            str,
            [
                164917005,
                164934002,
                59931005,
            ],
        )
    )
    
    samples = []
    for visit in record:
        root, pid, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )
        
        # X load
        X = loadmat(os.path.join(root, signal))["val"]
        label_content =  open(os.path.join(root, label), "r").readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")


        y = 1 if set(Dx).intersection(WA_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-WA-{index}.pkl")
            
                pickle.dump(
                    {"signal": X[:, (500 * shift) * index : (500 * shift) * index + 5000], "label": y},
                    open(save_file_path, "wb"),
                )
                
                samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": signal.split(".")[0],
                        "record_id": len(samples) + 1,
                        "Sex": Sex,
                        "Age": Age,
                        "epoch_path": save_file_path,
                        "label": y,
                    }
                )

    return samples




if __name__ == "__main__":
    from pyhealth.datasets import CardiologyDataset

    #index for cardiology symptoms
    """
    Arrhythmias
    Bundle branch blocks and fascicular blocks
    Axis deviations
    Conduction delays
    Wave abnormalities
    """

    dataset = CardiologyDataset(
        root="/srv/local/data/physionet.org/files/challenge-2020/1.0.2/training",
        dev=True,
        refresh_cache=True,
    )
    sleep_staging_ds = dataset.set_task(cardiology_isAR_fn)
    print(sleep_staging_ds.samples[0])
    # print(sleep_staging_ds.patient_to_index)
    # print(sleep_staging_ds.record_to_index)
    print(sleep_staging_ds.input_info)
    
    
    
    
