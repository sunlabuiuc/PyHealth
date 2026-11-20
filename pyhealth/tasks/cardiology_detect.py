import os
import pickle
from fractions import Fraction
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly

from pyhealth.tasks.base_task import BaseTask


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
        with open(os.path.join(root, label), "r") as label_file:
            label_content = label_file.readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), \
                label_content[-5].split(" ")[-1][:-1].split(","), \
                label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(AR_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-AR-{index}.pkl")
            
                with open(save_file_path, "wb") as epoch_file:
                    pickle.dump(
                        {
                            "signal": X[
                                :, (500 * shift) * index : (500 * shift) * index + 5000
                            ],
                            "label": y,
                        },
                        epoch_file,
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
        with open(os.path.join(root, label), "r") as label_file:
            label_content = label_file.readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(BBBFB_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-BBBFB-{index}.pkl")
            
                with open(save_file_path, "wb") as epoch_file:
                    pickle.dump(
                        {
                            "signal": X[
                                :, (500 * shift) * index : (500 * shift) * index + 5000
                            ],
                            "label": y,
                        },
                        epoch_file,
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
        with open(os.path.join(root, label), "r") as label_file:
            label_content = label_file.readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(AD_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-AD-{index}.pkl")
            
                with open(save_file_path, "wb") as epoch_file:
                    pickle.dump(
                        {
                            "signal": X[
                                :, (500 * shift) * index : (500 * shift) * index + 5000
                            ],
                            "label": y,
                        },
                        epoch_file,
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
        with open(os.path.join(root, label), "r") as label_file:
            label_content = label_file.readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")

        y = 1 if set(Dx).intersection(CD_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-CD-{index}.pkl")
            
                with open(save_file_path, "wb") as epoch_file:
                    pickle.dump(
                        {
                            "signal": X[
                                :, (500 * shift) * index : (500 * shift) * index + 5000
                            ],
                            "label": y,
                        },
                        epoch_file,
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
        with open(os.path.join(root, label), "r") as label_file:
            label_content = label_file.readlines()
        Dx, Sex, Age = label_content[-4].split(" ")[-1][:-1].split(","), label_content[-5].split(" ")[-1][:-1].split(","), label_content[-6].split(" ")[-1][:-1].split(",")


        y = 1 if set(Dx).intersection(WA_space) else 0
       
        
        # frequency * seconds (500 * 10)
        if X.shape[1] >= 500 * epoch_sec:
            for index in range((X.shape[1] - 500 * epoch_sec) // (500 * shift) + 1):
                save_file_path = os.path.join(save_path, f"{pid}-WA-{index}.pkl")
            
                with open(save_file_path, "wb") as epoch_file:
                    pickle.dump(
                        {
                            "signal": X[
                                :, (500 * shift) * index : (500 * shift) * index + 5000
                            ],
                            "label": y,
                        },
                        epoch_file,
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


#: SNOMED CT codes of each diagnosis category, as in the 1.x task functions
#: above: arrhythmias (AR), bundle branch blocks and fascicular blocks
#: (BBBFB), axis deviations (AD), conduction delays (CD) and wave
#: abnormalities (WA).
CARDIOLOGY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "AR": (
        "164889003",
        "164890007",
        "426627000",
        "284470004",
        "427172004",
        "427393009",
        "426177001",
        "427084000",
        "63593006",
        "17338001",
    ),
    "BBBFB": ("713427006", "713426002", "445118002", "164909002", "59118001"),
    "AD": ("39732003", "47665007"),
    "CD": ("270492004", "698252002", "164947007", "111975006"),
    "WA": ("164917005", "164934002", "59931005"),
}


def _text(value: Any) -> str:
    """Returns ``value`` as text, with missing values (None, NaN) as ``""``.

    pandas reads the empty ``age`` and ``sex`` cells of the metadata CSV back
    as NaN, which would otherwise become the string ``"nan"``.
    """
    return "" if pd.isna(value) else str(value)


class CardiologyDetection(BaseTask):
    """Binary detection of one diagnosis category in 12-lead ECG windows.

    Works with :class:`~pyhealth.datasets.CardiologyDataset`, and is the
    PyHealth 2.0 form of the 1.x functions ``cardiology_isAR_fn``,
    ``cardiology_isBBBFB_fn``, ``cardiology_isAD_fn``, ``cardiology_isCD_fn``
    and ``cardiology_isWA_fn``. For each recording it:

    1. reads the ``.mat`` signal (12 leads, in µV),
    2. resamples it from the recording's own rate to ``target_fs``,
    3. cuts ``epoch_sec`` windows every ``shift`` seconds (a recording
       shorter than one window gives no samples), and
    4. labels every window 1 if any of the recording's diagnoses is in the
       category (:data:`CARDIOLOGY_CATEGORIES`), else 0.

    With the defaults, the 500 Hz sources give the same windows as the 1.x
    functions. Three things differ from 1.x:

    - Labels come from the header's ``# Dx:`` line. The 1.x functions read
      it at a fixed position from the end of the header, which on Georgia
      headers (one comment line shorter) is the ``# Sex:`` line, so every
      Georgia recording was labeled 0.
    - The 1.x functions assumed 500 Hz everywhere, so their "10 s" windows
      were 5 s of PTB (1000 Hz) and 19.5 s of INCART (257 Hz).
    - The 1.x functions always cut 5,000 samples, whatever ``epoch_sec`` was.

    Each sample contains:

    - ``signal``: float tensor, shape ``(n_leads, epoch_sec * target_fs)``
    - ``label``: 1 if the recording has a diagnosis of the category, else 0
    - ``record_id``, ``source``, ``window`` (window index), ``age`` and ``sex``

    Args:
        category: ``"AR"``, ``"BBBFB"``, ``"AD"``, ``"CD"`` or ``"WA"``.
        epoch_sec: Window length in seconds. Default is 10.
        shift: Step between window starts in seconds. Default is 5.
        target_fs: Sampling rate after resampling. Default is 500.

    Raises:
        ValueError: If ``category`` is unknown, a length is not positive, or
            ``epoch_sec`` or ``shift`` rounds to zero samples at ``target_fs``.

    Examples:
        >>> from pyhealth.tasks import CardiologyDetection
        >>> task = CardiologyDetection("AR")
        >>> task.input_schema, task.output_schema
        ({'signal': 'tensor'}, {'label': 'binary'})
    """

    task_name: str = "CardiologyDetection"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        category: str,
        epoch_sec: float = 10,
        shift: float = 5,
        target_fs: float = 500,
    ) -> None:
        super().__init__()
        if category not in CARDIOLOGY_CATEGORIES:
            raise ValueError(
                f"category must be one of {list(CARDIOLOGY_CATEGORIES)}, "
                f"got {category!r}."
            )
        if min(epoch_sec, shift, target_fs) <= 0:
            raise ValueError("epoch_sec, shift and target_fs must be positive.")
        # Lengths are rounded to whole samples; a window or step that rounds
        # to zero would divide by zero in ``__call__``.
        if round(epoch_sec * target_fs) < 1 or round(shift * target_fs) < 1:
            raise ValueError(
                "epoch_sec and shift must each span at least one sample at "
                f"target_fs (got epoch_sec={epoch_sec}, shift={shift}, "
                f"target_fs={target_fs})."
            )
        self.category = category
        self.epoch_sec = float(epoch_sec)
        self.shift = float(shift)
        self.target_fs = float(target_fs)

    def _load(self, signal_file: str, fs: float) -> np.ndarray:
        """Reads a recording and resamples it to ``target_fs``."""
        signal = loadmat(signal_file)["val"].astype(np.float32)
        if fs != self.target_fs:
            ratio = Fraction(self.target_fs / fs).limit_denominator(1000)
            signal = resample_poly(
                signal, ratio.numerator, ratio.denominator, axis=1
            ).astype(np.float32)
        return signal

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the window samples of one recording.

        Args:
            patient: A PyHealth patient whose events are rows of the
                ``cardiology-<source>-pyhealth.csv`` tables.

        Returns:
            One sample per full window of each of the patient's recordings.
        """
        codes = set(CARDIOLOGY_CATEGORIES[self.category])
        width = round(self.epoch_sec * self.target_fs)
        step = round(self.shift * self.target_fs)
        samples = []
        for event in patient.get_events():
            dx = {code for code in str(event.dx or "").split(",") if code}
            label = int(bool(dx & codes))
            signal = self._load(event.signal_file, float(event.fs))
            n_windows = (signal.shape[1] - width) // step + 1
            for index in range(max(n_windows, 0)):
                start = index * step
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": event.record_id,
                        "source": event.source,
                        "window": index,
                        "signal": torch.from_numpy(
                            np.ascontiguousarray(signal[:, start : start + width])
                        ),
                        "label": label,
                        "age": _text(event.age),
                        "sex": _text(event.sex),
                    }
                )
        return samples
