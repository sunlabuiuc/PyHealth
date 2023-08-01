import os
import pickle
import pkg_resources
import mne
import pandas as pd
import numpy as np


def EEG_isAbnormal_fn(record):
    """Processes a single patient for the abnormal EEG detection task on TUAB.

    Abnormal EEG detection aims at determining whether a EEG is abnormal.

    Args:
        record: a singleton list of one subject from the TUABDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, patient_id, visit_id, signal_file, label_file, save_to_path

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, record_id,
            and epoch_path (the path to the saved epoch {"signal": signal, "label": label} as key.

    Note that we define the task as a binary classification task.

    Examples:
        >>> from pyhealth.datasets import TUABDataset
        >>> isabnormal = TUABDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/", download=True,
        ...     )
        >>> from pyhealth.tasks import EEG_isabnormal_fn
        >>> EEG_abnormal_ds = isabnormal.set_task(EEG_isAbnormal_fn)
        >>> EEG_abnormal_ds.samples[0]
        {
            'patient_id': 'aaaaamye',
            'visit_id': 's001',
            'record_id': '1',
            'epoch_path': '/home/zhenlin4/.cache/pyhealth/datasets/832afe6e6e8a5c9ea5505b47e7af8125/10-1/1/0.pkl',
            'label': 1
        }
    """
    
    samples = []
    for visit in record:
        root, pid, visit_id, signal, label, save_path = (
            visit["load_from_path"],
            visit["patient_id"],
            visit["visit_id"],
            visit["signal_file"],
            visit["label_file"],
            visit["save_to_path"],
        )

        raw = mne.io.read_raw_edf(os.path.join(root, signal), preload=True)
        raw.resample(200)
        ch_name = raw.ch_names
        raw_data = raw.get_data()
        channeled_data = raw_data.copy()[:16]
        try:
            channeled_data[0] = (
                raw_data[ch_name.index("EEG FP1-REF")]
                - raw_data[ch_name.index("EEG F7-REF")]
            )
            channeled_data[1] = (
                raw_data[ch_name.index("EEG F7-REF")]
                - raw_data[ch_name.index("EEG T3-REF")]
            )
            channeled_data[2] = (
                raw_data[ch_name.index("EEG T3-REF")]
                - raw_data[ch_name.index("EEG T5-REF")]
            )
            channeled_data[3] = (
                raw_data[ch_name.index("EEG T5-REF")]
                - raw_data[ch_name.index("EEG O1-REF")]
            )
            channeled_data[4] = (
                raw_data[ch_name.index("EEG FP2-REF")]
                - raw_data[ch_name.index("EEG F8-REF")]
            )
            channeled_data[5] = (
                raw_data[ch_name.index("EEG F8-REF")]
                - raw_data[ch_name.index("EEG T4-REF")]
            )
            channeled_data[6] = (
                raw_data[ch_name.index("EEG T4-REF")]
                - raw_data[ch_name.index("EEG T6-REF")]
            )
            channeled_data[7] = (
                raw_data[ch_name.index("EEG T6-REF")]
                - raw_data[ch_name.index("EEG O2-REF")]
            )
            channeled_data[8] = (
                raw_data[ch_name.index("EEG FP1-REF")]
                - raw_data[ch_name.index("EEG F3-REF")]
            )
            channeled_data[9] = (
                raw_data[ch_name.index("EEG F3-REF")]
                - raw_data[ch_name.index("EEG C3-REF")]
            )
            channeled_data[10] = (
                raw_data[ch_name.index("EEG C3-REF")]
                - raw_data[ch_name.index("EEG P3-REF")]
            )
            channeled_data[11] = (
                raw_data[ch_name.index("EEG P3-REF")]
                - raw_data[ch_name.index("EEG O1-REF")]
            )
            channeled_data[12] = (
                raw_data[ch_name.index("EEG FP2-REF")]
                - raw_data[ch_name.index("EEG F4-REF")]
            )
            channeled_data[13] = (
                raw_data[ch_name.index("EEG F4-REF")]
                - raw_data[ch_name.index("EEG C4-REF")]
            )
            channeled_data[14] = (
                raw_data[ch_name.index("EEG C4-REF")]
                - raw_data[ch_name.index("EEG P4-REF")]
            )
            channeled_data[15] = (
                raw_data[ch_name.index("EEG P4-REF")]
                - raw_data[ch_name.index("EEG O2-REF")]
            )
        except:
            with open("tuab-process-error-files.txt", "a") as f:
                f.write(os.path.join(root, signal) + "\n")
            continue

        # get the label
        data_field = pid.split("_")[0]
        if data_field == "0" or data_field == "2":
            label = 1
        else:
            label = 0

        # load data
        for i in range(channeled_data.shape[1] // 2000):
            dump_path = os.path.join(
                save_path, pid + "_" + visit_id + "_" + str(i) + ".pkl"
            )
            pickle.dump(
                {"signal": channeled_data[:, i * 2000 : (i + 1) * 2000], "label": label},
                open(dump_path, "wb"),
            )

            samples.append(
                    {   
                        "patient_id": pid,
                        "visit_id": visit_id,
                        "record_id": i,
                        "epoch_path": dump_path,
                        "label": label,
                    }
            )

        return samples



if __name__ == "__main__":
    from pyhealth.datasets import TUABDataset
    
    dataset = TUABDataset(
        root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/",
        dev=True,
        refresh_cache=True,
    )
    EEG_abnormal_ds = dataset.set_task(EEG_isAbnormal_fn)
    print(EEG_abnormal_ds.samples[0])
    print(EEG_abnormal_ds.input_info)
    
    
    
    
