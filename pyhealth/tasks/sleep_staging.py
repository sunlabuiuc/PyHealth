import os
import pickle
import pkg_resources
import mne
import pandas as pd
import numpy as np


def sleep_staging_isruc_fn(record, epoch_seconds=10, label_id=1):
    """Processes a single patient for the sleep staging task on ISRUC.

    Sleep staging aims at predicting the sleep stages (Awake, N1, N2, N3, REM) based on
    the multichannel EEG signals. The task is defined as a multi-class classification.

    Args:
        record: a singleton list of one subject from the ISRUCDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, signal_file, label1_file, label2_file, save_to_path, subject_id
        epoch_seconds: how long will each epoch be (in seconds).
            It has to be a factor of 30 because the original data was labeled every 30 seconds.
        label_id: which set of labels to use. ISURC is labeled by *two* experts.
            By default we use the first set of labels (label_id=1).

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Y": label} as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> isruc = ISRUCDataset(
        ...         root="/srv/local/data/data/ISRUC-I", download=True,
        ...     )
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> sleepstage_ds = isruc.set_task(sleep_staging_isruc_fn)
        >>> sleepstage_ds.samples[0]
        {
            'record_id': '1-0',
            'patient_id': '1',
            'epoch_path': '/home/zhenlin4/.cache/pyhealth/datasets/832afe6e6e8a5c9ea5505b47e7af8125/10-1/1/0.pkl',
            'label': 'W'
        }
    """
    SAMPLE_RATE = 200
    assert 30 % epoch_seconds == 0, "ISRUC is annotated every 30 seconds."
    _channels = [
        "F3",
        "F4",
        "C3",
        "C4",
        "O1",
        "O2",
    ]  # https://arxiv.org/pdf/1910.06100.pdf

    def _find_channels(potential_channels):
        keep = {}
        for c in potential_channels:
            # https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
            new_c = (
                c.replace("-M2", "")
                .replace("-A2", "")
                .replace("-M1", "")
                .replace("-A1", "")
            )
            if new_c in _channels:
                assert new_c not in keep, f"Unrecognized channels: {potential_channels}"
                keep[new_c] = c
        assert len(keep) == len(
            _channels
        ), f"Unrecognized channels: {potential_channels}"
        return {v: k for k, v in keep.items()}

    record = record[0]
    save_path = os.path.join(
        record["save_to_path"], f"{epoch_seconds}-{label_id}", record["subject_id"]
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data = mne.io.read_raw_edf(
        os.path.join(record["load_from_path"], record["signal_file"])
    ).to_data_frame()
    data = (
        data.rename(columns=_find_channels(data.columns))
        .reindex(columns=_channels)
        .values
    )
    ann = pd.read_csv(
        os.path.join(record["load_from_path"], record[f"label{label_id}_file"]),
        header=None,
    )[0]
    ann = ann.map(["W", "N1", "N2", "N3", "Unknown", "R"].__getitem__)
    assert "Unknown" not in ann.values, "bad annotations"
    samples = []
    sample_length = SAMPLE_RATE * epoch_seconds
    for i, epoch_label in enumerate(np.repeat(ann.values, 30 // epoch_seconds)):
        epoch_signal = data[i * sample_length : (i + 1) * sample_length].T
        save_file_path = os.path.join(save_path, f"{i}.pkl")
        pickle.dump(
            {
                "signal": epoch_signal,
                "label": epoch_label,
            },
            open(save_file_path, "wb"),
        )
        samples.append(
            {
                "record_id": f"{record['subject_id']}-{i}",
                "patient_id": record["subject_id"],
                "epoch_path": save_file_path,
                "label": epoch_label,  # use for counting the label tokens
            }
        )
    return samples


def sleep_staging_sleepedf_fn(record, epoch_seconds=30):
    """Processes a single patient for the sleep staging task on Sleep EDF.

    Sleep staging aims at predicting the sleep stages (Awake, REM, N1, N2, N3, N4) based on
    the multichannel EEG signals. The task is defined as a multi-class classification.

    Args:
        patient: a list of (load_from_path, signal_file, label_file, save_to_path) tuples, where PSG is the signal files and the labels are
        in label file
        epoch_seconds: how long will each epoch be (in seconds)

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Y": label} as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> sleepedf = SleepEDFDataset(
        ...         root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        ...     )
        >>> from pyhealth.tasks import sleep_staging_sleepedf_fn
        >>> sleepstage_ds = sleepedf.set_task(sleep_staging_sleepedf_fn)
        >>> sleepstage_ds.samples[0]
        {
            'record_id': 'SC4001-0',
            'patient_id': 'SC4001',
            'epoch_path': '/home/chaoqiy2/.cache/pyhealth/datasets/70d6dbb28bd81bab27ae2f271b2cbb0f/SC4001-0.pkl',
            'label': 'W'
        }
    """

    SAMPLE_RATE = 100

    root, psg_file, hypnogram_file, save_path = (
        record[0]["load_from_path"],
        record[0]["signal_file"],
        record[0]["label_file"],
        record[0]["save_to_path"],
    )
    # get patient id
    pid = psg_file[:6]

    # load signal "X" part
    data = mne.io.read_raw_edf(os.path.join(root, psg_file))

    X = data.get_data()
    # load label "Y" part
    ann = mne.read_annotations(os.path.join(root, hypnogram_file))

    labels = []
    for dur, des in zip(ann.duration, ann.description):
        """
        all possible des:
            - 'Sleep stage W'
            - 'Sleep stage 1'
            - 'Sleep stage 2'
            - 'Sleep stage 3'
            - 'Sleep stage 4'
            - 'Sleep stage R'
            - 'Sleep stage ?'
            - 'Movement time'
        """
        for _ in range(int(dur) // 30):
            labels.append(des)

    samples = []
    sample_length = SAMPLE_RATE * epoch_seconds
    # slice the EEG signals into non-overlapping windows
    # window size = sampling rate * second time = 100 * epoch_seconds
    for slice_index in range(min(X.shape[1] // sample_length, len(labels))):
        # ingore the no label epoch
        if labels[slice_index] not in [
            "Sleep stage W",
            "Sleep stage 1",
            "Sleep stage 2",
            "Sleep stage 3",
            "Sleep stage 4",
            "Sleep stage R",
        ]:
            continue

        epoch_signal = X[
            :, slice_index * sample_length : (slice_index + 1) * sample_length
        ]
        epoch_label = labels[slice_index][-1]  # "W", "1", "2", "3", "R"
        save_file_path = os.path.join(save_path, f"{pid}-{slice_index}.pkl")

        pickle.dump(
            {
                "signal": epoch_signal,
                "label": epoch_label,
            },
            open(save_file_path, "wb"),
        )

        samples.append(
            {
                "record_id": f"{pid}-{slice_index}",
                "patient_id": pid,
                "epoch_path": save_file_path,
                "label": epoch_label,  # use for counting the label tokens
            }
        )
    return samples


def sleep_staging_shhs_fn(record, epoch_seconds=30):
    """Processes a single recording for the sleep staging task on SHHS.

    Sleep staging aims at predicting the sleep stages (Awake, REM, N1, N2, N3) based on
    the multichannel EEG signals. The task is defined as a multi-class classification.

    Args:
        patient: a list of (load_from_path, signal file, label file, save_to_path) tuples, where the signal is in edf file and
        the labels are in the label file
        epoch_seconds: how long will each epoch be (in seconds), 30 seconds as default given by the label file

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, record_id,
            and epoch_path (the path to the saved epoch {"X": signal, "Y": label} as key.

    Note that we define the task as a multi-class classification task.

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> shhs = SHHSDataset(
        ...         root="/srv/local/data/SHHS/polysomnography",
        ...         dev=True,
        ...     )
        >>> from pyhealth.tasks import sleep_staging_shhs_fn
        >>> shhs_ds = shhs.set_task(sleep_staging_shhs_fn)
        >>> shhs_ds.samples[0]
        {
            'record_id': 'shhs1-200001-0', 
            'patient_id': 'shhs1-200001', 
            'epoch_path': '/home/chaoqiy2/.cache/pyhealth/datasets/76c1ce8195a2e1a654e061cb5df4671a/shhs1-200001-0.pkl', 
            'label': '0'
        }
    """
    
    # test whether the ogb and torch_scatter packages are ready
    dependencies = ["elementpath"]
    try:
        pkg_resources.require(dependencies)
        import xml.etree.ElementTree as ET
    except Exception as e:
        print(e)
        print ('-----------')
        print(
            "Please follow the error message and install the ['elementpath'] packages first."
        )
    
    SAMPLE_RATE = 125

    root, signal_file, label_file, save_path = (
        record[0]["load_from_path"],
        record[0]["signal_file"],
        record[0]["label_file"],
        record[0]["save_to_path"],
    )
    # get file prefix, e.g., shhs1-200001
    pid = signal_file.split("/")[-1].split(".")[0]

    # load signal "X" part
    data = mne.io.read_raw_edf(os.path.join(root, signal_file))

    X = data.get_data()
    
    # some EEG signals have missing channels, we treat them separately
    if X.shape[0] == 16:
        X = X[[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15], :]
    elif X.shape[0] == 15:
        X = X[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14], :]
    X = X[[2,7], :]
            
    # load label "Y" part
    with open(os.path.join(root, label_file), "r") as f:
        text = f.read()
        root = ET.fromstring(text)
        Y = [i.text for i in root.find('SleepStages').findall('SleepStage')]

    samples = []
    sample_length = SAMPLE_RATE * epoch_seconds
    
    # slice the EEG signals into non-overlapping windows
    # window size = sampling rate * second time = 125 * epoch_seconds
    for slice_index in range(X.shape[1] // sample_length):

        epoch_signal = X[
            :, slice_index * sample_length : (slice_index + 1) * sample_length
        ]
        epoch_label = Y[slice_index]
        save_file_path = os.path.join(save_path, f"{pid}-{slice_index}.pkl")

        pickle.dump(
            {
                "signal": epoch_signal,
                "label": epoch_label,
            },
            open(save_file_path, "wb"),
        )

        samples.append(
            {
                "record_id": f"{pid}-{slice_index}",
                "patient_id": pid,
                "epoch_path": save_file_path,
                "label": epoch_label,  # use for counting the label tokens
            }
        )
    return samples


if __name__ == "__main__":
    from pyhealth.datasets import SleepEDFDataset, SHHSDataset, ISRUCDataset

    """ test sleep edf"""
    # dataset = SleepEDFDataset(
    #     root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-telemetry",
    #     dev=True,
    #     refresh_cache=True,
    # )

    # sleep_staging_ds = dataset.set_task(sleep_staging_sleepedf_fn)
    # print(sleep_staging_ds.samples[0])
    # # print(sleep_staging_ds.patient_to_index)
    # # print(sleep_staging_ds.record_to_index)
    # print(sleep_staging_ds.input_info)

    # """ test ISRUC"""
    # dataset = ISRUCDataset(
    #     root="/srv/local/data/trash/",
    #     dev=True,
    #     refresh_cache=True,
    #     download=True,
    # )

    # sleep_staging_ds = dataset.set_task(sleep_staging_isruc_fn)
    # print(sleep_staging_ds.samples[0])
    # # print(sleep_staging_ds.patient_to_index)
    # # print(sleep_staging_ds.record_to_index)
    # print(sleep_staging_ds.input_info)
    
    dataset = SHHSDataset(
        root="/srv/local/data/SHHS/polysomnography",
        dev=True,
        refresh_cache=True,
    )
    sleep_staging_ds = dataset.set_task(sleep_staging_shhs_fn)
    print(sleep_staging_ds.samples[0])
    # print(sleep_staging_ds.patient_to_index)
    # print(sleep_staging_ds.record_to_index)
    print(sleep_staging_ds.input_info)
    
    
    
    
