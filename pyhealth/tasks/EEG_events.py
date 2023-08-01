import os
import pickle
import pkg_resources
import mne
import pandas as pd
import numpy as np

def EEG_events_fn(record):
    """Processes a single patient for the EEG events task on TUEV.

    This task aims at annotating of EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

    Args:
        record: a singleton list of one subject from the TUEVDataset.
            The (single) record is a dictionary with the following keys:
                load_from_path, patient_id, visit_id, signal_file, label_file, save_to_path

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id, record_id, label, offending_channel,
            and epoch_path (the path to the saved epoch {"signal": signal, "label": label} as key.

    Note that we define the task as a multiclass classification task.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> EEGevents = TUEVDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/", download=True,
        ...     )
        >>> from pyhealth.tasks import EEG_events_fn
        >>> EEG_events_ds = EEGevents.set_task(EEG_events_fn)
        >>> EEG_events_ds.samples[0]
        {
            'patient_id': '0_00002265',
            'visit_id': '00000001',
            'record_id': 0,
            'epoch_path': '/Users/liyanjing/.cache/pyhealth/datasets/d8f3cb92cc444d481444d3414fb5240c/0_00002265_00000001_0.pkl',
            'label': 6,
            'offending_channel': array([4.])
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

        
        # load data
        try:
            [signals, times, event, Rawdata] = readEDF(
                os.path.join(root, signal)
            )  # event is the .rec file in the form of an array
            signals = convert_signals(signals, Rawdata)
        except (ValueError, KeyError):
            print("something funky happened in " + os.path.join(root, signal))
            continue
        signals, offending_channels, labels = BuildEvents(signals, times, event)

        for idx, (signal, offending_channel, label) in enumerate(
            zip(signals, offending_channels, labels)
        ):
            dump_path = os.path.join(
                save_path, pid + "_" + visit_id + "_" + str(idx) + ".pkl"
            )

            pickle.dump(
                    {"signal": signal, "label": int(label[0])},
                    open(dump_path, "wb"),
                )
            
            samples.append(
                {
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "record_id": idx,
                    "epoch_path": dump_path,
                    "label": int(label[0]),
                    "offending_channel": offending_channel,
                }
            )

    return samples

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 250.0
    [numChan, numPoints] = signals.shape

    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]]
            - signals[signal_names["EEG F7-REF"]],  # 0
            (
                signals[signal_names["EEG F7-REF"]]
                - signals[signal_names["EEG T3-REF"]]
            ),  # 1
            (
                signals[signal_names["EEG T3-REF"]]
                - signals[signal_names["EEG T5-REF"]]
            ),  # 2
            (
                signals[signal_names["EEG T5-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 3
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F8-REF"]]
            ),  # 4
            (
                signals[signal_names["EEG F8-REF"]]
                - signals[signal_names["EEG T4-REF"]]
            ),  # 5
            (
                signals[signal_names["EEG T4-REF"]]
                - signals[signal_names["EEG T6-REF"]]
            ),  # 6
            (
                signals[signal_names["EEG T6-REF"]]
                - signals[signal_names["EEG O2-REF"]]
            ),  # 7
            (
                signals[signal_names["EEG FP1-REF"]]
                - signals[signal_names["EEG F3-REF"]]
            ),  # 14
            (
                signals[signal_names["EEG F3-REF"]]
                - signals[signal_names["EEG C3-REF"]]
            ),  # 15
            (
                signals[signal_names["EEG C3-REF"]]
                - signals[signal_names["EEG P3-REF"]]
            ),  # 16
            (
                signals[signal_names["EEG P3-REF"]]
                - signals[signal_names["EEG O1-REF"]]
            ),  # 17
            (
                signals[signal_names["EEG FP2-REF"]]
                - signals[signal_names["EEG F4-REF"]]
            ),  # 18
            (
                signals[signal_names["EEG F4-REF"]]
                - signals[signal_names["EEG C4-REF"]]
            ),  # 19
            (
                signals[signal_names["EEG C4-REF"]]
                - signals[signal_names["EEG P4-REF"]]
            ),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    signals, times = Rawdata[:]
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]



if __name__ == "__main__":
    from pyhealth.datasets import TUEVDataset
    
    dataset = TUEVDataset(
        root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
        dev=True,
        refresh_cache=True,
    )
    EEG_events_ds = dataset.set_task(EEG_events_fn)
    print(EEG_events_ds.samples[0])
    print(EEG_events_ds.input_info)
    
    
    
    
