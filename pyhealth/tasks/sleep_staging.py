import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import mne
import numpy as np
import pandas as pd
import polars as pl

from .base_task import BaseTask


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
        from importlib.metadata import version
        version(dependencies)
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


class SleepStagingDREAMT(BaseTask):
    """Three-class sleep staging task for DREAMT-style wearable sequences.

    This task converts one DREAMT subject recording into fixed-length windows of
    wearable features and maps detailed sleep stages to three classes:

    - ``0``: Wake
    - ``1``: NREM
    - ``2``: REM

    Supported input formats:

    - Raw per-subject CSV/Parquet/Pickle files containing wearable columns and a
      ``Sleep_Stage`` column.
    - Processed ``.npz``/``.npy`` dictionaries with keys such as
      ``features``, ``labels``, and optional ``feature_names``.

    The default feature set mirrors the smartwatch-oriented signals available in
    DREAMT and includes ``IBI`` plus common wearable context channels.
    """

    task_name: str = "SleepStagingDREAMT"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    DEFAULT_FEATURE_COLUMNS = (
        "IBI",
        "HR",
        "BVP",
        "EDA",
        "TEMP",
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
    )
    _IGNORE_LABEL = -1
    _LABEL_MAP = {
        "W": 0,
        "WAKE": 0,
        "WAKEFUL": 0,
        "0": 0,
        "N1": 1,
        "N2": 1,
        "N3": 1,
        "N4": 1,
        "NREM": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 1,
        "R": 2,
        "REM": 2,
        "5": 2,
    }
    _INVALID_LABELS = {"", "P", "PREPARATION", "UNKNOWN", "?", "NAN", "NONE"}

    def __init__(
        self,
        feature_columns: Optional[Sequence[str]] = None,
        label_column: str = "Sleep_Stage",
        source_preference: str = "wearable",
        window_seconds: float = 30.0,
        stride_seconds: Optional[float] = None,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        default_sampling_rate_hz: Optional[float] = None,
        min_labeled_fraction: float = 0.5,
        pad_short_windows: bool = False,
        include_partial_last_window: bool = False,
    ) -> None:
        if source_preference not in {"auto", "wearable", "psg"}:
            raise ValueError(
                "source_preference must be one of 'auto', 'wearable', or 'psg'."
            )
        if window_seconds <= 0 and window_size is None:
            raise ValueError(
                "window_seconds must be positive when window_size is None."
            )
        if min_labeled_fraction <= 0 or min_labeled_fraction > 1:
            raise ValueError("min_labeled_fraction must be in (0, 1].")

        self.feature_columns = tuple(feature_columns or self.DEFAULT_FEATURE_COLUMNS)
        self.label_column = label_column
        self.source_preference = source_preference
        self.window_seconds = float(window_seconds)
        self.stride_seconds = (
            float(stride_seconds) if stride_seconds is not None else None
        )
        self.window_size = window_size
        self.stride = stride
        self.default_sampling_rate_hz = default_sampling_rate_hz
        self.min_labeled_fraction = min_labeled_fraction
        self.pad_short_windows = pad_short_windows
        self.include_partial_last_window = include_partial_last_window

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(
            (pl.col("event_type") == "dreamt_sleep")
            & (
                pl.col("dreamt_sleep/signal_file").is_not_null()
                | pl.col("dreamt_sleep/file_64hz").is_not_null()
                | pl.col("dreamt_sleep/file_100hz").is_not_null()
            )
        )

    @classmethod
    def _normalize_stage_label(cls, value: Any) -> int:
        if pd.isna(value):
            return cls._IGNORE_LABEL
        normalized = str(value).strip().upper()
        if normalized in cls._INVALID_LABELS:
            return cls._IGNORE_LABEL
        return cls._LABEL_MAP.get(normalized, cls._IGNORE_LABEL)

    @staticmethod
    def _load_processed_payload(path: Path) -> Dict[str, Any]:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=True) as data:
                return {key: data[key] for key in data.files}

        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unsupported DREAMT numpy payload in {path}. "
                "Expected a dict-like object with features and labels."
            )
        return payload

    def _load_signal_source(self, event) -> tuple[Path, Optional[float]]:
        event_sampling_rate = getattr(event, "sampling_rate_hz", None)
        if event_sampling_rate is not None:
            try:
                event_sampling_rate = float(event_sampling_rate)
            except (TypeError, ValueError):
                event_sampling_rate = None

        candidates: list[tuple[Optional[str], Optional[float]]] = []
        if self.source_preference == "wearable":
            candidates.extend(
                [
                    (getattr(event, "file_64hz", None), 64.0),
                    (getattr(event, "signal_file", None), event_sampling_rate),
                    (getattr(event, "file_100hz", None), 100.0),
                ]
            )
        elif self.source_preference == "psg":
            candidates.extend(
                [
                    (getattr(event, "file_100hz", None), 100.0),
                    (getattr(event, "signal_file", None), event_sampling_rate),
                    (getattr(event, "file_64hz", None), 64.0),
                ]
            )
        else:
            candidates.extend(
                [
                    (getattr(event, "signal_file", None), event_sampling_rate),
                    (getattr(event, "file_64hz", None), 64.0),
                    (getattr(event, "file_100hz", None), 100.0),
                ]
            )

        for file_path, sample_rate in candidates:
            if file_path is None or (
                isinstance(file_path, float) and np.isnan(file_path)
            ):
                continue
            path = Path(str(file_path)).expanduser().resolve()
            if path.exists():
                return path, sample_rate

        raise FileNotFoundError(
            "No DREAMT signal file was found for the requested patient event."
        )

    def _dataframe_from_payload(
        self,
        payload: Dict[str, Any],
        path: Path,
    ) -> pd.DataFrame:
        if "frame" in payload:
            frame = payload["frame"]
            if isinstance(frame, pd.DataFrame):
                return frame.copy()
            return pd.DataFrame(frame)

        if "features" not in payload or "labels" not in payload:
            raise ValueError(
                f"Processed DREAMT file {path} must contain 'features' and 'labels'."
            )

        features = np.asarray(payload["features"], dtype=np.float32)
        labels = np.asarray(payload["labels"])
        if features.ndim != 2:
            raise ValueError(
                f"Processed DREAMT file {path} must provide features with shape [T, F]."
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"Processed DREAMT file {path} must provide labels with shape [T]."
            )

        feature_names = payload.get("feature_names", self.feature_columns)
        feature_names = [str(name) for name in feature_names]
        if len(feature_names) != features.shape[1]:
            raise ValueError(
                f"feature_names length does not match feature width in {path}."
            )

        frame = pd.DataFrame(features, columns=feature_names)
        frame[self.label_column] = labels
        if "timestamps" in payload:
            frame["TIMESTAMP"] = np.asarray(payload["timestamps"])
        return frame

    def _load_frame(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".pkl", ".pickle"}:
            payload = pd.read_pickle(path)
            if isinstance(payload, pd.DataFrame):
                return payload
            if isinstance(payload, dict):
                return self._dataframe_from_payload(payload, path)
            raise ValueError(f"Unsupported pickle payload in {path}.")
        if suffix in {".npz", ".npy"}:
            payload = self._load_processed_payload(path)
            return self._dataframe_from_payload(payload, path)
        raise ValueError(f"Unsupported DREAMT file format: {path.suffix}")

    @staticmethod
    def _infer_sampling_rate_hz(
        frame: pd.DataFrame,
        fallback: Optional[float],
    ) -> float:
        if "TIMESTAMP" in frame.columns:
            timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy()
            diffs = np.diff(timestamps)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size > 0:
                median_step = float(np.median(diffs))
                if median_step > 0:
                    return 1.0 / median_step
        if fallback is not None and fallback > 0:
            return float(fallback)
        raise ValueError(
            "Unable to infer DREAMT sampling rate. Provide TIMESTAMP values or "
            "set default_sampling_rate_hz."
        )

    def _resolve_window_params(self, sample_rate_hz: float) -> tuple[int, int]:
        window_size = self.window_size
        if window_size is None:
            window_size = max(1, int(round(self.window_seconds * sample_rate_hz)))

        stride = self.stride
        if stride is None:
            if self.stride_seconds is not None:
                stride = max(1, int(round(self.stride_seconds * sample_rate_hz)))
            else:
                stride = window_size
        return window_size, stride

    def _extract_feature_frame(self, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        available_columns = {str(column): column for column in frame.columns}
        feature_data = {}
        for column in self.feature_columns:
            source_column = available_columns.get(column)
            if source_column is None:
                feature_data[column] = np.zeros(len(frame), dtype=np.float32)
                continue
            values = pd.to_numeric(frame[source_column], errors="coerce")
            feature_data[column] = values.to_numpy(dtype=np.float32)

        if not feature_data:
            raise ValueError(f"No wearable feature columns were found in {path}.")

        feature_frame = pd.DataFrame(feature_data)
        feature_frame = feature_frame.ffill().bfill().fillna(0.0)
        return feature_frame

    def _extract_labels(self, frame: pd.DataFrame, path: Path) -> np.ndarray:
        if self.label_column not in frame.columns:
            raise ValueError(
                f"DREAMT file {path} is missing the label column '{self.label_column}'."
            )
        labels = frame[self.label_column].apply(self._normalize_stage_label).to_numpy()
        return labels.astype(np.int64, copy=False)

    @staticmethod
    def _majority_label(labels: np.ndarray) -> int:
        valid = labels[labels >= 0]
        if valid.size == 0:
            return SleepStagingDREAMT._IGNORE_LABEL
        return int(np.bincount(valid).argmax())

    def __call__(self, patient) -> list[dict[str, Any]]:
        events = patient.get_events("dreamt_sleep")
        if not events:
            return []

        samples = []
        for event in events:
            signal_path, sampling_rate_hz = self._load_signal_source(event)
            frame = self._load_frame(signal_path)
            labels = self._extract_labels(frame, signal_path)
            valid_mask = labels >= 0

            if valid_mask.sum() == 0:
                continue

            frame = frame.loc[valid_mask].reset_index(drop=True)
            labels = labels[valid_mask]
            feature_frame = self._extract_feature_frame(frame, signal_path)
            features = feature_frame.to_numpy(dtype=np.float32, copy=False)

            inferred_rate = self._infer_sampling_rate_hz(
                frame,
                sampling_rate_hz or self.default_sampling_rate_hz,
            )
            window_size, stride = self._resolve_window_params(inferred_rate)

            total_length = features.shape[0]
            if total_length < window_size and not self.pad_short_windows:
                continue

            starts = list(range(0, max(total_length - window_size + 1, 1), stride))
            if (
                self.include_partial_last_window
                and total_length > window_size
                and starts[-1] + window_size < total_length
            ):
                starts.append(starts[-1] + stride)

            for window_index, start in enumerate(starts):
                end = min(start + window_size, total_length)
                window_features = features[start:end]
                window_labels = labels[start:end]

                labeled_fraction = float((window_labels >= 0).mean())
                if labeled_fraction < self.min_labeled_fraction:
                    continue

                label = self._majority_label(window_labels)
                if label == self._IGNORE_LABEL:
                    continue

                if end - start < window_size:
                    if not self.pad_short_windows:
                        continue
                    pad_rows = window_size - (end - start)
                    window_features = np.pad(
                        window_features,
                        pad_width=((0, pad_rows), (0, 0)),
                        mode="constant",
                    )

                record_id = f"{patient.patient_id}-{signal_path.stem}-{window_index}"
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": record_id,
                        "signal": window_features.astype(np.float32, copy=False),
                        "label": int(label),
                        "signal_source": getattr(event, "signal_source", None),
                        "signal_file": str(signal_path),
                    }
                )

        return samples


class SleepStagingDREAMTSeq(SleepStagingDREAMT):
    """Sequence-style DREAMT sleep staging task closer to WatchSleepNet.

    This task converts a DREAMT recording into a sequence of epoch-level
    feature vectors and labels. Each sample contains:

    - ``signal``: ``[sequence_length, input_dim]``
    - ``mask``: ``[sequence_length]`` with 1 for valid epochs and 0 for padding
    - ``label``: ``[sequence_length]`` with padded labels set to
      ``ignore_index``

    By default, this class uses ``IBI`` only to better align with the paper's
    shared-modality representation.
    """

    task_name: str = "SleepStagingDREAMTSeq"
    input_schema: Dict[str, str] = {
        "signal": "tensor",
        "mask": "tensor",
    }
    output_schema: Dict[str, str] = {"label": "tensor"}

    def __init__(
        self,
        feature_columns: Optional[Sequence[str]] = ("IBI",),
        label_column: str = "Sleep_Stage",
        source_preference: str = "wearable",
        epoch_seconds: float = 30.0,
        sequence_length: int = 1100,
        stride_epochs: Optional[int] = None,
        default_sampling_rate_hz: Optional[float] = None,
        pad_value: float = 0.0,
        truncate: bool = True,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            source_preference=source_preference,
            default_sampling_rate_hz=default_sampling_rate_hz,
        )
        if epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be positive.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        self.epoch_seconds = float(epoch_seconds)
        self.sequence_length = int(sequence_length)
        self.stride_epochs = stride_epochs
        self.pad_value = float(pad_value)
        self.truncate = truncate
        self.ignore_index = int(ignore_index)

    def _epochize_features_and_labels(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_rate_hz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        epoch_size = max(1, int(round(self.epoch_seconds * sample_rate_hz)))
        num_epochs = features.shape[0] // epoch_size
        if num_epochs == 0:
            return (
                np.zeros((0, features.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        epoch_features: list[np.ndarray] = []
        epoch_labels: list[int] = []
        for epoch_index in range(num_epochs):
            start = epoch_index * epoch_size
            end = start + epoch_size
            epoch_feature_values = features[start:end]
            epoch_label_values = labels[start:end]
            epoch_label = self._majority_label(epoch_label_values)
            if epoch_label == self._IGNORE_LABEL:
                continue
            epoch_features.append(
                epoch_feature_values.mean(axis=0).astype(np.float32, copy=False)
            )
            epoch_labels.append(epoch_label)

        if not epoch_features:
            return (
                np.zeros((0, features.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return (
            np.stack(epoch_features).astype(np.float32, copy=False),
            np.asarray(epoch_labels, dtype=np.int64),
        )

    def _build_sequence_sample(
        self,
        patient_id: str,
        signal_path: Path,
        epoch_features: np.ndarray,
        epoch_labels: np.ndarray,
        chunk_index: int,
        start_epoch: int,
    ) -> dict[str, Any]:
        valid_length = min(epoch_features.shape[0], self.sequence_length)
        feature_dim = epoch_features.shape[1]

        signal = np.full(
            (self.sequence_length, feature_dim),
            fill_value=self.pad_value,
            dtype=np.float32,
        )
        mask = np.zeros((self.sequence_length,), dtype=np.float32)
        labels = np.full(
            (self.sequence_length,),
            fill_value=self.ignore_index,
            dtype=np.int64,
        )

        signal[:valid_length] = epoch_features[:valid_length]
        mask[:valid_length] = 1.0
        labels[:valid_length] = epoch_labels[:valid_length]

        return {
            "patient_id": patient_id,
            "record_id": f"{patient_id}-{signal_path.stem}-seq-{chunk_index}",
            "signal": signal,
            "mask": mask,
            "label": labels,
            "signal_source": None,
            "signal_file": str(signal_path),
            "start_epoch": int(start_epoch),
        }

    def __call__(self, patient) -> list[dict[str, Any]]:
        events = patient.get_events("dreamt_sleep")
        if not events:
            return []

        samples = []
        for event in events:
            signal_path, sampling_rate_hz = self._load_signal_source(event)
            frame = self._load_frame(signal_path)
            labels = self._extract_labels(frame, signal_path)
            feature_frame = self._extract_feature_frame(frame, signal_path)
            features = feature_frame.to_numpy(dtype=np.float32, copy=False)

            inferred_rate = self._infer_sampling_rate_hz(
                frame,
                sampling_rate_hz or self.default_sampling_rate_hz,
            )
            epoch_features, epoch_labels = self._epochize_features_and_labels(
                features,
                labels,
                inferred_rate,
            )
            if epoch_features.shape[0] == 0:
                continue

            stride_epochs = self.stride_epochs or self.sequence_length
            if epoch_features.shape[0] <= self.sequence_length:
                samples.append(
                    self._build_sequence_sample(
                        patient.patient_id,
                        signal_path,
                        epoch_features,
                        epoch_labels,
                        chunk_index=0,
                        start_epoch=0,
                    )
                )
                continue

            max_start = epoch_features.shape[0] - self.sequence_length
            starts = list(range(0, max_start + 1, stride_epochs))
            if (
                not self.truncate
                and starts[-1] != max_start
            ):
                starts.append(max_start)

            for chunk_index, start_epoch in enumerate(starts):
                end_epoch = start_epoch + self.sequence_length
                samples.append(
                    self._build_sequence_sample(
                        patient.patient_id,
                        signal_path,
                        epoch_features[start_epoch:end_epoch],
                        epoch_labels[start_epoch:end_epoch],
                        chunk_index=chunk_index,
                        start_epoch=start_epoch,
                    )
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
    
    
    
    
