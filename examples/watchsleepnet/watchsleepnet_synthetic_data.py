"""Generate synthetic DREAMT, SHHS, or MESA recordings.

The generated tree mirrors the folder and file layout expected by
:class:`~pyhealth.datasets.DREAMTDataset` and
:class:`~pyhealth.datasets.NSRRSleepDataset`, but contains deterministic mock
waveforms and hypnograms. It lets reviewers run the WatchSleepNet pipeline
without access to the protected source datasets.

DREAMT output::

    <output>/
        participant_info.csv
        dreamt-metadata.csv
        data_100Hz/S001_PSG_df.csv
        data_64Hz/S001_whole_df.csv

SHHS output::

    <output>/
        nsrr-shhs-pyhealth.csv
        edfs/shhs1/shhs1-200001.edf
        annotations-events-profusion/shhs1/shhs1-200001-profusion.xml

MESA output::

    <output>/
        nsrr-mesa-pyhealth.csv
        edfs/mesa-sleep-0001.edf
        annotations-events-profusion/mesa-sleep-0001-profusion.xml

Command:
    python examples/watchsleepnet/watchsleepnet_synthetic_data.py \
        --dataset shhs --num-samples 5 --output-dir synthetic/shhs
    python examples/watchsleepnet/watchsleepnet_synthetic_data.py \
        --dataset dreamt --num-samples 5 --output-dir synthetic/dreamt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

_RECORD_SECONDS = 30.0
_STAGE_CYCLES: dict[str, tuple[object, ...]] = {
    "shhs": (0, 0, 1, 1, 2, 2, 2, 3, 2, 2, 5, 5, 2, 1, 0),
    "mesa": (0, 0, 1, 1, 2, 2, 2, 3, 2, 2, 5, 5, 2, 1, 0),
    "dreamt": (
        "W", "W", "N1", "N2", "N2", "N3", "N2", "R", "R", "N2", "N1", "W",
    ),
}
_DREAMT_100HZ_COLUMNS = (
    "TIMESTAMP", "C4-M1", "F4-M1", "O2-M1", "Fp1-O2", "T3 - CZ", "CZ - T4",
    "CHIN", "E1", "E2", "ECG", "LAT", "RAT", "SNORE", "PTAF", "FLOW",
    "THORAX", "ABDOMEN", "SAO2", "BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP",
    "EDA", "HR", "IBI", "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea",
    "Hypopnea", "Multiple_Events",
)
_DREAMT_64HZ_COLUMNS = (
    "TIMESTAMP", "BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI",
    "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea", "Hypopnea",
    "Multiple_Events",
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser.

    Returns:
        A parser with ``--dataset`` and ``--num-samples`` required and the
        shape controls optional.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("shhs", "mesa", "dreamt"),
        help="dataset layout to simulate",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=int,
        help="number of synthetic recordings (one per patient)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of 30 s epochs per recording",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination; defaults to synthetic_data/<dataset>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for waveform and hypnogram generation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a non-empty output directory",
    )
    return parser


def _stage_sequence(dataset: str, epochs: int, seed: int) -> list[object]:
    """Returns a deterministic hypnogram containing every target class."""
    cycle = _STAGE_CYCLES[dataset]
    offset = seed % len(cycle)
    rotated = cycle[offset:] + cycle[:offset]
    return [rotated[i % len(rotated)] for i in range(epochs)]


def _heart_rate(
    n: int, fs: float, seed: int, bpm: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a smooth heart-rate trace and its sample times."""
    rng = np.random.default_rng(seed)
    times = np.arange(n, dtype=np.float64) / fs
    rate = bpm + 4.0 * np.sin(
        2.0 * np.pi * times / 300.0 + rng.uniform(0.0, 2.0 * np.pi)
    )
    rate += 0.6 * np.sin(
        2.0 * np.pi * times / 41.0 + rng.uniform(0.0, 2.0 * np.pi)
    )
    return rate, times


def _beat_phase(rate: np.ndarray, fs: float) -> np.ndarray:
    """Converts an instantaneous heart rate into a wrapped beat phase."""
    return np.mod(np.cumsum(rate / 60.0) / fs, 1.0)


def _cardiac_signal(
    n: int, fs: float, seed: int, kind: str, bpm: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a clean ECG or PPG waveform and its heart-rate trace.

    The signals have exactly one positive peak per cardiac cycle, which keeps
    the scipy peak-detector fallback stable without needing optional packages.
    """
    rate, _ = _heart_rate(n, fs, seed, bpm)
    phase = _beat_phase(rate, fs)
    if kind == "ecg":
        r_peak = np.exp(-0.5 * ((phase - 0.25) / 0.012) ** 2)
        q_wave = -0.15 * np.exp(-0.5 * ((phase - 0.22) / 0.020) ** 2)
        s_wave = -0.25 * np.exp(-0.5 * ((phase - 0.28) / 0.020) ** 2)
        return r_peak + q_wave + s_wave, rate
    if kind == "ppg":
        return np.exp(-0.5 * ((phase - 0.20) / 0.075) ** 2), rate
    raise ValueError(f"unknown cardiac signal {kind!r}")


def _respiratory_signal(n: int, fs: float, seed: int) -> np.ndarray:
    """Returns a slow respiratory-effort waveform."""
    rng = np.random.default_rng(seed)
    times = np.arange(n, dtype=np.float64) / fs
    phase = rng.uniform(0.0, 2.0 * np.pi)
    return np.sin(2.0 * np.pi * 0.25 * times + phase)


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Centers a signal and scales it to ``[-0.9, 0.9]``."""
    array = np.asarray(signal, dtype=np.float64)
    array = array - array.mean()
    peak = float(np.max(np.abs(array)))
    if peak > 0.0:
        array = 0.9 * array / peak
    return array


def _pad(value: object, width: int, align: str = "<") -> str:
    """Pads or truncates an ASCII field to an EDF header width."""
    text = str(value)
    if len(text) > width:
        text = text[:width]
    return f"{text:{align}{width}}"


def _write_edf(
    path: Path,
    channels: list[tuple[str, np.ndarray, float]],
    n_records: int,
) -> None:
    """Writes a minimal EDF+C file readable by PyHealth's NSRR loader.

    Args:
        path: Destination ``.edf`` file.
        channels: ``(label, signal, sampling_rate_hz)`` triples. Each signal
            must contain ``n_records * sampling_rate * 30`` samples.
        n_records: Number of 30 s data records.
    """
    record_samples = [
        int(round(rate * _RECORD_SECONDS)) for _, _, rate in channels
    ]
    for (label, signal, _), count in zip(channels, record_samples):
        expected = n_records * count
        if signal.size != expected:
            raise ValueError(
                f"{label} has {signal.size} samples, expected {expected}"
            )
    n_signals = len(channels)
    header_bytes = 256 + 256 * n_signals
    per_record = sum(record_samples)

    header = "".join(
        (
            _pad("0", 8),
            _pad("Synthetic WatchSleepNet", 80),
            _pad("Synthetic recording", 80),
            _pad("01.01.00", 8),
            _pad("00.00.00", 8),
            _pad(header_bytes, 8, ">"),
            _pad("EDF+C", 44),
            _pad(n_records, 8, ">"),
            _pad(f"{_RECORD_SECONDS:g}", 8, ">"),
            _pad(n_signals, 4, ">"),
        )
    )
    # EDF signal headers are column-major: every label, then every
    # transducer, and so on. This is the layout PyHealth's reader expects.
    for label, _, _ in channels:
        header += _pad(label, 16)
    for _ in channels:
        header += _pad("synthetic", 80)
    for _ in channels:
        header += _pad("mV", 8)
    for value in ("-1", "1", "-32768", "32767"):
        for _ in channels:
            header += _pad(value, 8, ">")
    for _ in channels:
        header += _pad("", 80)
    for count in record_samples:
        header += _pad(count, 8, ">")
    for _ in channels:
        header += _pad("", 32)

    data = np.empty((n_records, per_record), dtype="<i2")
    offset = 0
    for (_, signal, _), count in zip(channels, record_samples):
        scaled = (_normalize(signal) + 1.0) * 32767.5 - 32768.0
        raw = np.clip(np.rint(scaled), -32768, 32767).astype(np.int16)
        data[:, offset : offset + count] = raw.reshape(n_records, count)
        offset += count
    path.write_bytes(header.encode("ascii") + data.tobytes())


def _write_profusion_xml(path: Path, stages: list[object]) -> None:
    """Writes a minimal NSRR profusion XML with one stage per epoch."""
    root = ET.Element("CMPStudyConfig")
    ET.SubElement(root, "EpochLength").text = str(int(_RECORD_SECONDS))
    sleep_stages = ET.SubElement(root, "SleepStages")
    for stage in stages:
        ET.SubElement(sleep_stages, "SleepStage").text = str(int(stage))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _write_nsrr_metadata(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Writes the metadata CSV that NSRRSleepDataset normally builds."""
    frame = pd.DataFrame(
        rows,
        columns=("patient_id", "visit", "signal_file", "label_file"),
    )
    frame.to_csv(path, index=False)


def generate_shhs(
    output: Path, num_samples: int, epochs: int, seed: int
) -> list[Path]:
    """Generates SHHS-shaped EDF, XML, and metadata files."""
    edf_dir = output / "edfs" / "shhs1"
    xml_dir = output / "annotations-events-profusion" / "shhs1"
    edf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    files: list[Path] = []
    for index in range(num_samples):
        patient_id = f"{200001 + index:06d}"
        run_seed = seed + index
        ecg, _ = _cardiac_signal(epochs * 125 * 30, 125.0, run_seed, "ecg")
        thorax = _respiratory_signal(epochs * 32 * 30, 32.0, run_seed + 1)
        abdomen = _respiratory_signal(epochs * 32 * 30, 32.0, run_seed + 2)

        edf_path = edf_dir / f"shhs1-{patient_id}.edf"
        _write_edf(
            edf_path,
            [
                ("ECG", ecg, 125.0),
                ("THOR RES", thorax, 32.0),
                ("ABDO RES", abdomen, 32.0),
            ],
            epochs,
        )
        stages = _stage_sequence("shhs", epochs, run_seed)
        xml_path = xml_dir / f"shhs1-{patient_id}-profusion.xml"
        _write_profusion_xml(xml_path, stages)
        rows.append(
            {
                "patient_id": patient_id,
                "visit": "shhs1",
                "signal_file": str(edf_path.resolve()),
                "label_file": str(xml_path.resolve()),
            }
        )
        files.extend((edf_path, xml_path))

    _write_nsrr_metadata(
        output / "nsrr-shhs-pyhealth.csv", rows
    )
    return files


def generate_mesa(
    output: Path, num_samples: int, epochs: int, seed: int
) -> list[Path]:
    """Generates MESA-shaped EDF, XML, and metadata files."""
    edf_dir = output / "edfs"
    xml_dir = output / "annotations-events-profusion"
    edf_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    files: list[Path] = []
    for index in range(num_samples):
        patient_id = f"{1 + index:04d}"
        run_seed = seed + index
        pleth, _ = _cardiac_signal(
            epochs * 256 * 30, 256.0, run_seed, "ppg"
        )
        ekg, _ = _cardiac_signal(
            epochs * 256 * 30, 256.0, run_seed + 1, "ecg"
        )
        thorax = _respiratory_signal(epochs * 32 * 30, 32.0, run_seed + 2)
        abdomen = _respiratory_signal(epochs * 32 * 30, 32.0, run_seed + 3)

        edf_path = edf_dir / f"mesa-sleep-{patient_id}.edf"
        _write_edf(
            edf_path,
            [
                ("Pleth", pleth, 256.0),
                ("EKG", ekg, 256.0),
                ("Thor", thorax, 32.0),
                ("Abdo", abdomen, 32.0),
            ],
            epochs,
        )
        stages = _stage_sequence("mesa", epochs, run_seed)
        xml_path = (
            xml_dir / f"mesa-sleep-{patient_id}-profusion.xml"
        )
        _write_profusion_xml(xml_path, stages)
        rows.append(
            {
                "patient_id": patient_id,
                "visit": "mesa",
                "signal_file": str(edf_path.resolve()),
                "label_file": str(xml_path.resolve()),
            }
        )
        files.extend((edf_path, xml_path))

    _write_nsrr_metadata(
        output / "nsrr-mesa-pyhealth.csv", rows
    )
    return files


def _dreamt_frame(
    columns: tuple[str, ...],
    signal: np.ndarray,
    rate: np.ndarray,
    stages: list[object],
    samples_per_epoch: int,
) -> pd.DataFrame:
    """Builds one DREAMT-style frame with mock values in unused columns."""
    n = signal.size
    values: dict[str, np.ndarray] = {
        column: np.zeros(n, dtype=np.float64) for column in columns
    }
    values["TIMESTAMP"] = np.arange(n, dtype=np.float64) / (
        samples_per_epoch / _RECORD_SECONDS
    )
    values["BVP"] = signal
    values["HR"] = rate
    values["IBI"] = np.divide(
        60.0, rate, out=np.zeros_like(rate), where=rate > 0
    )
    frame = pd.DataFrame(values, columns=list(columns))
    frame["Sleep_Stage"] = np.repeat(
        np.asarray(stages, dtype=object), samples_per_epoch
    )
    return frame


def generate_dreamt(
    output: Path, num_samples: int, epochs: int, seed: int
) -> list[Path]:
    """Generates DREAMT-shaped 100 Hz and 64 Hz CSV files."""
    data_100 = output / "data_100Hz"
    data_64 = output / "data_64Hz"
    data_100.mkdir(parents=True, exist_ok=True)
    data_64.mkdir(parents=True, exist_ok=True)

    participant_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    files: list[Path] = []
    for index in range(num_samples):
        patient_id = f"S{1 + index:03d}"
        run_seed = seed + index
        stages = _stage_sequence("dreamt", epochs, run_seed)

        bvp_100, rate_100 = _cardiac_signal(
            epochs * 100 * 30, 100.0, run_seed, "ppg"
        )
        frame_100 = _dreamt_frame(
            _DREAMT_100HZ_COLUMNS, bvp_100, rate_100, stages, 100 * 30
        )
        ecg_100, _ = _cardiac_signal(
            epochs * 100 * 30, 100.0, run_seed + 1, "ecg"
        )
        frame_100["ECG"] = ecg_100
        frame_100["SAO2"] = 95.0
        frame_100["TEMP"] = 32.0
        frame_100["EDA"] = 1.0
        path_100 = data_100 / f"{patient_id}_PSG_df.csv"
        frame_100.to_csv(path_100, index=False, float_format="%.4f")

        bvp_64, rate_64 = _cardiac_signal(
            epochs * 64 * 30, 64.0, run_seed + 2, "ppg"
        )
        frame_64 = _dreamt_frame(
            _DREAMT_64HZ_COLUMNS, bvp_64, rate_64, stages, 64 * 30
        )
        path_64 = data_64 / f"{patient_id}_whole_df.csv"
        frame_64.to_csv(path_64, index=False, float_format="%.4f")

        participant_rows.append(
            {
                "SID": patient_id,
                "AGE": 50.0 + index,
                "GENDER": "M" if index % 2 == 0 else "F",
                "BMI": 25.0 + index,
                "OAHI": 5.0 + index,
                "AHI": 6.0 + index,
                "Mean_SaO2": "95%",
                "Arousal Index": 10.0 + index,
                "MEDICAL_HISTORY": "None",
                "Sleep_Disorders": "None",
            }
        )
        metadata_rows.append(
            {
                "patient_id": patient_id,
                "age": 50.0 + index,
                "gender": "M" if index % 2 == 0 else "F",
                "bmi": 25.0 + index,
                "oahi": 5.0 + index,
                "ahi": 6.0 + index,
                "mean_sao2": 95.0,
                "arousal_index": 10.0 + index,
                "medical_history": "None",
                "sleep_disorders": "None",
                "file_64hz": str(path_64.resolve()),
                "file_100hz": str(path_100.resolve()),
            }
        )
        files.extend((path_100, path_64))

    pd.DataFrame(participant_rows).to_csv(
        output / "participant_info.csv", index=False
    )
    pd.DataFrame(metadata_rows).to_csv(
        output / "dreamt-metadata.csv", index=False
    )
    return files


def _prepare_output(
    path: Path, overwrite: bool
) -> Path:
    """Creates the destination, optionally replacing existing contents."""
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"{path} exists and is not a directory")
        if any(path.iterdir()):
            if not overwrite:
                raise SystemExit(
                    f"{path} is not empty; pass --overwrite to replace it"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    """Generates the requested synthetic dataset tree."""
    args = build_parser().parse_args(argv)
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")

    output = args.output_dir or (
        Path("synthetic_data") / args.dataset
    )
    output = _prepare_output(output, args.overwrite)
    generators = {
        "shhs": generate_shhs,
        "mesa": generate_mesa,
        "dreamt": generate_dreamt,
    }
    files = generators[args.dataset](
        output, args.num_samples, args.epochs, args.seed
    )

    print(
        f"Generated {args.num_samples} {args.dataset} recording(s) "
        f"with {args.epochs} epochs each"
    )
    print(f"Output: {output.resolve()}")
    print(f"Files:  {len(files)}")
    print("Run the pipeline with:")
    print(
        "  python examples/watchsleepnet/watchsleepnet_finetune.py "
        f"--dataset {args.dataset} --root {output} --epochs 2 "
        f"--output-dir {output / 'output'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
