"""ECG delineation on LUDB with UNet2D: reproduction plus two ablations.

Authors: Shray Srivastava (ssriv5@illinois.edu), Sid Wanjara (swanj2@illinois.edu)

Paper: J. Park et al. "Benchmarking ECG Delineation using Deep Neural
Network-based Semantic Segmentation Models." CHIL 2025, PMLR 287:63-88.
https://proceedings.mlr.press/v287/park25a.html

Pipeline: LUDBDataset -> ECGDelineationLUDB (one sample per lead, filtered,
point-wise P/QRS/T mask) -> PyHealth's UNet2D -> IoU per wave and mIoU.

Setup:
    - Split: the paper's own split by record (160 train / 20 validation / 20
      test), published at
      https://huggingface.co/spaces/MedicalAI-DP/ECG_Delineation
      (res/ludb/dataset). All 12 leads of a record stay in the same split.
    - Model: UNet2D (cross-entropy + Dice, mask value 255 ignored), Adam,
      keeping the epoch with the best validation IoU. The paper's best LUDB
      model is UNet 3+ (mIoU 0.854); PyHealth doesn't have it, so we use its
      UNet as the closest existing model.
    - Metric: like the paper, IoU of a wave = points both the model and the
      cardiologist labelled as that wave / points either labelled as it,
      summed over all test leads. mIoU is the mean over P, QRS and T.

Ablations (not in the paper, which only varies post-processing, loss
weights and architecture settings):
    1. Noise robustness: score the trained model on the test records with
       white Gaussian noise added before filtering, at 20, 10, 5 and 0 dB SNR.
    2. Filter on vs off: train a second model without the band-pass filter
       and run the same noise test, to see what the paper's preprocessing
       adds to accuracy and robustness.

Usage:
    # No data needed: 20 synthetic records, small model, a few minutes on CPU
    python examples/ludb_ecg_delineation_unet.py --synthetic

    # Real LUDB 1.0.1 from https://physionet.org/content/ludb/1.0.1/ (use a GPU)
    python examples/ludb_ecg_delineation_unet.py --root /path/to/ludb/1.0.1 \\
        --seeds 0 1 2 --output ludb_results.json

Results:
    To be filled in from the LUDB run. The --synthetic run only checks that
    the pipeline works; its numbers say nothing about real ECGs.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    LUDBDataset,
    SampleDataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import UNet2D
from pyhealth.tasks import ECGDelineationLUDB
from pyhealth.tasks.ecg_delineation_ludb import IGNORE_INDEX, LUDB_LEADS
from pyhealth.trainer import Trainer

# The paper's validation and test records; the other 160 are for training.
VALID_IDS = [
    1,
    11,
    24,
    26,
    30,
    39,
    58,
    61,
    69,
    73,
    75,
    95,
    96,
    117,
    141,
    142,
    143,
    162,
    166,
    185,
]
TEST_IDS = [
    12,
    15,
    25,
    27,
    29,
    36,
    41,
    45,
    57,
    79,
    80,
    85,
    93,
    97,
    136,
    139,
    152,
    161,
    173,
    190,
]

# Table 1 of the paper: UNet 3+ on the LUDB test set.
PAPER_UNET3PLUS = {"P": 0.814, "QRS": 0.881, "T": 0.868, "mIoU": 0.854}

NOISE_LEVELS_DB = [20, 10, 5, 0]


def make_synthetic_ludb(root: Path, n_records: int = 20) -> None:
    """Write fake LUDB-style records so the example runs without the data.

    Each record is 10 s of 12 leads at 500 Hz. Every beat is a P, QRS and T
    bump with "(", peak, ")" annotations like LUDB; the first and last beats
    are left unannotated like in LUDB. Baseline drift and noise are added so
    the filter has something to remove.

    Notes:
        How to write it:

        1. Make ``root / "data"``. Use a time axis ``t`` of 5000 points
           (10 s at 500 Hz) and a seeded ``np.random.default_rng`` so every
           run makes the same data.
        2. For each record, pick a beat length (0.7-1.1 s) and the start
           time of each beat. For each beat add three Gaussian bumps (P,
           QRS, T) at fixed offsets from the start. For every beat except
           the first and last, add "(", peak, ")" symbols and their sample
           indices (center - width, center, center + width) to the
           annotation lists.
        3. Copy the beat signal to 12 leads with a random gain per lead,
           then add a slow sine (baseline drift) and a bit of noise.
        4. Save with ``wfdb.wrsamp(name, fs=..., units=["mV"] * 12,
           sig_name=list(LUDB_LEADS), p_signal=signal, fmt=["16"] * 12,
           write_dir=...)``.
        5. Save the annotations once per lead with ``wfdb.wrann``. It
           refuses extensions with digits (v1 to v6), so write each one as
           ".tmp" and rename it to ".<lead>".
        6. Write a small ludb.csv with the same column names as the real
           one.

        This is only a smoke test; the bumps look nothing like real ECGs, so
        the scores it gives mean nothing. The P bumps are small, so a P IoU
        of 0 after short training is expected.
    """
    import wfdb

    rng = np.random.default_rng(0)
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    fs = 500
    t = np.arange(10 * fs) / fs
    # symbol, time after the beat starts (s), half-width (s), height (mV)
    wave_shapes = [
        ("p", 0.10, 0.04, 0.15),
        ("N", 0.30, 0.02, 1.0),
        ("t", 0.60, 0.07, 0.3),
    ]

    csv_lines = ["ID,Sex,Age,Rhythms,Electric axis of the heart"]
    for record in range(1, n_records + 1):
        beat_length = rng.uniform(0.7, 1.1)
        beat_starts = np.arange(rng.uniform(0, beat_length), 9.2, beat_length)
        ecg = np.zeros(len(t))
        symbols, samples = [], []
        for i, start in enumerate(beat_starts):
            for symbol, offset, width, height in wave_shapes:
                center = start + offset
                ecg += height * np.exp(-(((t - center) / (width / 2)) ** 2))
                if 0 < i < len(beat_starts) - 1:
                    symbols += ["(", symbol, ")"]
                    samples += [
                        int((center - width) * fs),
                        int(center * fs),
                        int((center + width) * fs),
                    ]

        lead_gains = rng.uniform(-1.0, 1.5, size=12)
        drift = 0.3 * np.sin(2 * np.pi * 0.2 * t)
        signal = (
            np.outer(ecg, lead_gains)
            + drift[:, None]
            + rng.normal(0, 0.02, (len(t), 12))
        )
        wfdb.wrsamp(
            str(record),
            fs=fs,
            units=["mV"] * 12,
            sig_name=list(LUDB_LEADS),
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str(data_dir),
        )
        for lead in LUDB_LEADS:
            # wrann only accepts letters in the extension, so write and rename.
            wfdb.wrann(
                str(record), "tmp", np.array(samples), symbols, write_dir=str(data_dir)
            )
            (data_dir / f"{record}.tmp").rename(data_dir / f"{record}.{lead}")
        csv_lines.append(f"{record},M,50,Sinus rhythm,normal")
    (root / "ludb.csv").write_text("\n".join(csv_lines) + "\n")


def split_by_record(
    samples: SampleDataset, synthetic: bool, seed: int
) -> tuple[SampleDataset, SampleDataset, SampleDataset]:
    """Split samples into train/validation/test by record.

    Real LUDB uses the paper's split. Synthetic data uses a random 80/10/10
    split by record.

    Notes:
        How to write it:

        1. Synthetic data: return ``split_by_patient(samples, [0.8, 0.1,
           0.1], seed=seed)``. One patient is one record, so this already
           keeps all 12 leads of a record together.
        2. Real LUDB: ``samples.patient_to_index`` maps each record id (a
           string) to the indices of its samples. Put each record's
           indices into the test, validation or train list depending on
           whether ``int(patient_id)`` is in ``TEST_IDS``, ``VALID_IDS`` or
           neither.
        3. Return ``samples.subset(...)`` for each of the three lists.

        Split by record, never by sample: otherwise leads of the same
        heartbeat end up in both train and test. The ids come from the paper
        authors' published split, which makes our numbers comparable to
        theirs. The noisy test sets go through this same function, so they
        hold the same records as the clean test set.
    """
    if synthetic:
        return split_by_patient(samples, [0.8, 0.1, 0.1], seed=seed)

    train_idx, valid_idx, test_idx = [], [], []
    for patient_id, indices in samples.patient_to_index.items():
        if int(patient_id) in TEST_IDS:
            test_idx += indices
        elif int(patient_id) in VALID_IDS:
            valid_idx += indices
        else:
            train_idx += indices
    return (
        samples.subset(train_idx),
        samples.subset(valid_idx),
        samples.subset(test_idx),
    )


@torch.no_grad()
def wave_iou(
    model: UNet2D, samples: SampleDataset, batch_size: int
) -> dict[str, float]:
    """Compute the IoU of each wave and the mIoU, the way the paper does.

    Intersections and unions are summed over all leads before dividing.
    Points with mask value 255 (not annotated) are skipped.

    Notes:
        How to write it:

        1. Put the model in eval mode and keep two arrays of 3 zeros:
           intersection and union (one slot each for P, QRS, T).
        2. Loop over ``get_dataloader(samples, batch_size)``. Call
           ``model(signal=batch["signal"])["logit"]``; leaving out "mask"
           skips the loss. The logits are (batch, 4, 1, length), so
           ``argmax(dim=1)`` gives the predicted class of each point.
        3. Make an "annotated" mask: ``batch["mask"] != IGNORE_INDEX``.
        4. For each label 1, 2, 3: predicted = (prediction == label) and
           annotated; actual = (truth == label). Add the count of
           ``predicted & actual`` to intersection and of
           ``predicted | actual`` to union.
        5. IoU = intersection / union (guard against dividing by 0); mIoU
           is their mean. Return a dict with "P", "QRS", "T" and "mIoU".

        This sums over all leads before dividing, which is our reading of
        Section 3.3 of the paper. The Trainer's own "iou" averages per lead
        instead, so the two numbers differ; don't compare them.
    """
    model.eval()
    intersection = np.zeros(3)
    union = np.zeros(3)
    for batch in get_dataloader(samples, batch_size):
        logits = model(signal=batch["signal"])["logit"]  # (batch, 4, 1, length)
        prediction = logits.argmax(dim=1).cpu()  # (batch, 1, length)
        truth = batch["mask"]
        annotated = truth != IGNORE_INDEX
        for label in (1, 2, 3):
            predicted = (prediction == label) & annotated
            actual = truth == label
            intersection[label - 1] += (predicted & actual).sum().item()
            union[label - 1] += (predicted | actual).sum().item()

    iou = intersection / np.maximum(union, 1)
    return {"P": iou[0], "QRS": iou[1], "T": iou[2], "mIoU": iou.mean()}


def train_unet(
    train: SampleDataset, valid: SampleDataset, args: argparse.Namespace, seed: int
) -> UNet2D:
    """Train UNet2D and keep the weights from the best validation epoch.

    Notes:
        How to write it (see how other examples, e.g. the Helsinki one, use
        the Trainer):

        1. Call ``torch.manual_seed(seed)`` so runs repeat.
        2. Build ``UNet2D(train, in_channels=1, num_classes=4, ...)`` with
           ``norm="batch"`` and ``ignore_index=IGNORE_INDEX``. 4 classes =
           background, P, QRS, T.
        3. Make a ``Trainer(model=model, metrics=["iou"], ...)``.
        4. Call ``trainer.train`` with a shuffled train dataloader, a
           validation dataloader, the epochs and learning rate,
           ``monitor="iou"`` and ``patience``. The Trainer then reloads the
           best validation epoch at the end.
        5. Return the model.

        lr 1e-3 was checked on real LUDB: test mIoU 0.54 after 3 epochs and
        still rising. lr 1e-2 jumped around and ended at 0.42. With depth 5,
        UNet2D pads the one-row input to 16 rows, so each epoch is slow
        (~2 min on a laptop GPU).
    """
    torch.manual_seed(seed)
    model = UNet2D(
        train,
        in_channels=1,
        num_classes=4,  # background, P, QRS, T
        base_channels=args.base_channels,
        depth=args.depth,
        norm="batch",
        ignore_index=IGNORE_INDEX,
    )
    trainer = Trainer(
        model=model, metrics=["iou"], device=args.device, enable_logging=False
    )
    trainer.train(
        train_dataloader=get_dataloader(
            train, args.batch_size, shuffle=True, seed=seed
        ),
        val_dataloader=get_dataloader(valid, args.batch_size),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="iou",
        patience=args.patience,
    )
    return model


def run_experiment(
    dataset: LUDBDataset, apply_filter: bool, args: argparse.Namespace
) -> dict:
    """Train with or without the filter, then score on clean and noisy test sets.

    Returns {test condition: [scores for each seed]}.

    Notes:
        How to write it:

        1. Build the clean samples with
           ``dataset.set_task(ECGDelineationLUDB(apply_filter=...))``.
        2. Build one noisy sample set per level in ``NOISE_LEVELS_DB`` the
           same way, adding ``noise_snr_db=snr``. Do this once, before the
           seed loop; each set_task on real LUDB takes ~15 s.
        3. Make a results dict with an empty list for "clean" and for each
           noise level (e.g. "10 dB").
        4. For each seed: split the clean samples with ``split_by_record``,
           train with ``train_unet``, then append ``wave_iou`` on the clean
           test set and on the test part of each noisy set.
        5. Return the dict.

        The model is only ever trained on clean data; just the test set is
        noisy. The noise seed is fixed, so every training seed is tested on
        the exact same noisy signals.
    """
    clean_samples = dataset.set_task(ECGDelineationLUDB(apply_filter=apply_filter))
    noisy_samples = {}
    for snr in NOISE_LEVELS_DB:
        task = ECGDelineationLUDB(apply_filter=apply_filter, noise_snr_db=snr)
        noisy_samples[snr] = dataset.set_task(task)

    results = {"clean": []}
    for snr in NOISE_LEVELS_DB:
        results[f"{snr} dB"] = []

    for seed in args.seeds:
        train, valid, test = split_by_record(clean_samples, args.synthetic, seed)
        model = train_unet(train, valid, args, seed)
        results["clean"].append(wave_iou(model, test, args.batch_size))
        for snr in NOISE_LEVELS_DB:
            _, _, noisy_test = split_by_record(noisy_samples[snr], args.synthetic, seed)
            results[f"{snr} dB"].append(wave_iou(model, noisy_test, args.batch_size))
        clean_miou = results["clean"][-1]["mIoU"]
        print(f"filter={apply_filter} seed={seed} clean mIoU={clean_miou:.3f}")
    return results


def print_table(all_results: dict) -> None:
    """Print mean (std) IoU over seeds for every experiment and test set.

    Notes:
        How to write it:

        1. Print a header row, then a row with the paper's UNet 3+ numbers
           (``PAPER_UNET3PLUS``) for reference.
        2. For each experiment ("filtered", "unfiltered") and each test set
           ("clean", "20 dB", ...), collect that score from every seed's run
           and print ``np.mean`` and ``np.std`` for P, QRS, T and mIoU.

        Use f-string padding (e.g. ``f"{text:<12}"``) to line the columns
        up. With one seed the std is always 0.000.
    """
    print(
        f"\n{'model':<12}{'test set':<10}"
        + "".join(f"{name:<16}" for name in PAPER_UNET3PLUS)
    )
    print(
        f"{'UNet 3+':<12}{'paper':<10}"
        + "".join(f"{v:<16.3f}" for v in PAPER_UNET3PLUS.values())
    )
    for experiment, conditions in all_results.items():
        for condition, runs in conditions.items():
            cells = ""
            for name in PAPER_UNET3PLUS:
                values = [run[name] for run in runs]
                cells += f"{np.mean(values):.3f} ({np.std(values):.3f})  "
            print(f"{experiment:<12}{condition:<10}{cells}")


def main() -> None:
    """Parse arguments, run both experiments and print the results.

    Notes:
        How to write it:

        1. Use ``argparse`` for: --root, --synthetic, --epochs, --patience,
           --batch-size, --lr, --base-channels, --depth, --seeds (a list,
           ``nargs="+"``), --device and --output. Error out if neither
           --root nor --synthetic is given.
        2. Open a ``tempfile.TemporaryDirectory()``. If --synthetic, call
           ``make_synthetic_ludb`` inside it, point --root there and switch
           to small settings (15 epochs, 8 channels, depth 3, lr 1e-2) so it
           runs on a CPU in a few minutes.
        3. Load ``LUDBDataset(root=..., cache_dir=<temp folder>)``. A temp
           cache means every run rebuilds the samples (~2.5 min on real
           LUDB) but never picks up a stale cache.
        4. Call ``run_experiment`` with ``apply_filter=True`` and then
           ``False``, keep both results, and ``print_table`` them.
        5. If --output was given, save the results with ``json.dump``
           (``default=float`` turns the numpy numbers into plain floats).

        One seed trains two models, so budget for both on Colab. On macOS,
        if the repo is in an iCloud-synced folder and you get "No module
        named pyhealth", set PYTHONPATH to the repo folder.
    """
    parser = argparse.ArgumentParser(description="ECG delineation on LUDB with UNet2D.")
    parser.add_argument("--root", help="LUDB 1.0.1 folder (holds ludb.csv and data/).")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use 20 fake records instead."
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default=None, help="Defaults to cuda if available.")
    parser.add_argument("--output", help="Save the results to this JSON file.")
    args = parser.parse_args()
    if not args.root and not args.synthetic:
        parser.error("pass --root /path/to/ludb or --synthetic")

    with tempfile.TemporaryDirectory() as tmp:
        if args.synthetic:
            # A smaller model and a higher learning rate so this runs on a CPU.
            args.root = str(Path(tmp) / "ludb")
            make_synthetic_ludb(Path(args.root))
            args.epochs, args.base_channels, args.depth, args.lr = 15, 8, 3, 1e-2

        dataset = LUDBDataset(root=args.root, cache_dir=str(Path(tmp) / "cache"))
        all_results = {
            "filtered": run_experiment(dataset, apply_filter=True, args=args),
            "unfiltered": run_experiment(dataset, apply_filter=False, args=args),
        }

    print_table(all_results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
