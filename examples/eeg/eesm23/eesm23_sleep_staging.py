"""Sleep staging on EESM23 ear-EEG with a subject-wise split.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: Loads OpenNeuro ds005178, builds 30-second epochs with
    SleepStagingEarEEG, and fine-tunes TFMTokenizer. The split is the
    TFM-Tokenizer subject split (sub-001-sub-006 train, sub-007-sub-008
    validation, sub-009-sub-010 test).

References:
    - Mikkelsen, K. B. et al. Ear-EEG sleep monitoring data sets.
      Scientific Data 12, 2025.
      https://www.nature.com/articles/s41597-025-04579-8
    - Rezai Tabar, Y. et al. Ear-EEG Sleep Monitoring 2023 (EESM23).
      https://doi.org/10.18112/openneuro.ds005178.v1.0.0
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer

Download ds005178 from OpenNeuro first, then run::

    python examples/eeg/eesm23/eesm23_sleep_staging.py --root /path/to/ds005178

The full corpus is large. Pass ``--subjects`` to try a few cases first,
for example ``--subjects 1 2 3``. Only ``ses-001`` and ``ses-002`` nights
carry manual scoring.
"""

import argparse

from pyhealth.datasets import (
    EESM23Dataset,
    SubjectSplit,
    get_dataloader,
)
from pyhealth.models import TFMTokenizer
from pyhealth.tasks import SleepStagingEarEEG
from pyhealth.tasks.sleep_staging_ear_eeg import normalize_subject
from pyhealth.trainer import Trainer


def main() -> None:
    """Train TFMTokenizer on EESM23 sleep staging.

    Args:
        None. Options come from the command line: ``--root``, ``--epochs``,
        ``--batch-size``, ``--device``, ``--subjects``, ``--resample-rate``
        and ``--include-artefact``.

    Returns:
        None. Prints the test-set metrics.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="BIDS root of OpenNeuro ds005178 (the folder that contains "
        "sub-001, sub-002, ...).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resample-rate",
        type=float,
        default=200,
        help="Target sampling rate after per-epoch resampling.",
    )
    parser.add_argument(
        "--include-artefact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Artefact as class 5 (TFM-Tokenizer default).",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject ids. With three or more, the first is train, "
        "the second is validation and the rest are test.",
    )
    args = parser.parse_args()

    dataset = EESM23Dataset(root=args.root)
    print(dataset.stats())

    if args.subjects:
        subjects = [normalize_subject(s) for s in args.subjects]
        if len(subjects) < 3:
            parser.error("--subjects needs at least one train, one val and one test id")
        split = SubjectSplit((subjects[:1], subjects[1:2], subjects[2:]))
    else:
        split = dataset.default_subject_split()

    task = SleepStagingEarEEG(
        resample_rate=args.resample_rate,
        normalization="95th_percentile",
        include_artefact=args.include_artefact,
        compute_stft=True,
        subjects=[s for part in split.subjects for s in part],
    )
    train_dataset, val_dataset, test_dataset = dataset.set_task(task, split=split)
    print(
        "samples train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = TFMTokenizer(dataset=train_dataset)
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["accuracy", "balanced_accuracy", "f1_macro", "cohen_kappa"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="cohen_kappa",
        optimizer_params={"lr": 1e-4},
    )
    print(trainer.evaluate(test_loader))


if __name__ == "__main__":
    main()
