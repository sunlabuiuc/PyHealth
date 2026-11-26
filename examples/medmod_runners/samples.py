"""Build and check a MedMod IHM sample set from the command line.

    python -m examples.medmod_runners.samples --data-root /path/to/mimic_data \\
        --hours-zero keep

In Python, use the PyHealth pattern directly (see ``MedModInHospitalMortality``)::

    dataset = MedModMIMIC4Dataset(ehr_root=..., cxr_root=..., cache_dir=...)
    task = MedModInHospitalMortality(hours_zero="keep", image_dir=...)
    train, val, test = dataset.set_task(task, split=MedModSplit())

The default task uses MedMod's released-code settings (1-hour bins, phenotyping
normalizer file): ehr 48 x 76, the set that matches MedMod's own extraction.
``--timestep 2`` gives 2-hour bins (ehr 24 x 76); ``--fit-normalizer`` fits the
normalizer on the train split instead of loading MedMod's file.

Two sample sets share one cohort and differ only in ``ehr`` (rows at Hours == 0):

    hours_zero="keep"  MedFuse's loader (datasets/ehr_dataset.py) → Early Fusion
    hours_zero="drop"  MedMod's loader (src/data/datasets.py)     → ConVIRT
    (MedModInHospitalMortality docs, "Per-model settings")

PyHealth caches each set under ``cache_dir``, keyed by the task's settings
(``BaseDataset._task_cache_dir``). The first build runs the task over every patient
(~15 min with 4 workers); later builds with the same settings load in seconds.
With ``num_workers > 1`` PyHealth starts "spawn" worker processes
(``BaseDataset._task_transform``) that re-import the main script. Run this as a
module, or keep your own code under ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from pyhealth.datasets.medmod_mimic4 import SPLITS, MedModSplit

HOURS_ZERO = ("drop", "keep")
# the image-paired cohort on the real data (mimic4_medmod_ihm_results.md, Settings)
VERIFIED_SPLITS = {"train": 4485, "val": 488, "test": 1241}


@dataclass(frozen=True)
class MedModPaths:
    """Where the local MIMIC data lives. ``from_root`` fills in the layout the
    pipeline writes (README.md in this folder, "Data layout"); pass any field to
    override it."""

    ehr_root: Path
    cxr_root: Path
    cache_dir: Path
    image_dir: Optional[Path]

    @classmethod
    def from_root(
        cls,
        data_root: str | Path,
        ehr_root: Optional[str | Path] = None,
        cxr_root: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        use_resized_images: bool = True,
    ) -> "MedModPaths":
        root = Path(data_root)
        cxr = Path(cxr_root) if cxr_root else root / "mimic-cxr-jpg" / "2.0.0"
        images = Path(image_dir) if image_dir else cxr / "resized"
        return cls(
            ehr_root=(
                Path(ehr_root) if ehr_root else root / "mimiciv_medmod_slim" / "1.0"
            ),
            cxr_root=cxr,
            cache_dir=Path(cache_dir) if cache_dir else root / "pyhealth_cache",
            image_dir=images if use_resized_images else None,
        )

    def check(self) -> None:
        """Fail early, with the folder name, if an input folder is missing."""
        needed = {"ehr_root": self.ehr_root, "cxr_root": self.cxr_root}
        if self.image_dir is not None:
            needed["image_dir"] = self.image_dir
        missing = [f"{k}={v}" for k, v in needed.items() if not Path(v).is_dir()]
        if missing:
            raise FileNotFoundError(
                "missing folder(s): " + ", ".join(missing) + ". Check --data-root "
                "(examples/medmod_runners/README.md, 'Data layout') or pass the "
                "folder explicitly."
            )

    def open_dataset(self, num_workers: int = 1, dev: bool = False):
        """``MedModMIMIC4Dataset`` on these folders (nothing loads until used)."""
        from pyhealth.datasets.medmod_mimic4 import MedModMIMIC4Dataset

        return MedModMIMIC4Dataset(
            ehr_root=self.ehr_root,
            cxr_root=self.cxr_root,
            cache_dir=self.cache_dir,
            dev=dev,
            num_workers=num_workers,
        )


def make_task(
    paths: MedModPaths,
    hours_zero: str,
    timestep: float = 1.0,
    fit_normalizer: bool = False,
):
    """The task with these paths' images. Defaults: MedMod's released-code EHR
    settings (1-hour bins, phenotyping normalizer file)."""
    from pyhealth.datasets.medmod.discretizer import PH_NORMALIZER
    from pyhealth.tasks.medmod_mortality_mimic4 import MedModInHospitalMortality

    return MedModInHospitalMortality(
        hours_zero=hours_zero,
        timestep=timestep,
        normalizer=None if fit_normalizer else PH_NORMALIZER,
        image_dir=paths.image_dir,
    )


def build_samples(paths: MedModPaths, task, num_workers: int = 1):
    """The PyHealth pattern: dataset → ``set_task`` with MedMod's split.
    Returns ``(dataset, (train, val, test))``."""
    paths.check()
    dataset = paths.open_dataset(num_workers)
    parts = dataset.set_task(task, split=MedModSplit(), num_workers=num_workers)
    return dataset, parts


def add_path_arguments(ap: argparse.ArgumentParser, required: bool = True) -> None:
    """--data-root plus optional per-folder overrides (shared by the CLIs)."""
    ap.add_argument(
        "--data-root",
        required=required,
        help="local MIMIC folder with mimiciv_medmod_slim/1.0, "
        "mimic-cxr-jpg/2.0.0 and pyhealth_cache",
    )
    ap.add_argument("--ehr-root", help="override <data-root>/mimiciv_medmod_slim/1.0")
    ap.add_argument("--cxr-root", help="override <data-root>/mimic-cxr-jpg/2.0.0")
    ap.add_argument("--cache-dir", help="override <data-root>/pyhealth_cache")
    ap.add_argument("--image-dir", help="override <cxr-root>/resized")


def paths_from_args(
    args: argparse.Namespace, use_resized_images: bool = True
) -> MedModPaths:
    return MedModPaths.from_root(
        args.data_root,
        args.ehr_root,
        args.cxr_root,
        args.cache_dir,
        args.image_dir,
        use_resized_images=use_resized_images,
    )


def add_ehr_arguments(ap: argparse.ArgumentParser) -> None:
    """--timestep and --fit-normalizer (shared by the CLIs)."""
    ap.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="bin width in hours (default 1, MedMod's released code)",
    )
    ap.add_argument(
        "--fit-normalizer",
        action="store_true",
        help="fit the normalizer on the train split instead of loading "
        "MedMod's phenotyping normalizer file",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_path_arguments(ap)
    ap.add_argument(
        "--hours-zero",
        choices=HOURS_ZERO,
        required=True,
        help='"keep" = MedFuse loader (Early Fusion), "drop" = MedMod loader (ConVIRT)',
    )
    ap.add_argument("--num-workers", type=int, default=4)
    add_ehr_arguments(ap)
    args = ap.parse_args(argv)

    paths = paths_from_args(args)
    task = make_task(paths, args.hours_zero, args.timestep, args.fit_normalizer)
    _, (train, val, test) = build_samples(paths, task, args.num_workers)
    counts = {"train": len(train), "val": len(val), "test": len(test)}
    labels = Counter(
        int(part[i]["label"].reshape(-1)[0])
        for part in (train, val, test)
        for i in range(len(part))
    )
    wrong_split = sum(
        part[i]["split"] != name
        for name, part in zip(SPLITS, (train, val, test))
        for i in range(len(part))
    )
    s = train[0]
    shape = (int(48 / task.timestep), 76)
    print(
        f"\nsamples: {sum(counts.values())} | splits: {counts} | "
        f"verified: {VERIFIED_SPLITS}"
    )
    print(
        f"ehr {tuple(s['ehr'].shape)} {s['ehr'].dtype} ({task.timestep:g} h bins) | "
        f"labels {dict(labels)} | first image file found: "
        f"{Path(s['image_path']).is_file()}"
    )
    ok = (
        counts == VERIFIED_SPLITS
        and wrong_split == 0
        and tuple(s["ehr"].shape) == shape
    )
    print(
        "OK: matches the verified cohort."
        if ok
        else "MISMATCH: expected 4,485 / 488 / 1,241 stays (results md, Settings)."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
