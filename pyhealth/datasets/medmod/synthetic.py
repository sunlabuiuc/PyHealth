"""Synthetic MedMod IHM samples with the exact sample contract, no MIMIC needed.

For writing and testing model code without MIMIC access. Works like PyHealth's
``create_sample_dataset``; everything after it is the same as on real data::

    from pyhealth.datasets import get_dataloader
    from pyhealth.datasets import split_by_medmod
    from pyhealth.datasets.medmod.synthetic import create_synthetic_samples

    samples = create_synthetic_samples(hours_zero="keep", n_stays=300)
    # real data instead: samples = dataset.set_task(MedModInHospitalMortality(...))
    train, val, test = split_by_medmod(samples)
    train_loader = get_dataloader(train, batch_size=32, shuffle=True)

Sample contract (identical to ``MedModInHospitalMortality``):

    patient_id  str           synthetic, prefixed "syn-" (never a MIMIC id)
    visit_id    str           synthetic stay id, "syn-stay-…"
    ehr         float32 48x76 MedMod layout: 12 z-scored continuous columns,
                              47 one-hot columns, 17 masks, in discretizer_config order
                              (task defaults; ``timestep=2.0`` gives 24x76)
    image_path  str           a dummy grayscale JPG, 512 px wide like resized/
    label       0/1           (a float tensor [1] after PyHealth's binary processor)
    split       "train" | "val" | "test"

How the EHR is made: synthetic hourly rows (plausible ICU values, GCS text
categories, random gaps, sometimes a row at Hours == 0) go through the SAME
loader (``medmod.extraction.loader_rows``), MedMod discretizer and
``MedModEHRProcessor`` as the real pipeline. So one-hot blocks, masks, imputation and normalization behave like the real
data. The values are made up and carry no clinical signal, except
a weak link between label and a few vitals so a model has something to learn.
Real data also has occasional extreme values (MedMod does not clip outliers);
``extreme_rate`` adds a few to exercise that path.

Split proportions follow the real cohort (4,485 / 488 / 1,241) and are assigned
per patient, like MedMod's subject lists. About 15% of stays are deaths, like the
real 14.9%.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .discretizer import PH_NORMALIZER, Discretizer
from .extraction import DEFAULT_TIMESTEP, N_HOURS, loader_rows

SPLIT_FRACTIONS = {"train": 4485 / 6214, "val": 488 / 6214, "test": 1241 / 6214}
DEATH_RATE = 924 / 6214

# Continuous channels: (mean, sd, decimals, hours between measurements).
# Plausible adult ICU ranges in the units the pipeline produces after cleaning
# (FiO2 as a fraction, temperature in Celsius).
_CONT = {
    "Diastolic blood pressure": (60.0, 12.0, 0, 1),
    "Fraction inspired oxygen": (0.45, 0.15, 2, 6),
    "Glucose": (135.0, 40.0, 0, 6),
    "Heart Rate": (88.0, 16.0, 0, 1),
    "Height": (170.0, 10.0, 0, 48),
    "Mean blood pressure": (78.0, 12.0, 0, 1),
    "Oxygen saturation": (96.5, 2.5, 0, 1),
    "Respiratory rate": (19.0, 5.0, 0, 1),
    "Systolic blood pressure": (120.0, 20.0, 0, 1),
    "Temperature": (36.9, 0.6, 1, 4),
    "Weight": (82.0, 18.0, 1, 48),
    "pH": (7.38, 0.06, 2, 6),
}
_CAT_EVERY = {  # categorical channels: hours between measurements
    "Capillary refill rate": 12,
    "Glascow coma scale eye opening": 4,
    "Glascow coma scale motor response": 4,
    "Glascow coma scale total": 8,
    "Glascow coma scale verbal response": 4,
}


@dataclass
class SyntheticStay:
    patient_id: str
    visit_id: str
    label: int
    split: str
    rows: List[List[str]]  # IHM rows: [Hours, v_1..v_17] as strings ('' = missing)
    full_stay_hours: float


def _make_rows(
    rng: np.random.Generator,
    channels: Sequence[str],
    possible: Dict[str, List[str]],
    label: int,
    extreme_rate: float,
) -> List[List[str]]:
    times = np.sort(rng.uniform(0.05, N_HOURS, size=rng.integers(30, 90)))
    if rng.random() < 0.3:
        times = np.concatenate([[0.0], times])  # a row exactly at Hours == 0
    rows = []
    shift = (
        {
            "Heart Rate": 12.0,
            "Respiratory rate": 4.0,
            "Oxygen saturation": -2.5,
            "Systolic blood pressure": -12.0,
        }
        if label
        else {}
    )
    for t in times:
        row = [repr(float(t))] + [""] * len(channels)
        for j, ch in enumerate(channels, start=1):
            if ch in _CONT:
                mean, sd, dec, every = _CONT[ch]
                if rng.random() > 1.0 / every:
                    continue
                v = rng.normal(mean + shift.get(ch, 0.0), sd)
                if rng.random() < extreme_rate:
                    v = mean * rng.choice(
                        [50.0, 1000.0]
                    )  # an absurd value, never clipped
                row[j] = f"{v:.{dec}f}" if dec else str(int(round(v)))
            else:
                if rng.random() > 1.0 / _CAT_EVERY[ch]:
                    continue
                row[j] = str(rng.choice(possible[ch]))
        if all(v == "" for v in row[1:]):
            # Real rows always hold >= 1 value: MedMod builds them by pivoting
            # events per charttime (bench/subject.py:56-68). Keep the Hours == 0
            # row (so drop/keep differ) by giving it a heart rate; skip others.
            if t != 0.0:
                continue
            hr = channels.index("Heart Rate") + 1
            row[hr] = str(
                int(round(rng.normal(_CONT["Heart Rate"][0], _CONT["Heart Rate"][1])))
            )
        rows.append(row)
    return rows


def make_synthetic_stays(
    n_stays: int = 300, seed: int = 0, extreme_rate: float = 0.001
) -> List[SyntheticStay]:
    """Synthetic stays: about 1.1 stays per patient, splits per patient."""
    rng = np.random.default_rng(seed)
    disc = Discretizer()
    possible = disc._possible_values
    stays: List[SyntheticStay] = []
    p = 0
    while len(stays) < n_stays:
        p += 1
        split = rng.choice(list(SPLIT_FRACTIONS), p=list(SPLIT_FRACTIONS.values()))
        for _ in range(1 if rng.random() < 0.9 else 2):
            if len(stays) >= n_stays:
                break
            label = int(rng.random() < DEATH_RATE)
            stays.append(
                SyntheticStay(
                    patient_id=f"syn-{p:05d}",
                    visit_id=f"syn-stay-{len(stays) + 1:05d}",
                    label=label,
                    split=str(split),
                    rows=_make_rows(rng, disc.channels, possible, label, extreme_rate),
                    full_stay_hours=float(np.round(rng.uniform(N_HOURS, 240.0), 3)),
                )
            )
    # PyHealth's binary label processor needs both classes; so do most metrics.
    for split in ("train", "val", "test"):
        idx = [i for i, s in enumerate(stays) if s.split == split]
        if idx and len({stays[i].label for i in idx}) == 1:
            stays[idx[0]].label = 1 - stays[idx[0]].label
    return stays


def write_dummy_jpg(path: Path, rng: np.random.Generator) -> None:
    """A grayscale JPG shaped like a resized portable chest X-ray (512 wide)."""
    h = int(rng.integers(560, 640))
    yy, xx = np.mgrid[0:h, 0:512]
    base = 90 + 60 * np.exp(-(((xx - 256) / 150.0) ** 2 + ((yy - h / 2) / 220.0) ** 2))
    img = np.clip(base + rng.normal(0, 12, size=base.shape), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "L").save(path, format="JPEG")


def make_synthetic_samples(
    hours_zero: str,
    out_dir: str | Path,
    n_stays: int = 300,
    seed: int = 0,
    extreme_rate: float = 0.001,
    timestep: float = DEFAULT_TIMESTEP,
) -> List[Dict]:
    """Raw sample dicts (what the task's ``__call__`` returns: ``ehr``
    unnormalized float64), plus dummy JPGs in ``out_dir/resized/``. No PyHealth
    needed. ``hours_zero`` and ``timestep`` as in the task."""
    if hours_zero not in ("drop", "keep"):
        raise ValueError('hours_zero must be "drop" or "keep"')
    disc = Discretizer(timestep=timestep)
    rng = np.random.default_rng(seed + 1)
    out = []
    for s in make_synthetic_stays(n_stays, seed, extreme_rate):
        rows, _ = loader_rows(s.rows, hours_zero, s.full_stay_hours)
        ehr = disc.transform(rows, end=float(N_HOURS))
        img = Path(out_dir) / "resized" / f"{s.visit_id}.jpg"
        if not img.exists():
            write_dummy_jpg(img, rng)
        out.append(
            {
                "patient_id": s.patient_id,
                "visit_id": s.visit_id,
                "ehr": ehr,
                "image_path": str(img),
                "label": s.label,
                "split": s.split,
            }
        )
    return out


def create_synthetic_samples(
    hours_zero: str,
    n_stays: int = 300,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    extreme_rate: float = 0.001,
    timestep: float = DEFAULT_TIMESTEP,
    normalizer: Optional[str | Path] = PH_NORMALIZER,
):
    """A PyHealth ``SampleDataset`` of synthetic stays, built with the task's own
    schema and processor, so ``samples[i]`` has the same keys, types and shapes
    as the real set. Split it with ``split_by_medmod(samples)``.

    Args:
        hours_zero: "keep" (Early Fusion) or "drop" (ConVIRT), as in the task.
        n_stays: number of stays; use 300+ so val (~8%) isn't tiny.
        seed: makes the data reproducible.
        out_dir: where the dummy JPGs go (``out_dir/resized``); a temp folder
            if not given.
        extreme_rate: share of values replaced by an absurd one (no clipping).
        timestep, normalizer: as in ``MedModInHospitalMortality``; by default the
            released code's (1-hour bins, phenotyping normalizer file).

    Examples:
        >>> from pyhealth.datasets import split_by_medmod
        >>> from pyhealth.datasets.medmod.synthetic import create_synthetic_samples
        >>> samples = create_synthetic_samples("keep", n_stays=50)  # doctest: +SKIP
        >>> train, val, test = split_by_medmod(samples)  # doctest: +SKIP
    """
    from pyhealth.datasets import create_sample_dataset

    from ...processors.medmod_ehr_processor import MedModEHRProcessor
    from ...tasks.medmod_mortality_mimic4 import MedModInHospitalMortality

    task = MedModInHospitalMortality(
        hours_zero=hours_zero, timestep=timestep, normalizer=normalizer
    )

    out_dir = (
        Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="medmod_synthetic_"))
    )
    raw = make_synthetic_samples(
        hours_zero, out_dir, n_stays, seed, extreme_rate, timestep
    )
    # create_sample_dataset has no split= (set_task does), so with
    # normalizer=None the statistics are fitted on the train split here.
    ehr = MedModEHRProcessor(normalizer)
    ehr.fit([s for s in raw if s["split"] == "train"], "ehr")
    return create_sample_dataset(
        raw,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        input_processors={"ehr": ehr},
        dataset_name="medmod_synthetic",
        task_name="MedModInHospitalMortality",
        in_memory=False,
    )
