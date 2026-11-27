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
``MedModEHRProcessor`` as the real pipeline. So one-hot blocks, masks,
imputation and normalization behave like the real
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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ...processors.medmod_ehr_processor import (
    MedModEHRProcessor,
    MedModEHRSequenceProcessor,
)
from ...tasks.medmod_mortality_mimic4 import MedModInHospitalMortality
from ..sample_dataset import SampleDataset, create_sample_dataset
from .discretizer import PH_NORMALIZER, Discretizer
from .extraction import DEFAULT_TIMESTEP, N_HOURS, loader_rows
from .medsiglip import image_key, save_embeddings

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
    """One synthetic stay: IHM rows as MedMod's episode files hold them
    (strings, '' = missing), label, split and stay length.

    Examples:
        >>> from pyhealth.datasets.medmod.synthetic import make_synthetic_stays
        >>> stay = make_synthetic_stays(n_stays=1)[0]
        >>> len(stay.rows[0])  # Hours + 17 channels
        18
    """
    patient_id: str
    visit_id: str
    label: int
    split: str
    rows: list[list[str]]  # IHM rows: [Hours, v_1..v_17] as strings ('' = missing)
    full_stay_hours: float


def _make_rows(
    rng: np.random.Generator,
    channels: Sequence[str],
    possible: dict[str, list[str]],
    label: int,
    extreme_rate: float,
    hours: float = N_HOURS,
) -> list[list[str]]:
    n_rows = max(1, int(rng.integers(30, 90) * hours / N_HOURS))
    times = np.sort(rng.uniform(0.05, hours, size=n_rows))
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
                row[j] = f"{v:.{dec}f}" if dec else str(int(round(v)))  # noqa: RUF046
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
            hr_mean, hr_sd = _CONT["Heart Rate"][0], _CONT["Heart Rate"][1]
            row[hr] = str(int(round(rng.normal(hr_mean, hr_sd))))  # noqa: RUF046
        rows.append(row)
    return rows


def make_synthetic_stays(
    n_stays: int = 300, seed: int = 0, extreme_rate: float = 0.001
) -> list[SyntheticStay]:
    """Synthetic stays: about 1.1 stays per patient, splits per patient.

    Examples:
        >>> from pyhealth.datasets.medmod.synthetic import make_synthetic_stays
        >>> stays = make_synthetic_stays(n_stays=10, seed=0)
        >>> len(stays), sorted({s.split for s in stays}) <= ["test", "train", "val"]
        (10, True)
    """
    rng = np.random.default_rng(seed)
    disc = Discretizer()
    possible = disc._possible_values
    stays: list[SyntheticStay] = []
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
    """A grayscale JPG shaped like a resized portable chest X-ray (512 wide).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.synthetic import write_dummy_jpg
        >>> path = Path(tempfile.mkdtemp()) / "x.jpg"
        >>> write_dummy_jpg(path, np.random.default_rng(0))
        >>> Image.open(path).size[0]
        512
    """
    h = int(rng.integers(560, 640))
    yy, xx = np.mgrid[0:h, 0:512]
    base = 90 + 60 * np.exp(-(((xx - 256) / 150.0) ** 2 + ((yy - h / 2) / 220.0) ** 2))
    img = np.clip(base + rng.normal(0, 12, size=base.shape), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "L").save(path, format="JPEG")


def write_synthetic_embeddings(
    image_paths: Iterable[str | Path],
    out: str | Path,
    dim: int = 1152,
    seed: int = 0,
) -> Path:
    """Random stand-ins for MedSigLIP embeddings of synthetic images.

    Writes the same ``.npz`` format as ``python -m
    pyhealth.datasets.medmod.medsiglip``, one random vector per image file
    name, so the MedSigLIP extension (``--embeddings``) runs end to end
    without the gated model or real images. The vectors carry no information
    about the images.

    Args:
        image_paths: Paths of the synthetic images, e.g. the samples'
            ``image_path`` values. A file name that repeats gets one vector.
        out: The ``.npz`` to write.
        dim: Vector size. MedSigLIP-448's vectors have 1152 values.
        seed: Seed of the random vectors.

    Returns:
        The path of the written file.

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.medsiglip import load_embeddings
        >>> from pyhealth.datasets.medmod.synthetic import (
        ...     write_synthetic_embeddings,
        ... )
        >>> out = os.path.join(tempfile.mkdtemp(), "emb.npz")
        >>> _ = write_synthetic_embeddings(["/x/a.jpg", "/y/b.jpg"], out, dim=4)
        >>> index, emb, meta = load_embeddings(out)
        >>> index, emb.shape, meta["model"]
        ({'a': 0, 'b': 1}, (2, 4), 'synthetic (random vectors)')
    """
    keys = list(dict.fromkeys(image_key(p) for p in image_paths))
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(len(keys), dim)).astype(np.float32)
    meta = {"model": "synthetic (random vectors)", "dim": dim, "seed": seed}
    return save_embeddings(out, keys, embeddings, meta)


def make_synthetic_samples(
    hours_zero: str,
    out_dir: str | Path,
    n_stays: int = 300,
    seed: int = 0,
    extreme_rate: float = 0.001,
    timestep: float = DEFAULT_TIMESTEP,
) -> list[dict]:
    """Raw sample dicts (what the task's ``__call__`` returns: ``ehr``
    unnormalized float64), plus dummy JPGs in ``out_dir/resized/``. No PyHealth
    needed. ``hours_zero`` and ``timestep`` as in the task.

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets.medmod.synthetic import make_synthetic_samples
        >>> samples = make_synthetic_samples("keep", tempfile.mkdtemp(), n_stays=5)
        >>> samples[0]["ehr"].shape, sorted(samples[0])
        ((48, 76), ['ehr', 'image_path', 'label', 'patient_id', 'split', 'visit_id'])
    """
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
    out_dir: str | Path | None = None,
    extreme_rate: float = 0.001,
    timestep: float = DEFAULT_TIMESTEP,
    normalizer: str | Path | None = PH_NORMALIZER,
) -> SampleDataset:
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
        timestep: Bin width in hours, as in ``MedModInHospitalMortality``.
            The default is 1 hour, as in the released code.
        normalizer: As in ``MedModInHospitalMortality``. The default is the
            phenotyping normalizer file that the released code uses.

    Returns:
        The synthetic samples, with a ``split`` field for ``split_by_medmod``.

    Examples:
        >>> from pyhealth.datasets import split_by_medmod
        >>> from pyhealth.datasets.medmod.synthetic import create_synthetic_samples
        >>> samples = create_synthetic_samples("keep", n_stays=50)  # doctest: +SKIP
        >>> train, val, test = split_by_medmod(samples)  # doctest: +SKIP
    """
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


# ConVIRT pretraining (MedMod's phenotyping train/val stays, whole ICU stay).
PRETRAIN_SPLIT_FRACTIONS = {"train": 7755 / 8637, "val": 882 / 8637}


def make_synthetic_pretraining_samples(
    out_dir: str | Path,
    n_stays: int = 200,
    seed: int = 0,
    extreme_rate: float = 0.001,
    max_hours: float = 480.0,
) -> list[dict]:
    """Raw pretraining samples, as MedMod's loader builds them for the whole
    ICU stay: rows with 0 < Hours < full_stay_time (src/data/datasets.py:
    137-160), discretized up to the listfile's period_length (datasets.py:196,
    201). ``ehr`` is unnormalized float64, ``(bins, 76)`` with
    ``bins = int(period_length + 1 - 1e-6)``, so it varies per stay. Stay
    lengths are log-normal (median 60 h, 6 h to ``max_hours``). Samples have
    ``patient_id``, ``visit_id``, ``ehr``, ``image_path`` and ``split``
    (``"train"`` / ``"val"``, per patient).

    Examples:
        >>> import tempfile
        >>> from pyhealth.datasets.medmod.synthetic import (
        ...     make_synthetic_pretraining_samples,
        ... )
        >>> samples = make_synthetic_pretraining_samples(tempfile.mkdtemp(), n_stays=5)
        >>> samples[0]["ehr"].shape[1], sorted(samples[0])
        (76, ['ehr', 'image_path', 'patient_id', 'split', 'visit_id'])
    """
    disc = Discretizer()
    rng = np.random.default_rng(seed)
    img_rng = np.random.default_rng(seed + 1)
    out = []
    p = 0
    while len(out) < n_stays:
        p += 1
        names, probs = zip(*PRETRAIN_SPLIT_FRACTIONS.items())
        split = str(rng.choice(names, p=probs))
        for _ in range(1 if rng.random() < 0.9 else 2):
            if len(out) >= n_stays:
                break
            hours = float(np.clip(rng.lognormal(np.log(60.0), 0.8), 6.0, max_hours))
            period_length = round(hours, 6)  # the listfile's {:.6f}
            full_stay_hours = float(np.round(hours, 3))
            rows = _make_rows(
                rng, disc.channels, disc._possible_values, 0, extreme_rate, hours
            )
            rows, _ = loader_rows(rows, "drop", full_stay_hours)
            visit_id = f"syn-stay-{len(out) + 1:05d}"
            img = Path(out_dir) / "resized" / f"{visit_id}.jpg"
            if not img.exists():
                write_dummy_jpg(img, img_rng)
            out.append(
                {
                    "patient_id": f"syn-{p:05d}",
                    "visit_id": visit_id,
                    "ehr": disc.transform(rows, end=period_length),
                    "image_path": str(img),
                    "split": split,
                }
            )
    return out


def create_synthetic_pretraining_samples(
    n_stays: int = 200,
    seed: int = 0,
    out_dir: str | Path | None = None,
    extreme_rate: float = 0.001,
) -> SampleDataset:
    """A PyHealth ``SampleDataset`` of synthetic ConVIRT pretraining stays with
    the real pretraining set's contract: ``ehr`` of varying length,
    standardized with MedMod's phenotyping normalizer; a batch's ``ehr`` is
    ``(padded [B, max_bins, 76], lengths [B])`` (``MedModEHRSequenceProcessor``).
    Split it with ``split_by_medmod(samples)`` (train, val; test is empty).

    Examples:
        >>> from pyhealth.datasets.medmod.synthetic import (
        ...     create_synthetic_pretraining_samples,
        ... )
        >>> samples = create_synthetic_pretraining_samples(50)  # doctest: +SKIP
    """
    out_dir = (
        Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="medmod_pretrain_"))
    )
    raw = make_synthetic_pretraining_samples(out_dir, n_stays, seed, extreme_rate)
    return create_sample_dataset(
        raw,
        input_schema={
            "ehr": (MedModEHRSequenceProcessor, {"normalizer": str(PH_NORMALIZER)}),
            "image_path": "raw",
        },
        output_schema={},
        dataset_name="medmod_synthetic_pretraining",
        task_name="MedModConVIRTPretraining",
        in_memory=False,
    )
