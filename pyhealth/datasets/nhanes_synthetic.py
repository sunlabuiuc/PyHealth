"""Synthetic NHANES 2003-2006 data for testing the MotionAge pipeline.

Writes small, fake versions of the five NHANES tables that the MotionAge
pipeline uses, with the same column names and codes as the real files:

* ``PAXRAW_<s>.csv``: minute-level accelerometer counts (one row per minute);
  2005-2006 also has the per-minute step count ``PAXSTEP``.
* ``DEMO_<s>.csv``: sex, age and household income.
* ``BMX_<s>.csv``: BMI and waist circumference.
* ``BPX_<s>.csv``: up to four systolic/diastolic blood pressure readings.
* ``NHANES_<cycle>_MORT_2019_PUBLIC.dat``: linked mortality, written in the
  exact fixed-width layout of the NCHS public-use file.

``<s>`` is the NHANES cycle suffix ("C" for 2003-2004, "D" for 2005-2006).

The real NHANES examination files are SAS transport (.xpt) files. This module
writes CSV instead so it needs only numpy and pandas. The dataset loader's
``prepare_metadata`` should use these CSV files when they are already present
in ``root``, and convert the real .xpt files to the same CSV layout only when
they are not, so real and synthetic data share the rest of the pipeline. The
mortality file is written in its real fixed-width format because parsing it
is logic worth testing.

The generator also returns the ground truth that tests need: which minutes
were actually worn, and each participant's 5-year mortality label.

Example:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     data = generate_synthetic_nhanes(tmp, n_participants=5, n_days=1)
    ...     sorted(data.paths)
    ['bmx', 'bpx', 'demo', 'mortality', 'paxraw']
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
MAX_INTENSITY = 32767  # PAXINTEN hard edit in the NHANES codebook.

MAX_STEPS = 150  # Synthetic cap on PAXSTEP; real values allow up to 32767.

CYCLES: dict[str, dict[str, object]] = {
    "2003_2004": {
        "suffix": "C",
        "seqn_start": 21005,
        "sddsrvyr": 3,
        "has_steps": False,
    },
    "2005_2006": {
        "suffix": "D",
        "seqn_start": 31127,
        "sddsrvyr": 4,
        "has_steps": True,  # PAXRAW_D adds PAXSTEP; PAXRAW_C lacks it.
    },
}

PAXRAW_COLUMNS = (
    "SEQN",
    "PAXSTAT",
    "PAXCAL",
    "PAXDAY",
    "PAXN",
    "PAXHOUR",
    "PAXMINUT",
    "PAXINTEN",
)

# The first participants are hand-built edge cases, in this order, when
# ``include_edge_cases`` is True. Everyone after them is randomly generated.
EDGE_CASE_ROLES = (
    "died_within_5y",  # label 1; carries the planted wear patterns below.
    "died_after_5y",  # MORTSTAT=1 but PERMTH_INT > 60, so label 0.
    "child",  # ELIGSTAT=2: under 18, no public mortality data.
    "not_examined",  # Interviewed only: no exam, no accelerometer.
    "ineligible_missing_covariates",  # ELIGSTAT=3, refused income, no BMX/BPX.
)

# Planted patterns on the "died_within_5y" participant, day 1 (minute of day).
PLANTED_NONWEAR = (10 * 60, 13 * 60)  # 180 zero minutes: Choi should flag.
PLANTED_SPIKE = (11 * 60 + 29, 11 * 60 + 31)  # 2-minute spike inside it.
PLANTED_SEDENTARY = (14 * 60 + 30, 15 * 60 + 30)  # 60 zero minutes: worn.
# 120 zero minutes between two counts of 150, with one 1-100 spike inside:
# both Choi and the authors' rule should flag the whole block.
PLANTED_BOUNDED_NONWEAR = (16 * 60 + 30, 18 * 60 + 30)
PLANTED_LONE_SPIKE = 17 * 60 + 30  # 60 zero minutes before it, 59 after.


@dataclass
class SyntheticNHANES:
    """Paths and ground truth for one generated synthetic dataset.

    Attributes:
        out_dir: Directory the files were written to.
        paths: Maps "paxraw", "demo", "bmx", "bpx" and "mortality" to files.
        participants: One row per participant with SEQN, role, age, sex,
            eligibility, whether they have accelerometer data and the 5-year
            mortality label: 1 if MORTSTAT is 1 and PERMTH_INT (months from
            the interview) is at most 60, as in
            :class:`~pyhealth.tasks.NHANESMortalityPrediction`; <NA> when not
            eligible, or alive but followed for under 60 months.
        wear_truth: Maps SEQN to a boolean array, one entry per accelerometer
            minute, True where the device was actually worn. Minutes inside
            a planted spike count as non-wear, matching the Choi algorithm.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = generate_synthetic_nhanes(tmp, n_participants=5, n_days=1)
        ...     len(data.wear_truth[21005])
        1440
    """

    out_dir: str
    paths: dict[str, str]
    participants: pd.DataFrame
    wear_truth: dict[int, np.ndarray]


def _zero_runs(values: np.ndarray) -> list[tuple]:
    """Returns (start, end) index pairs of consecutive zeros, end exclusive."""
    is_zero = np.concatenate(([False], values == 0, [False]))
    edges = np.flatnonzero(np.diff(is_zero.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def _true_runs(mask: np.ndarray) -> list[tuple]:
    """Returns (start, end) index pairs of consecutive True values."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def _simulate_activity(
    rng: np.random.Generator,
    n_minutes: int,
    age: int,
    vitality: float,
    plant_patterns: bool,
) -> tuple:
    """Simulates minute-level intensity counts and the true wear mask.

    The device is removed overnight, sometimes for a daytime block, and
    occasionally for a whole day. While it is worn, zero-count runs are kept
    under 30 minutes so the Choi algorithm cannot mistake them for non-wear,
    and the first and last worn minutes are always nonzero so worn and
    non-worn stretches never merge into one zero run.

    Args:
        rng: Random number generator.
        n_minutes: Length of the recording in minutes.
        age: Age in years; older participants are less active.
        vitality: Latent health score; higher means more active and lower
            mortality risk. This is the planted activity-mortality signal.
        plant_patterns: Whether to add the planted day-1 test patterns.

    Returns:
        A tuple ``(intensity, wear)`` of int64 and bool arrays.
    """
    n_days = math.ceil(n_minutes / MINUTES_PER_DAY)
    wear = np.zeros(n_days * MINUTES_PER_DAY, dtype=bool)
    for day in range(n_days):
        base = day * MINUTES_PER_DAY
        if not plant_patterns and rng.random() < 0.05:
            continue  # Forgot to wear the device all day.
        wake = int(rng.integers(6 * 60, 8 * 60 + 31))
        # On the last day, go to bed by 22:30 so the final non-wear stretch
        # is at least 90 minutes long and detectable by the Choi algorithm.
        latest_bed = 22 * 60 + 30 if day == n_days - 1 else 23 * 60 + 45
        bed = int(rng.integers(22 * 60, latest_bed + 1))
        wear[base + wake : base + bed] = True
        if not plant_patterns and rng.random() < 0.3:
            length = int(rng.integers(90, 241))
            latest = bed - length - 60
            if latest > wake + 60:
                start = int(rng.integers(wake + 60, latest))
                wear[base + start : base + start + length] = False
    wear = wear[:n_minutes]

    intensity = np.zeros(n_minutes, dtype=np.int64)
    worn = np.flatnonzero(wear)
    scale = math.exp(5.3 + 0.45 * vitality - 0.012 * (age - 40))
    p_active = float(np.clip(0.35 + 0.08 * vitality, 0.1, 0.6))
    active = rng.random(worn.size) < p_active
    light = np.where(
        rng.random(worn.size) < 0.5, 0, rng.integers(1, 101, size=worn.size)
    )
    heavy = rng.lognormal(math.log(scale), 0.8, size=worn.size)
    intensity[worn] = np.clip(np.where(active, heavy, light), 0, MAX_INTENSITY)

    # Break up long zero runs inside worn stretches.
    for start, end in _true_runs(wear):
        segment = intensity[start:end]
        for z_start, z_end in _zero_runs(segment):
            for i in range(z_start + 25, z_end, 25):
                segment[i] = int(rng.integers(1, 50))

    if plant_patterns:
        nw_start, nw_end = PLANTED_NONWEAR
        wear[nw_start:nw_end] = False
        intensity[nw_start:nw_end] = 0
        sp_start, sp_end = PLANTED_SPIKE
        intensity[sp_start:sp_end] = [40, 60]
        sed_start, sed_end = PLANTED_SEDENTARY
        intensity[sed_start:sed_end] = 0  # Still worn: wear stays True.
        intensity[sed_start - 1] = intensity[sed_end] = 50
        bn_start, bn_end = PLANTED_BOUNDED_NONWEAR
        wear[bn_start:bn_end] = False
        intensity[bn_start:bn_end] = 0
        intensity[PLANTED_LONE_SPIKE] = 50
        intensity[bn_start - 1] = intensity[bn_end] = 150  # > 100: authors split

    # Nonzero first and last worn minute of every worn stretch.
    for start, end in _true_runs(wear):
        for i in (start, end - 1):
            if intensity[i] == 0:
                intensity[i] = int(rng.integers(1, 50))
    return intensity, wear


def _steps_from_intensity(intensity: np.ndarray) -> np.ndarray:
    """Maps intensity counts to per-minute step counts for PAXSTEP.

    Steps are 0 exactly when the intensity is 0, at least 1 otherwise, and
    never decrease as intensity rises, up to ``MAX_STEPS``. The mapping is
    deterministic so adding PAXSTEP does not change the random stream.
    """
    return np.minimum(np.ceil(intensity / 20), MAX_STEPS).astype(np.int64)


def _format_mortality_line(
    seqn: int,
    eligstat: int,
    mortstat: int | None,
    ucod: int | None,
    diabetes: int | None,
    hyperten: int | None,
    permth_int: int | None,
    permth_exm: int | None,
) -> str:
    """Formats one record in the NCHS public-use fixed-width layout.

    Layout (1-based, inclusive): SEQN 1-14 (left-justified), ELIGSTAT 15,
    MORTSTAT 16, UCOD_LEADING 17-19 (zero-padded), DIABETES 20, HYPERTEN 21,
    blank 22-42, PERMTH_INT 43-45, PERMTH_EXM 46-48. Missing values are "."
    (blank for UCOD_LEADING) and numbers are left-justified, as in the real
    file.
    """

    def one(value: int | None) -> str:
        return "." if value is None else str(value)

    def three(value: int | None) -> str:
        return ".".ljust(3) if value is None else str(value).ljust(3)

    ucod_text = "   " if ucod is None else f"{ucod:03d}"
    line = (
        f"{seqn:<14d}{eligstat}{one(mortstat)}{ucod_text}{one(diabetes)}"
        f"{one(hyperten)}{' ' * 21}{three(permth_int)}{three(permth_exm)}"
    )
    return line.rstrip()


def _make_participants(
    rng: np.random.Generator,
    n_participants: int,
    seqn_start: int,
    include_edge_cases: bool,
) -> pd.DataFrame:
    """Draws participant attributes, mortality outcomes and follow-up."""
    rows = []
    for i in range(n_participants):
        role = (
            EDGE_CASE_ROLES[i]
            if include_edge_cases and i < len(EDGE_CASE_ROLES)
            else "random"
        )
        if role == "random":
            if rng.random() < 0.3:
                age = int(rng.integers(6, 18))
            else:
                age = int(min(rng.integers(18, 91), 85))  # Top-coded at 85.
        else:
            age = {
                "died_within_5y": 72,
                "died_after_5y": 64,
                "child": 12,
                "not_examined": 45,
                "ineligible_missing_covariates": 38,
            }[role]
        vitality = float(rng.normal())
        if role == "died_within_5y":
            vitality = -1.5

        if role == "child" or age < 18:
            eligstat = 2
        elif role == "ineligible_missing_covariates" or (
            role == "random" and rng.random() < 0.002
        ):
            eligstat = 3
        else:
            eligstat = 1

        examined = role != "not_examined" and not (
            role == "random" and rng.random() < 0.03
        )
        has_accelerometer = (
            examined
            and role != "ineligible_missing_covariates"
            and not (role == "random" and rng.random() < 0.04)
        )

        mortstat = permth_exm = None
        if eligstat == 1:
            if role == "died_within_5y":
                died_soon, died_later = True, False
            elif role == "died_after_5y":
                died_soon, died_later = False, True
            else:
                logit = -4.2 + 0.075 * (age - 50) - 0.8 * vitality
                died_soon = rng.random() < 1.0 / (1.0 + math.exp(-logit))
                died_later = not died_soon and rng.random() < min(
                    0.9, max(0.0, (age - 30) / 60)
                )
            if died_soon:
                mortstat, months = 1, int(rng.integers(1, 61))
                if role == "died_within_5y":
                    months = 30
            elif died_later:
                mortstat, months = 1, int(rng.integers(61, 191))
                if role == "died_after_5y":
                    months = 98
            else:
                mortstat, months = 0, int(rng.integers(180, 206))
            permth_exm = months if examined else None
            permth_int = months + int(rng.integers(0, 3))
        else:
            permth_int = None

        # Same rule as NHANESMortalityPrediction: months from the interview
        # (PERMTH_INT); a survivor followed for under 60 months is unknown.
        label = pd.NA
        if eligstat == 1 and (mortstat == 1 or permth_int >= 60):
            label = int(mortstat == 1 and permth_int <= 60)

        rows.append(
            {
                "SEQN": seqn_start + i,
                "role": role,
                "age": age,
                "sex": int(rng.integers(1, 3)),
                "vitality": vitality,
                "eligstat": eligstat,
                "mortstat": mortstat,
                "permth_int": permth_int,
                "permth_exm": permth_exm,
                "examined": examined,
                "has_accelerometer": has_accelerometer,
                "died_within_5y": label,
            }
        )
    participants = pd.DataFrame(rows)
    participants["died_within_5y"] = participants["died_within_5y"].astype("Int64")
    return participants


def _blood_pressure_row(rng: np.random.Generator, age: int) -> dict:
    """Draws up to four BP readings; diastolic can be 0, as in NHANES."""
    row = {}
    n_readings = 4 if rng.random() < 0.1 else 3
    for k in range(1, 5):
        if k > n_readings:
            row[f"BPXSY{k}"] = row[f"BPXDI{k}"] = np.nan
            continue
        systolic = float(np.clip(rng.normal(105 + 0.5 * age, 15), 72, 240))
        diastolic = float(np.clip(rng.normal(70, 11), 0, 120))
        if rng.random() < 0.01:
            diastolic = 0.0
        row[f"BPXSY{k}"] = round(systolic)
        row[f"BPXDI{k}"] = round(min(diastolic, systolic - 1))
    return row


def generate_synthetic_nhanes(
    out_dir: str,
    n_participants: int = 200,
    n_days: int = 7,
    cycle: str = "2003_2004",
    seed: int = 0,
    include_edge_cases: bool = True,
) -> SyntheticNHANES:
    """Writes a synthetic NHANES dataset and returns its ground truth.

    Use 2-5 participants and 1 day for unit tests (they run in
    milliseconds), and a few hundred participants over 7 days for the
    example script. With ``include_edge_cases``, the first five participants
    are the roles in ``EDGE_CASE_ROLES``.

    Args:
        out_dir: Directory to write into; created if it does not exist.
        n_participants: Number of participants (at least 1).
        n_days: Days of accelerometer recording per participant (1-7).
        cycle: "2003_2004" or "2005_2006"; sets file names and SEQN range.
        seed: Random seed, so the same arguments give the same data.
        include_edge_cases: Whether to start with the hand-built edge cases.

    Returns:
        A ``SyntheticNHANES`` with file paths and ground truth.

    Raises:
        ValueError: If ``cycle`` is unknown, ``n_participants`` is less
            than 1, or ``n_days`` is not 1-7.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     data = generate_synthetic_nhanes(tmp, n_participants=5, n_days=1)
        ...     data.participants["died_within_5y"].tolist()
        [1, 0, <NA>, 0, <NA>]
    """
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {sorted(CYCLES)}, got {cycle!r}")
    if n_participants < 1:
        raise ValueError(f"n_participants must be at least 1, got {n_participants}")
    if not 1 <= n_days <= 7:
        raise ValueError(f"n_days must be between 1 and 7, got {n_days}")

    os.makedirs(out_dir, exist_ok=True)
    info = CYCLES[cycle]
    suffix = info["suffix"]
    rng = np.random.default_rng(seed)
    participants = _make_participants(
        rng, n_participants, int(info["seqn_start"]), include_edge_cases
    )

    pax_frames, wear_truth = [], {}
    for person in participants.itertuples():
        if not person.has_accelerometer:
            continue
        n_minutes = n_days * MINUTES_PER_DAY
        if person.role == "died_after_5y":
            n_minutes -= MINUTES_PER_DAY // 2  # Record ends at noon.
        intensity, wear = _simulate_activity(
            rng,
            n_minutes,
            person.age,
            person.vitality,
            plant_patterns=person.role == "died_within_5y",
        )
        wear_truth[person.SEQN] = wear
        minute = np.arange(n_minutes)
        first_weekday = int(rng.integers(1, 8))
        calibrated = rng.choice([1, 2, 9], p=[0.94, 0.05, 0.01])
        frame = pd.DataFrame(
            {
                "SEQN": person.SEQN,
                "PAXSTAT": 2 if rng.random() < 0.03 else 1,
                "PAXCAL": int(calibrated),
                "PAXDAY": (first_weekday - 1 + minute // MINUTES_PER_DAY) % 7 + 1,
                "PAXN": minute + 1,
                "PAXHOUR": (minute % MINUTES_PER_DAY) // 60,
                "PAXMINUT": minute % 60,
                "PAXINTEN": intensity,
            }
        )
        if info["has_steps"]:
            frame["PAXSTEP"] = _steps_from_intensity(intensity)
        pax_frames.append(frame)
    pax_columns = list(PAXRAW_COLUMNS) + (["PAXSTEP"] if info["has_steps"] else [])
    if pax_frames:
        paxraw = pd.concat(pax_frames, ignore_index=True)
    else:
        # Nobody wore the device: write the header only.
        paxraw = pd.DataFrame(columns=pax_columns)

    incomes = [*range(1, 14), 77, 99]
    income_p = np.array([2, 5, 8, 8, 8, 13, 10, 9, 6, 5, 20, 1, 0.5, 0.5, 1.0])
    demo_rows, bmx_rows, bpx_rows = [], [], []
    for person in participants.itertuples():
        income: object = int(rng.choice(incomes, p=income_p / income_p.sum()))
        if person.role == "ineligible_missing_covariates":
            income = 77
        elif rng.random() < 0.03:
            income = np.nan
        demo_rows.append(
            {
                "SEQN": person.SEQN,
                "SDDSRVYR": info["sddsrvyr"],
                "RIAGENDR": person.sex,
                "RIDAGEYR": person.age,
                "INDHHINC": income,
            }
        )
        if not person.examined or person.role == "ineligible_missing_covariates":
            continue
        bmi = float(np.clip(rng.normal(28 - 1.5 * person.vitality, 6), 12.4, 65))
        waist = float(np.clip(40 + 2.0 * bmi + rng.normal(0, 6), 32, 170))
        if person.age < 18:
            bmi, waist = bmi * 0.75, waist * 0.8
        bmx_rows.append(
            {
                "SEQN": person.SEQN,
                "BMXBMI": round(bmi, 2) if rng.random() > 0.03 else np.nan,
                "BMXWAIST": round(waist, 1) if rng.random() > 0.04 else np.nan,
            }
        )
        if person.age >= 8 and rng.random() > 0.05:
            bpx_rows.append(
                {"SEQN": person.SEQN, **_blood_pressure_row(rng, person.age)}
            )

    paths = {
        "paxraw": os.path.join(out_dir, f"PAXRAW_{suffix}.csv"),
        "demo": os.path.join(out_dir, f"DEMO_{suffix}.csv"),
        "bmx": os.path.join(out_dir, f"BMX_{suffix}.csv"),
        "bpx": os.path.join(out_dir, f"BPX_{suffix}.csv"),
        "mortality": os.path.join(out_dir, f"NHANES_{cycle}_MORT_2019_PUBLIC.dat"),
    }
    paxraw.to_csv(paths["paxraw"], index=False)
    demo = pd.DataFrame(demo_rows)
    demo["INDHHINC"] = demo["INDHHINC"].astype("Int64")
    demo.to_csv(paths["demo"], index=False)
    pd.DataFrame(bmx_rows, columns=["SEQN", "BMXBMI", "BMXWAIST"]).to_csv(
        paths["bmx"], index=False
    )
    bpx_columns = ["SEQN"] + [
        f"BPX{kind}{k}" for kind in ("SY", "DI") for k in range(1, 5)
    ]
    bpx = pd.DataFrame(bpx_rows, columns=bpx_columns)
    for column in bpx_columns[1:]:
        bpx[column] = bpx[column].astype("Int64")
    bpx.to_csv(paths["bpx"], index=False)

    def int_or_none(value: object) -> int | None:
        return None if pd.isna(value) else int(value)

    with open(paths["mortality"], "w") as handle:
        for person in participants.itertuples():
            dead = person.mortstat == 1
            line = _format_mortality_line(
                seqn=person.SEQN,
                eligstat=person.eligstat,
                mortstat=int_or_none(person.mortstat),
                ucod=int(rng.integers(1, 11)) if dead else None,
                diabetes=int(rng.random() < 0.15) if dead else None,
                hyperten=int(rng.random() < 0.2) if dead else None,
                permth_int=int_or_none(person.permth_int),
                permth_exm=int_or_none(person.permth_exm),
            )
            handle.write(line + "\n")

    return SyntheticNHANES(out_dir, paths, participants, wear_truth)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: writes a synthetic dataset to a folder.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     main([tmp, "--n-participants", "5", "--n-days", "1"])
        ... # doctest: +ELLIPSIS
        Wrote 5 participants to ...
          with accelerometer: 3
          5-year deaths: 1 of 3 labeled
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--n-participants", type=int, default=200)
    parser.add_argument("--n-days", type=int, default=7)
    parser.add_argument("--cycle", default="2003_2004", choices=sorted(CYCLES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data = generate_synthetic_nhanes(
        args.out_dir, args.n_participants, args.n_days, args.cycle, args.seed
    )
    labels = data.participants["died_within_5y"]
    print(f"Wrote {len(data.participants)} participants to {args.out_dir}")
    print(f"  with accelerometer: {len(data.wear_truth)}")
    print(
        f"  5-year deaths: {int(labels.sum())} of {int(labels.notna().sum())} labeled"
    )


if __name__ == "__main__":
    main()
