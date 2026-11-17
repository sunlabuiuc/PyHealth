"""Spectral baseline for seizure onset zone localization on HUP iEEG.

Author: Axel Coiffard-Donaty

Classifies each intracranial contact as inside or outside the clinically
determined seizure onset zone from its interictal spectrum, with a
validation by patient:

1. ``HUPiEEGDataset`` and ``SOZLocalizationHUP`` cut the interictal
   recordings into 30 second single-contact windows (average reference,
   0.5-80 Hz, 60 Hz notch, 200 Hz);
2. each window is described by its relative power in the delta (0.5-4 Hz),
   theta (4-8), alpha (8-13), beta (13-30) and gamma (30-80) bands, from
   Welch's method with 2 second Hamming segments and 1 second of overlap,
   relative to 0.5-80 Hz;
3. the windows of a contact are summarized by their median, so that a
   contact counts once, and each feature is standardized within the patient
   (no label is used);
4. a random forest is evaluated with 10-fold cross-validation by patient:
   all the contacts of a patient are in the same fold.

The spectral estimate, the bands, the random forest and the 10 folds by
patient follow Bernabei, Sinha et al. (Brain 145(6):1949-1961, 2022,
https://doi.org/10.1093/brain/awab480) and their code
(https://github.com/jbernabei/iEEG_atlas). The scores are not comparable
with theirs: they used a bipolar montage, expressed each contact relative to
a normative atlas of the same brain region, added connectivity features, and
separated resected seizure onset contacts from contacts judged normal.
Choices of this example: the median over windows, the standardization within
the patient, and scikit-learn's default random forest.

Reported: the area under the ROC curve over all held-out contacts, as in the
paper, with a 95% percentile bootstrap interval that resamples patients
(their held-out scores, without refitting the model); and
the area under the curve inside each patient (median and quartiles), which
is the one that matters for localizing the zone in a new patient.

On version 1.1.3 (6119 contacts of 56 patients, 538 in the seizure onset
zone) this gives an area of 0.594 over all contacts (interval 0.551 to
0.636) and a median of 0.592 inside a patient (quartiles 0.485 and 0.718):
above chance, far from a usable localization. With ``--target resect
--seizure-free`` (3667 contacts of 36 patients, 456 resected) the areas are
0.601 (0.558 to 0.650) and 0.669 (0.569 and 0.750). These are starting
points to compare models against, not results to match.

Data: open under CC0. Download https://openneuro.org/datasets/ds004100
(14.2 GB, or 5.4 GB for the interictal recordings), then run::

    python examples/eeg/hup_ieeg/hup_soz_localization.py --root /path/to/ds004100

``--target resect`` labels the contacts of the resected or ablated tissue,
and ``--seizure-free`` keeps the patients who became seizure free. The task
cache takes several gigabytes for the whole dataset.
"""

import argparse
from collections import defaultdict

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pyhealth.datasets import HUPiEEGDataset
from pyhealth.tasks import SOZLocalizationHUP
from pyhealth.tasks.soz_localization_hup import HUP_TARGETS

SEED = 0
SFREQ = 200  # Hz, the output rate of the task
WELCH_SECONDS = 2
WELCH_OVERLAP_SECONDS = 1
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
N_FOLDS = 10
N_BOOTSTRAP = 2000


def relative_band_power(signal: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Relative power of each band of ``BANDS`` in one window.

    Args:
        signal: One window of one contact, shaped ``(samples,)`` or
            ``(1, samples)``.
        sfreq: Sampling rate in Hz.

    Returns:
        Array of shape ``(5,)``: the share of the 0.5-80 Hz power in each
        band.
    """
    freqs, power = welch(
        np.asarray(signal).ravel(),
        fs=sfreq,
        window="hamming",
        nperseg=int(sfreq * WELCH_SECONDS),
        noverlap=int(sfreq * WELCH_OVERLAP_SECONDS),
    )
    total = power[(freqs >= 0.5) & (freqs < 80.0)].sum()
    if total == 0:  # flat window: no power to share out
        return np.zeros(len(BANDS))
    return np.array(
        [
            power[(freqs >= low) & (freqs < high)].sum() / total
            for low, high in BANDS.values()
        ]
    )


def contact_table(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize the windows of each contact and standardize by patient.

    Args:
        samples: Iterable of task samples with ``patient_id``, ``channel``,
            ``signal`` and ``label``.

    Returns:
        ``(features, labels, patients)`` with one row per contact:
        ``features`` is ``(contacts, 5)``, the median over the windows of
        the contact of each relative band power, standardized over the
        contacts of the patient; ``labels`` is 0 or 1; ``patients`` holds
        the patient of each contact.
    """
    windows = defaultdict(list)
    labels = {}
    for sample in samples:
        key = (sample["patient_id"], sample["channel"])
        windows[key].append(relative_band_power(sample["signal"]))
        labels[key] = int(sample["label"])
    keys = sorted(windows)
    features = np.stack([np.median(windows[key], axis=0) for key in keys])
    patients = np.array([patient for patient, _ in keys])
    for patient in np.unique(patients):
        rows = patients == patient
        spread = features[rows].std(axis=0)
        spread[spread == 0] = 1.0
        features[rows] = (features[rows] - features[rows].mean(axis=0)) / spread
    return features, np.array([labels[key] for key in keys]), patients


def cross_validate(
    features: np.ndarray, labels: np.ndarray, patients: np.ndarray
) -> np.ndarray:
    """Score every contact with a model that has not seen its patient.

    Args:
        features: Array ``(contacts, n_features)``.
        labels: Array ``(contacts,)`` of 0 and 1.
        patients: Array ``(contacts,)`` with the patient of each contact.

    Returns:
        Array ``(contacts,)``: the predicted probability of the positive
        class, from ``N_FOLDS``-fold cross-validation by patient (one fold
        per patient when there are fewer patients).

    Raises:
        ValueError: If the contacts come from a single patient.
    """
    n_patients = len(np.unique(patients))
    if n_patients < 2:
        raise ValueError(
            f"Cross-validation by patient needs at least 2 patients, got {n_patients}."
        )
    scores = np.zeros(len(labels))
    folds = GroupKFold(n_splits=min(N_FOLDS, n_patients))
    for train, test in folds.split(features, labels, groups=patients):
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        model.fit(features[train], labels[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return scores


def summarize(
    scores: np.ndarray, labels: np.ndarray, patients: np.ndarray
) -> dict[str, float]:
    """Area under the ROC curve, over all contacts and inside each patient.

    Args:
        scores: Array ``(contacts,)`` of held-out scores.
        labels: Array ``(contacts,)`` of 0 and 1.
        patients: Array ``(contacts,)`` with the patient of each contact.

    Returns:
        ``auc``: over all contacts, with ``auc_low`` and ``auc_high``, the
        95% percentile bootstrap interval from ``N_BOOTSTRAP`` resamplings
        of the patients; ``patient_auc_q1``, ``patient_auc_median`` and
        ``patient_auc_q3``: quartiles of the area computed inside each
        patient that has contacts of both classes.
    """
    names = np.unique(patients)
    rows = {name: np.where(patients == name)[0] for name in names}
    # The area is undefined for a patient whose contacts all share a label.
    inside = [
        roc_auc_score(labels[rows[name]], scores[rows[name]])
        for name in names
        if 0 < labels[rows[name]].sum() < len(rows[name])
    ]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_BOOTSTRAP):
        chosen = np.concatenate([rows[name] for name in rng.choice(names, len(names))])
        draws.append(roc_auc_score(labels[chosen], scores[chosen]))
    low, high = np.percentile(draws, [2.5, 97.5])
    q1, median, q3 = np.percentile(inside, [25, 50, 75]) if inside else [np.nan] * 3
    return {
        "auc": roc_auc_score(labels, scores),
        "auc_low": low,
        "auc_high": high,
        "patient_auc_q1": q1,
        "patient_auc_median": median,
        "patient_auc_q3": q3,
    }


def main() -> None:
    """Run the spectral baseline and print its cross-validated scores."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", required=True, help="BIDS root, the folder with participants.tsv."
    )
    parser.add_argument("--target", default="soz", choices=HUP_TARGETS)
    parser.add_argument(
        "--seizure-free",
        action="store_true",
        help="Keep only the patients who became seizure free after surgery.",
    )
    args = parser.parse_args()

    dataset = HUPiEEGDataset(root=args.root)
    dataset.stats()
    task = SOZLocalizationHUP(
        target=args.target, outcomes=("S",) if args.seizure_free else None
    )
    samples = dataset.set_task(task)

    features, labels, patients = contact_table(samples)
    print(
        f"{len(labels)} contacts of {len(np.unique(patients))} patients, "
        f"{int(labels.sum())} marked {args.target}"
    )
    scores = cross_validate(features, labels, patients)
    result = summarize(scores, labels, patients)
    print({key: round(float(value), 3) for key, value in result.items()})


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()
