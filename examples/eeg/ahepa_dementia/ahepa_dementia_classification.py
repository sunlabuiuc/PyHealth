"""Band-power baseline on the AHEPA dementia EEG dataset, by subject.

Author: Axel Coiffard-Donaty

Follows the protocol of the classification benchmark of the data descriptor
(Miltiadous et al., Data 8(6):95, 2023, section 3.3,
https://doi.org/10.3390/data8060095):

1. ``AHEPADementiaDataset`` and ``DementiaClassificationAHEPA`` cut the
   artifact-cleaned recordings into 4 second windows with 50% overlap;
2. each window is described by its relative band power in the delta
   (0.5-4 Hz), theta (4-8), alpha (8-13), beta (13-25) and gamma (25-45)
   bands, from Welch's method, relative to 0.5-45 Hz;
3. a k-nearest-neighbour classifier (k = 3) and a random forest are
   evaluated with leave-one-subject-out validation: all the windows of one
   subject are the test set, the others the training set, for every subject
   in turn. Accuracy, sensitivity, specificity and F1 come from the pooled
   confusion matrix, as in the paper.

Every window of a subject carries that subject's diagnosis, so the split has
to be by subject; a split by window puts the same person in training and
test data and inflates the scores.

Choices the paper does not specify: the five relative powers are averaged
over the 19 channels (the paper reports 5 features per window; use
``--per-channel`` for the 95 per-channel features), Welch segments last 2
seconds, the random forest uses scikit-learn's defaults, and features are
not scaled. The paper also evaluates LightGBM, an SVM and a multilayer
perceptron, left out here. The recordings are resampled from 500 to 125 Hz,
above twice the 45 Hz upper edge of the bands, to keep the windows small.

``--reference average`` subtracts the mean over channels, which removes the
signal that the mastoid reference adds to every channel (see the task's
documentation).

Data: open under CC0. Download https://openneuro.org/datasets/ds004504
(5.8 GB, or about 3 GB for the ``derivatives`` folder and
``participants.tsv`` alone), then run::

    python examples/eeg/ahepa_dementia/ahepa_dementia_classification.py \\
        --root /path/to/ds004504

Other problems: ``--problem ftd_vs_cn``, ``ad_vs_ftd``, ``dementia_vs_cn``.
"""

import argparse
from collections.abc import Callable

import numpy as np
from scipy.signal import welch
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from pyhealth.datasets import AHEPADementiaDataset
from pyhealth.tasks import DementiaClassificationAHEPA
from pyhealth.tasks.dementia_classification_ahepa import AHEPA_REFERENCES

SEED = 0
RESAMPLE_RATE = 125  # Hz, above twice the 45 Hz upper edge of the bands
WELCH_SECONDS = 2
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 25.0),
    "gamma": (25.0, 45.0),
}
BINARY_PROBLEMS = ("ad_vs_cn", "ftd_vs_cn", "ad_vs_ftd", "dementia_vs_cn")


def relative_band_power(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Relative power of each band, per channel.

    Args:
        signal: One window, shaped ``(channels, samples)``.
        sfreq: Sampling rate in Hz.

    Returns:
        Array of shape ``(channels, 5)``: the share of the 0.5-45 Hz power in
        each band of ``BANDS``.
    """
    freqs, power = welch(signal, fs=sfreq, nperseg=int(sfreq * WELCH_SECONDS))
    total = power[:, (freqs >= 0.5) & (freqs < 45.0)].sum(axis=1)
    bands = [
        power[:, (freqs >= low) & (freqs < high)].sum(axis=1) / total
        for low, high in BANDS.values()
    ]
    return np.stack(bands, axis=1)


def leave_one_subject_out(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    make_model: Callable[[], ClassifierMixin],
) -> dict[str, float]:
    """Evaluate a classifier with leave-one-subject-out validation.

    Args:
        features: Array ``(windows, n_features)``.
        labels: Array ``(windows,)`` of 0 and 1.
        subjects: Array ``(windows,)`` with the subject of each window.
        make_model: Function returning a new scikit-learn classifier.

    Returns:
        Accuracy, sensitivity, specificity and F1 from the confusion matrix
        pooled over the left-out subjects, and the share of subjects whose
        majority of windows is classified correctly.
    """
    predictions = np.zeros_like(labels)
    for subject in np.unique(subjects):
        test = subjects == subject
        model = make_model().fit(features[~test], labels[~test])
        predictions[test] = model.predict(features[test])
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    correct = predictions == labels
    by_subject = [correct[subjects == s].mean() > 0.5 for s in np.unique(subjects)]
    return {
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "subjects_correct": float(np.mean(by_subject)),
    }


def main() -> None:
    """Run the band-power baseline and print its leave-one-subject-out scores."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", required=True, help="BIDS root, the folder with participants.tsv."
    )
    parser.add_argument("--problem", default="ad_vs_cn", choices=BINARY_PROBLEMS)
    parser.add_argument("--reference", default="stored", choices=AHEPA_REFERENCES)
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use the 5 band powers of each channel (95 features) instead of "
        "their average over channels (5 features).",
    )
    args = parser.parse_args()

    dataset = AHEPADementiaDataset(root=args.root)
    dataset.stats()
    task = DementiaClassificationAHEPA(
        problem=args.problem,
        resample_rate=RESAMPLE_RATE,
        reference=args.reference,
    )
    samples = dataset.set_task(task)

    features, labels, subjects = [], [], []
    for sample in samples:
        power = relative_band_power(np.asarray(sample["signal"]), RESAMPLE_RATE)
        features.append(power.ravel() if args.per_channel else power.mean(axis=0))
        labels.append(int(sample["label"]))
        subjects.append(sample["patient_id"])
    features = np.stack(features)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects)
    print(
        f"{len(labels)} windows of {len(np.unique(subjects))} subjects, "
        f"{features.shape[1]} features, {int(labels.sum())} positive windows"
    )

    models = {
        "kNN (k=3)": lambda: KNeighborsClassifier(n_neighbors=3),
        "random forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    }
    for name, make_model in models.items():
        scores = leave_one_subject_out(features, labels, subjects, make_model)
        print(name, {key: round(value, 4) for key, value in scores.items()})


# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    main()
