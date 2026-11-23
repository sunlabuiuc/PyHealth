"""Participant-level evaluation of MotionAge mortality risk and biological age.

Paper: Song, Yin, et al., "MotionAge: A Deep Learning Framework for Biological
Age Prediction from Wearable Activity." https://openreview.net/pdf?id=xGw1o9acWa

:class:`~pyhealth.tasks.NHANESMortalityPrediction` gives several windows per
participant, while the paper evaluates one prediction per participant. These
helpers follow the authors' code (https://github.com/jackie-jiaqi-yin/MotionAge,
MIT License):

1. :func:`aggregate_window_predictions` averages each participant's window
   probabilities (not logits) into one risk, and :func:`participant_fields`
   collects the age and sex that the task passes through unprocessed.
2. :class:`~pyhealth.models.motionage.MotionAgeMapping` turns risk into
   MotionAge, and :func:`age_acceleration` subtracts chronological age.
3. :func:`label_concordance` is the paper's C-index C(M, X) of a raw
   predictor, and :func:`evaluate_feature_sets` is the AUROC of logistic
   regressions fitted on training participants and scored on test ones, as in
   the authors' ``evaluation/secondary.py``.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "age_acceleration",
    "aggregate_window_predictions",
    "evaluate_feature_sets",
    "label_concordance",
    "participant_fields",
]


def aggregate_window_predictions(
    patient_ids: Sequence[Any], y_prob: Any, y_true: Any
) -> pd.DataFrame:
    """Averages window predictions into one risk per participant.

    The risk is the mean of the window probabilities, as in the authors'
    code, not the probability of the mean logit. The result does not depend
    on the order of the windows.

    Args:
        patient_ids: Participant ID of each window, e.g. from
            ``Trainer.inference(..., return_patient_ids=True)``.
        y_prob: Predicted probability of each window, shape ``[n]`` or
            ``[n, 1]``.
        y_true: Binary label of each window, the same shape.

    Returns:
        One row per participant, sorted by ``patient_id`` (as a string), with
        columns ``patient_id``, ``risk``, ``label`` and ``n_windows``.

    Raises:
        ValueError: If the inputs differ in length or are empty, a
            probability is outside ``[0, 1]`` or missing, or a participant's
            windows have different labels.

    Examples:
        >>> from pyhealth.metrics.motionage import aggregate_window_predictions
        >>> table = aggregate_window_predictions(
        ...     ["a", "a", "b"], [0.1, 0.9, 0.2], [1, 1, 0]
        ... )
        >>> table.to_dict("list")
        {'patient_id': ['a', 'b'], 'risk': [0.5, 0.2], 'label': [1, 0], 'n_windows': [2, 1]}
    """
    ids = np.array([str(pid) for pid in patient_ids], dtype=object)
    prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    label = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if not ids.size == prob.size == label.size:
        raise ValueError(
            f"patient_ids, y_prob and y_true must have the same length, got "
            f"{ids.size}, {prob.size} and {label.size}"
        )
    if ids.size == 0:
        raise ValueError("No window predictions to aggregate")
    if not np.all((prob >= 0.0) & (prob <= 1.0)):
        raise ValueError("y_prob must hold probabilities in [0, 1], without NaN")
    grouped = (
        pd.DataFrame({"patient_id": ids, "prob": prob, "label": label})
        .groupby("patient_id", sort=True)
        .agg(
            risk=("prob", "mean"),
            label=("label", "first"),
            n_labels=("label", "nunique"),
            n_windows=("prob", "size"),
        )
        .reset_index()
    )
    mixed = grouped.loc[grouped["n_labels"] > 1, "patient_id"].tolist()
    if mixed:
        raise ValueError(f"Windows of the same participant disagree on label: {mixed}")
    grouped["label"] = grouped["label"].astype(np.int64)
    return grouped[["patient_id", "risk", "label", "n_windows"]]


def participant_fields(
    samples: Any, fields: Sequence[str] = ("age", "sex")
) -> pd.DataFrame:
    """Collects per-participant fields that the task passes through.

    ``Trainer.inference`` returns only model outputs and patient IDs, so this
    reads unprocessed sample fields such as ``age`` and ``sex`` (from
    :class:`~pyhealth.tasks.NHANESMortalityPrediction`) to join with
    :func:`aggregate_window_predictions` on ``patient_id``.

    Args:
        samples: A :class:`~pyhealth.datasets.SampleDataset`, or any sequence
            of sample dicts with ``patient_id`` and ``fields``.
        fields: Names of the fields to collect.

    Returns:
        One row per participant, sorted by ``patient_id`` (as a string), with
        ``patient_id`` and one column per field.

    Raises:
        KeyError: If a sample lacks ``patient_id`` or one of ``fields``.
        ValueError: If a field differs between windows of a participant.

    Examples:
        >>> from pyhealth.metrics.motionage import participant_fields
        >>> samples = [
        ...     {"patient_id": "a", "age": 61.0, "sex": 2},
        ...     {"patient_id": "a", "age": 61.0, "sex": 2},
        ...     {"patient_id": "b", "age": 45.0, "sex": 1},
        ... ]
        >>> participant_fields(samples).to_dict("list")
        {'patient_id': ['a', 'b'], 'age': [61.0, 45.0], 'sex': [2, 1]}
    """
    fields = list(fields)
    rows: dict[str, dict[str, Any]] = {}
    for index in range(len(samples)):
        sample = samples[index]
        pid = str(sample["patient_id"])
        values = {field: _plain(sample[field]) for field in fields}
        seen = rows.setdefault(pid, values)
        for field in fields:
            if not _same(seen[field], values[field]):
                raise ValueError(
                    f"Field {field!r} differs between windows of participant "
                    f"{pid}: {seen[field]!r} and {values[field]!r}"
                )
    table = pd.DataFrame(
        [{"patient_id": pid, **rows[pid]} for pid in sorted(rows)],
        columns=["patient_id", *fields],
    )
    return table


def age_acceleration(motion_age: Any, age: Any) -> np.ndarray:
    """Returns MotionAge acceleration, MotionAge minus chronological age.

    Positive values mean a participant's predicted mortality risk is that of
    someone older. A NaN MotionAge, which
    :meth:`~pyhealth.models.motionage.MotionAgeMapping.transform` gives for a
    sex it was not fitted on, stays NaN.

    Args:
        motion_age: MotionAge per participant.
        age: Chronological age per participant, the same shape.

    Returns:
        The acceleration per participant, as float64.

    Raises:
        ValueError: If the shapes differ.

    Examples:
        >>> from pyhealth.metrics.motionage import age_acceleration
        >>> age_acceleration([52.0, 40.0, float("nan")], [50, 45, 60])
        array([ 2., -5., nan])
    """
    motion_age = np.asarray(motion_age, dtype=np.float64)
    age = np.asarray(age, dtype=np.float64)
    if motion_age.shape != age.shape:
        raise ValueError(
            f"motion_age and age must have the same shape, got "
            f"{motion_age.shape} and {age.shape}"
        )
    return motion_age - age


def label_concordance(label: Any, score: Any) -> float:
    """The paper's C-index C(M, X) between a binary label and a raw score.

    This is the fraction of (death, survivor) pairs in which the participant
    who died has the higher score, counting ties as one half. The paper
    (Sec. 3.2, Table 1) reports C(M, X) for X = age, MotionAge and age
    acceleration without mentioning time-to-event, so this is our reading:
    M is the binary label (death within the follow-up horizon), and C(M, X)
    equals the AUROC of the raw score X. Participants with a missing score
    are left out. For Harrell's C on follow-up time, see
    :func:`pyhealth.metrics.survival.concordance_index`.

    Args:
        label: Binary label per participant (1 = died).
        score: Score per participant; higher should mean higher risk.

    Returns:
        The concordance in ``[0, 1]``, or NaN if fewer than one death and one
        survivor have a score.

    Raises:
        ValueError: If the lengths differ or ``label`` is not 0/1.

    Examples:
        >>> from pyhealth.metrics.motionage import label_concordance
        >>> label_concordance([0, 0, 1, 1], [40, 55, 70, 55])
        0.875
    """
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if label.shape != score.shape:
        raise ValueError(
            f"label and score must have the same length, got {label.size} "
            f"and {score.size}"
        )
    if not np.isin(label, (0.0, 1.0)).all():
        raise ValueError("label must contain only 0/1 values")
    keep = np.isfinite(score)
    label, score = label[keep], score[keep]
    if np.unique(label).size < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def evaluate_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    label: str = "label",
    baseline: str | None = None,
) -> pd.DataFrame:
    """AUROC of logistic regressions on participant-level feature sets.

    For each feature set, fits ``LogisticRegression(max_iter=1000)`` (lbfgs,
    no feature scaling) on the training participants and scores the test
    participants, as in the authors' ``evaluation/secondary.py``. Rows with a
    missing feature or label are dropped, separately for each feature set.

    Args:
        train: Training participants, one row each.
        test: Test participants, with the same columns.
        feature_sets: Maps a name to the columns it uses, e.g.
            ``{"age": ["age", "sex"], "motionage": ["motion_age", "sex"]}``.
        label: Name of the binary label column.
        baseline: Optional feature set name; adds ``delta_auroc``, each
            set's test AUROC minus the baseline's.

    Returns:
        One row per feature set with ``feature_set``, ``features``,
        ``n_train``, ``n_test``, ``test_events``, ``test_auroc`` and
        ``test_auprc`` (and ``delta_auroc`` with ``baseline``). The test
        metrics are NaN, with a warning, if the test set has one class.

    Raises:
        KeyError: If a column is missing, or ``baseline`` is not one of
            ``feature_sets``.
        ValueError: If ``feature_sets`` is empty or a set's training rows do
            not have both classes.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.metrics.motionage import evaluate_feature_sets
        >>> train = pd.DataFrame({"age": [30, 40, 50, 60, 70, 80],
        ...                       "label": [0, 0, 0, 1, 0, 1]})
        >>> test = pd.DataFrame({"age": [35, 75], "label": [0, 1]})
        >>> table = evaluate_feature_sets(train, test, {"age": ["age"]})
        >>> float(table.loc[0, "test_auroc"])
        1.0
    """
    if not feature_sets:
        raise ValueError("feature_sets must name at least one feature set")
    if baseline is not None and baseline not in feature_sets:
        raise KeyError(f"baseline {baseline!r} is not one of {list(feature_sets)}")
    rows = []
    for name, columns in feature_sets.items():
        columns = list(dict.fromkeys(columns))
        needed = [*columns, label]
        for part, frame in (("train", train), ("test", test)):
            missing = [column for column in needed if column not in frame.columns]
            if missing:
                raise KeyError(f"{part} is missing columns {missing} for {name!r}")
        fit = train[needed].dropna()
        held_out = test[needed].dropna()
        if fit[label].nunique() < 2:
            raise ValueError(
                f"Feature set {name!r} needs deaths and survivors in train"
            )
        model = LogisticRegression(max_iter=1000)
        model.fit(fit[columns].to_numpy(np.float64), fit[label].to_numpy(np.int64))
        y_test = held_out[label].to_numpy(np.int64)
        auroc = auprc = float("nan")
        if np.unique(y_test).size == 2:
            prob = model.predict_proba(held_out[columns].to_numpy(np.float64))[:, 1]
            auroc = float(roc_auc_score(y_test, prob))
            auprc = float(average_precision_score(y_test, prob))
        else:
            warnings.warn(
                f"Test set for {name!r} has one class; AUROC is undefined",
                stacklevel=2,
            )
        rows.append(
            {
                "feature_set": name,
                "features": ",".join(columns),
                "n_train": len(fit),
                "n_test": len(held_out),
                "test_events": int(y_test.sum()),
                "test_auroc": auroc,
                "test_auprc": auprc,
            }
        )
    table = pd.DataFrame(rows)
    if baseline is not None:
        reference = table.loc[table["feature_set"] == baseline, "test_auroc"].iloc[0]
        table["delta_auroc"] = table["test_auroc"] - reference
    return table


def _plain(value: Any) -> Any:
    """Turns a 0-d tensor or numpy scalar into a Python scalar."""
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        return value.item()
    return value


def _same(a: Any, b: Any) -> bool:
    """Equality that treats two NaNs as equal."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return a == b
