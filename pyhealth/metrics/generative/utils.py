"""Shared utilities for synthetic-EHR generative evaluation metrics.

This module contains the data-preparation helpers, distance functions, and
the lightweight predictive models (an LSTM classifier and a random-forest
baseline) that the privacy and utility metrics build on. It is not intended
to be used directly; see :mod:`pyhealth.metrics.generative.privacy` and
:mod:`pyhealth.metrics.generative.utility` for the public metric functions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = [
    "summarize_metric_runs",
    "convert_visits_to_sets",
    "calculate_hamming_distance_cutoff",
    "find_nearest_neighbor_dist",
    "process_patient_data_for_lstm",
    "collate_fn",
    "EHRDataset",
    "EHR_LSTM_Classifier",
    "train_lstm_model",
    "aggregate_patient_visits",
    "train_sklearn_model",
    "build_next_visit_prediction_dataset",
    "convert_cols_to_multihot",
]


def summarize_metric_runs(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Summarizes a list of per-run metric dicts into (mean, std) tuples.

    Args:
        metrics_list: List of dicts, one per run, mapping metric name to value.

    Returns:
        Dictionary mapping each metric name to a ``(mean, std)`` tuple computed
            across the runs. Returns an empty dict if ``metrics_list`` is empty.
    """
    if not metrics_list:
        return {}
    summary: Dict[str, Tuple[float, float]] = {}
    for key in metrics_list[0].keys():
        values = [run[key] for run in metrics_list if key in run]
        summary[key] = (float(np.mean(values)), float(np.std(values)))
    return summary


# --- Privacy distance helpers ---------------------------------------------


def convert_visits_to_sets(
    df: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
) -> List[List[set]]:
    """Converts a flat EHR dataframe into per-patient lists of code sets.

    Each patient becomes a list of visits, and each visit is a ``set`` of the
    codes recorded at that timestep.

    Args:
        df: Input dataframe with one row per (patient, visit, code) event.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.

    Returns:
        List of patients, where each patient is a list of code sets.
    """
    records = (
        df.groupby(subject_col)[[visit_col, code_col]]
        .apply(lambda x: x.groupby(visit_col)[code_col].apply(set).tolist())
        .tolist()
    )
    return records


def calculate_hamming_distance_cutoff(
    v1: List[set], v2: List[set], cutoff: float
) -> float:
    """Computes a set-based Hamming distance between two patients, with cutoff.

    The distance accumulates the symmetric-difference size of aligned visits
    plus a penalty for differing sequence lengths. Computation stops early once
    the running distance reaches ``cutoff``.

    Args:
        v1: First patient as a list of code sets.
        v2: Second patient as a list of code sets.
        cutoff: Distance value at which to stop early.

    Returns:
        The distance between ``v1`` and ``v2``, capped at ``cutoff``.
    """
    len1, len2 = len(v1), len(v2)
    dist = 0 if len1 == len2 else 1
    if dist >= cutoff:
        return cutoff

    min_len = min(len1, len2)
    for i in range(min_len):
        dist += len(v1[i] ^ v2[i])
        if dist >= cutoff:
            return cutoff

    if len1 > min_len:
        dist += sum(len(v) for v in v1[min_len:])
    elif len2 > min_len:
        dist += sum(len(v) for v in v2[min_len:])
    return dist


def find_nearest_neighbor_dist(
    query: List[set],
    reference_dataset: List[List[set]],
    skip_index: Optional[int] = None,
) -> float:
    """Finds the distance from a query patient to its nearest neighbor.

    Args:
        query: Query patient as a list of code sets.
        reference_dataset: Patients to search over.
        skip_index: Optional index in ``reference_dataset`` to skip. Use this
            when ``query`` is itself a member of ``reference_dataset`` (i.e. a
            within-set nearest-neighbor search) so the patient does not match
            itself at distance 0. Genuine duplicates at other indices can still
            legitimately produce a distance of 0.

    Returns:
        The smallest :func:`calculate_hamming_distance_cutoff` distance between
            ``query`` and any patient in ``reference_dataset`` (excluding
            ``skip_index`` when provided).
    """
    best = float("inf")
    for i, ref in enumerate(reference_dataset):
        if i == skip_index:
            continue
        d = calculate_hamming_distance_cutoff(query, ref, best)
        if d == 0:
            return 0
        if d < best:
            best = d
    return best


# --- LSTM classifier -------------------------------------------------------


def process_patient_data_for_lstm(
    df: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    code_to_idx: Optional[Dict] = None,
) -> Tuple[List[Tuple[torch.Tensor, int]], Dict]:
    """Transforms a flat EHR dataframe into multi-hot visit sequences.

    Each patient is converted into a ``(seq_len, vocab_size)`` tensor of
    multi-hot visit vectors, paired with a single static label (the per-patient
    max of ``label_col``).

    Args:
        df: Input dataframe with one row per (patient, visit, code) event.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the binary label.
        code_to_idx: Optional precomputed mapping from code to integer index.
            If ``None``, one is built from ``df``.

    Returns:
        A tuple ``(patients, code_to_idx)`` where ``patients`` is a list of
            ``(sequence_tensor, label)`` tuples.
    """
    assert label_col in df.columns, f"Label column '{label_col}' not found."
    assert subject_col in df.columns, f"Subject column '{subject_col}' not found."
    assert visit_col in df.columns, f"Visit column '{visit_col}' not found."

    df = df.copy()
    if code_to_idx is None:
        vocab_size = df[code_col].nunique() + 1
        code_to_idx = {
            code: idx for idx, code in enumerate(df[code_col].unique(), start=0)
        }
    else:
        vocab_size = len(code_to_idx) + 1
    df[code_col] = df[code_col].map(code_to_idx)

    patients = []
    for _, group in df.groupby(subject_col):
        # Static per-patient label: the max over visits (e.g. "ever diagnosed").
        label = group[label_col].max()
        visits = group.sort_values(visit_col).groupby(visit_col)
        patient_seq = []
        for _, visit_data in visits:
            multi_hot = torch.zeros(vocab_size)
            codes = visit_data[code_col].values
            multi_hot[codes] = 1.0
            patient_seq.append(multi_hot)
        patient_seq_tensor = torch.stack(patient_seq)
        patients.append((patient_seq_tensor, label))

    return patients, code_to_idx


def collate_fn(batch):
    """Pads variable-length visit sequences for batched LSTM training.

    Args:
        batch: List of ``(sequence_tensor, label)`` tuples.

    Returns:
        A tuple ``(padded_seqs, lengths, labels)``.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded_seqs, lengths, labels


class EHRDataset(torch.utils.data.Dataset):
    """A minimal :class:`torch.utils.data.Dataset` wrapper over a list."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EHR_LSTM_Classifier(nn.Module):
    """A simple LSTM classifier over multi-hot EHR visit sequences.

    The model embeds each multi-hot visit vector, encodes the sequence with an
    LSTM, and classifies using the final hidden state.

    Args:
        vocab_size: Size of the code vocabulary (input dimension per visit).
        embed_dim: Dimension of the dense visit embedding.
        hidden_dim: Hidden dimension of the LSTM.
        num_layers: Number of stacked LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed_x)
        final_encoding = h_n[-1]
        logits = self.fc(final_encoding)
        probs = self.sigmoid(logits)
        return probs.squeeze(-1)


def train_lstm_model(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    code_col: str,
    label_col: str,
    embed_dim: int = 32,
    hidden_dim: int = 32,
    batch_size: int = 32,
    epochs: int = 5,
    verbose: bool = True,
    seed: int = 4,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    """Trains :class:`EHR_LSTM_Classifier` and evaluates it on a test set.

    Args:
        train_ehr: Training EHR dataframe.
        test_ehr: Test EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the binary label.
        embed_dim: Visit embedding dimension.
        hidden_dim: LSTM hidden dimension.
        batch_size: Training/eval batch size.
        epochs: Number of training epochs.
        verbose: Whether to print per-epoch loss.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(model, y_true, y_pred)`` where ``y_true`` and ``y_pred`` are
            numpy arrays of test labels and binary predictions.
    """
    torch.manual_seed(seed)
    all_codes = set()
    all_codes.update(train_ehr[code_col].unique().tolist())
    all_codes.update(test_ehr[code_col].unique().tolist())
    # Sort before enumerating: a Python set has non-deterministic iteration
    # order across processes, which would make the feature mapping (and thus
    # the trained model / reported metrics) irreproducible even with a fixed
    # seed. Start indices at 1 to reserve 0 for padding.
    code_to_idx = {
        code: idx for idx, code in enumerate(sorted(all_codes), start=1)
    }

    train_data, _ = process_patient_data_for_lstm(
        train_ehr, subject_col, visit_col, code_col, label_col, code_to_idx
    )
    test_data, _ = process_patient_data_for_lstm(
        test_ehr, subject_col, visit_col, code_col, label_col, code_to_idx
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=EHRDataset(train_data),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=EHRDataset(test_data),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    model = EHR_LSTM_Classifier(
        vocab_size=len(code_to_idx) + 1,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_lens, batch_y in train_dataloader:
            optimizer.zero_grad()
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            predictions = model(batch_x, batch_lens)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            avg_loss = total_loss / max(len(train_dataloader), 1)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    all_preds: List[float] = []
    all_labels: List[float] = []
    with torch.no_grad():
        for batch_x, batch_lens, batch_y in test_dataloader:
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            predictions = model(batch_x, batch_lens)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array([1 if p >= 0.5 else 0 for p in all_preds])
    return model, y_true, y_pred


# --- Random-forest baseline ------------------------------------------------


def aggregate_patient_visits(
    df: pd.DataFrame,
    subject_col: str,
    code_col: str,
    label_col: str,
    code_to_idx: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregates each patient's visits into a single multi-hot vector.

    Args:
        df: Input dataframe with integer-encoded codes in ``code_col``.
        subject_col: Column name for patient/subject identifiers.
        code_col: Column name for the (integer-encoded) medical codes.
        label_col: Column name for the binary label.
        code_to_idx: Mapping from code to index (used to size the vector).

    Returns:
        A tuple ``(patient_vectors, patient_labels)`` of numpy arrays.
    """
    patient_vectors = []
    patient_labels = []
    for _, group in df.groupby(subject_col):
        codes = group[code_col].unique()
        multi_hot = np.zeros(len(code_to_idx) + 1)
        multi_hot[codes] = 1
        patient_vectors.append(multi_hot)
        patient_labels.append(group[label_col].max())
    return np.array(patient_vectors), np.array(patient_labels)


def train_sklearn_model(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    code_col: str,
    label_col: str,
    model: str = "rf",
    seed: int = 4,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Trains an sklearn classifier on aggregated patient-level multi-hot data.

    Args:
        train_ehr: Training EHR dataframe.
        test_ehr: Test EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers (unused, kept for
            a uniform signature with :func:`train_lstm_model`).
        code_col: Column name for the medical codes.
        label_col: Column name for the binary label.
        model: Which model to train. Only ``"rf"`` (random forest) is supported.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(model, y_true, y_pred)``.
    """
    train_ehr = train_ehr.copy()
    test_ehr = test_ehr.copy()

    all_codes = set()
    all_codes.update(train_ehr[code_col].unique().tolist())
    all_codes.update(test_ehr[code_col].unique().tolist())
    # Sort before enumerating: a Python set has non-deterministic iteration
    # order across processes, which would make the feature mapping (and thus
    # the trained model / reported metrics) irreproducible even with a fixed
    # seed. Start indices at 1 to reserve 0 for padding.
    code_to_idx = {
        code: idx for idx, code in enumerate(sorted(all_codes), start=1)
    }
    train_ehr[code_col] = train_ehr[code_col].map(code_to_idx)
    test_ehr[code_col] = test_ehr[code_col].map(code_to_idx)

    X_train, y_train = aggregate_patient_visits(
        train_ehr, subject_col, code_col, label_col, code_to_idx
    )
    X_test, y_test = aggregate_patient_visits(
        test_ehr, subject_col, code_col, label_col, code_to_idx
    )

    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise NotImplementedError(f"Model '{model}' not implemented.")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    return clf, y_test, y_pred


# --- Task / feature construction ------------------------------------------


def build_next_visit_prediction_dataset(
    df: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    label_col: str,
    multi_visit_sample_frac: float = 0.5,
    seed: int = 4,
) -> pd.DataFrame:
    """Builds a next-visit prediction task from an EHR dataframe.

    For patients with multiple visits, a fraction is sampled and their last
    visit is dropped; these patients are labeled 1 (has a next visit). The
    remaining multi-visit patients are kept intact and labeled 0. Single-visit
    patients are labeled 0 by definition.

    Args:
        df: Input EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        label_col: Column name to overwrite with the next-visit label.
        multi_visit_sample_frac: Fraction of multi-visit patients to truncate.
        seed: Random seed for reproducibility.

    Returns:
        A new dataframe with ``label_col`` set to the next-visit label.
    """
    assert 0.0 <= multi_visit_sample_frac <= 1.0, (
        "multi_visit_sample_frac must be in [0, 1]."
    )

    rng = np.random.default_rng(seed)
    transformed_groups = []

    for _, group in df.groupby(subject_col):
        group_sorted = group.sort_values(visit_col)
        unique_visits = np.sort(group_sorted[visit_col].unique())
        n_visits = len(unique_visits)

        if n_visits <= 1:
            g = group_sorted.copy()
            g[label_col] = 0
            transformed_groups.append(g)
            continue

        should_truncate = rng.random() < multi_visit_sample_frac
        if should_truncate:
            last_visit = unique_visits[-1]
            g = group_sorted[group_sorted[visit_col] != last_visit].copy()
            if g.empty:
                # Defensive fallback for unexpected edge cases.
                g = group_sorted.copy()
                g[label_col] = 0
            else:
                g[label_col] = 1
        else:
            g = group_sorted.copy()
            g[label_col] = 0
        transformed_groups.append(g)

    if len(transformed_groups) == 0:
        return df.copy()
    return pd.concat(transformed_groups, ignore_index=True)


def convert_cols_to_multihot(
    df: pd.DataFrame,
    code_col: str,
    visit_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    bins_per_num: int = 5,
) -> pd.DataFrame:
    """Folds categorical and numeric columns into per-visit multi-hot codes.

    Categorical columns are prefixed with their column name; numeric columns
    are quantile-binned and likewise prefixed. All values are combined with the
    original code into a single comma-separated ``combined_codes`` column.

    Args:
        df: Input dataframe.
        code_col: Column name for the existing medical codes.
        visit_col: Column name for visit/timestep identifiers (kept for a
            uniform signature; not modified).
        cat_cols: Categorical column names to fold in.
        num_cols: Numeric column names to bin and fold in.
        bins_per_num: Number of quantile bins per numeric column.

    Returns:
        A copy of ``df`` with an added ``combined_codes`` column.
    """
    df = df.copy()
    for col in cat_cols:
        df[col] = col + "_" + df[col].astype(str)

    for col in num_cols:
        df[col + "_binned"] = pd.qcut(
            df[col], q=bins_per_num, duplicates="drop"
        ).astype(str)
        df[col + "_binned"] = col + "_" + df[col + "_binned"]

    def combine_codes(row):
        codes = [str(row[code_col])]
        for col in cat_cols:
            codes.append(str(row[col]))
        for col in num_cols:
            codes.append(str(row[col + "_binned"]))
        return ",".join(codes)

    df["combined_codes"] = df.apply(combine_codes, axis=1)
    return df
