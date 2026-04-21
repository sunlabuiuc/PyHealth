"""
Example usage of DynamicSurvivalRNN on MIMIC-IV Demo.

This script:
- uses first 48 hours of ICU charted measurements as input
- predicts death within 7 days of ICU admission
- compares a baseline EEP GRU against DynamicSurvivalRNN
- can be used for a small ablation, e.g. hidden_dim=32 vs 64
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pyhealth.datasets import SampleDataset
from pyhealth.models.dynamic_survival_rnn import DynamicSurvivalRNN

SEED = 7
DATA_DIR = "/mnt/data"

ADMISSIONS_PATH = os.path.join(DATA_DIR, "hosp", "admissions.csv.gz")
PATIENTS_PATH = os.path.join(DATA_DIR, "hosp", "patients.csv.gz")
ICUSTAYS_PATH = os.path.join(DATA_DIR, "icu", "icustays.csv.gz")
CHARTEVENTS_PATH = os.path.join(DATA_DIR, "icu", "chartevents.csv.gz")
D_ITEMS_PATH = os.path.join(DATA_DIR, "icu", "d_items.csv.gz")

MAX_HOURS = 48
PRED_HORIZON = 168
MIN_FEATURE_COUNT = 10
BATCH_SIZE = 16
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_FEATURE_PATTERNS = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Arterial Blood Pressure mean",
    "Non Invasive Blood Pressure mean",
    "Temperature Celsius",
    "Temperature Fahrenheit",
    "Glucose (serum)",
]

BAD_LABEL_PATTERNS = [
    "sensor",
    "in place",
    "alarm",
    "limit",
    "desat",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def safe_stratify(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return None
    values, counts = np.unique(labels, return_counts=True)
    if len(values) < 2:
        return None
    if counts.min() < 2:
        return None
    return labels


def load_tables():
    admissions = pd.read_csv(
        ADMISSIONS_PATH,
        compression="gzip",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    patients = pd.read_csv(
        PATIENTS_PATH,
        compression="gzip",
        parse_dates=["dod"],
    )
    icustays = pd.read_csv(
        ICUSTAYS_PATH,
        compression="gzip",
        parse_dates=["intime", "outtime"],
    )
    chartevents = pd.read_csv(
        CHARTEVENTS_PATH,
        compression="gzip",
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
    )
    d_items = pd.read_csv(
        D_ITEMS_PATH,
        compression="gzip",
        usecols=["itemid", "label", "category", "unitname", "param_type"],
    )
    return admissions, patients, icustays, chartevents, d_items


def choose_itemids(d_items: pd.DataFrame) -> pd.DataFrame:
    d_items = d_items.copy()
    d_items["label_lower"] = d_items["label"].astype(str).str.lower()
    d_items = d_items[
        d_items["param_type"].astype(str).str.contains("Numeric", case=False, na=False)
    ].copy()
    d_items = d_items[
        ~d_items["label_lower"].str.contains("|".join(BAD_LABEL_PATTERNS), na=False)
    ].copy()

    chosen_rows = []
    used_itemids = set()

    for pattern in TARGET_FEATURE_PATTERNS:
        mask = d_items["label_lower"].str.contains(pattern.lower(), regex=False, na=False)
        candidates = d_items[mask].copy()
        if len(candidates) == 0:
            continue

        candidates["numeric_score"] = candidates["param_type"].astype(str).str.contains(
            "Numeric", case=False, na=False
        ).astype(int)
        candidates["routine_score"] = candidates["category"].astype(str).str.contains(
            "Routine Vital Signs|Respiratory|Labs", case=False, na=False
        ).astype(int)
        candidates = candidates.sort_values(
            ["numeric_score", "routine_score", "itemid"], ascending=[False, False, True]
        )
        row = candidates.iloc[0]
        itemid = int(row["itemid"])
        if itemid not in used_itemids:
            used_itemids.add(itemid)
            chosen_rows.append(row)

    chosen = pd.DataFrame(chosen_rows)
    if chosen.empty:
        raise RuntimeError("No candidate itemids found in d_items.")
    return chosen[["itemid", "label", "category", "unitname", "param_type"]].drop_duplicates()


def build_icu_hourly_sequences(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    icustays: pd.DataFrame,
    chartevents: pd.DataFrame,
    chosen_items: pd.DataFrame,
    max_hours: int = 48,
    min_feature_count: int = 10,
) -> Tuple[List[np.ndarray], List[int], List[int], List[Dict]]:
    admissions = admissions.copy()
    patients = patients.copy()
    icustays = icustays.copy()
    chartevents = chartevents.copy()

    admissions["hadm_id"] = admissions["hadm_id"].astype(int)
    patients["subject_id"] = patients["subject_id"].astype(int)
    icustays["stay_id"] = icustays["stay_id"].astype(int)
    icustays["hadm_id"] = icustays["hadm_id"].astype(int)
    icustays["subject_id"] = icustays["subject_id"].astype(int)
    chartevents = chartevents.dropna(subset=["stay_id", "charttime", "itemid", "valuenum"]).copy()
    chartevents["stay_id"] = chartevents["stay_id"].astype(int)
    chartevents["itemid"] = chartevents["itemid"].astype(int)

    itemids = chosen_items["itemid"].astype(int).tolist()
    itemid_to_idx = {itemid: i for i, itemid in enumerate(itemids)}
    feature_names = chosen_items["label"].tolist()

    chartevents = chartevents[chartevents["itemid"].isin(itemids)]

    adm_lookup = admissions.set_index("hadm_id").to_dict(orient="index")
    pat_lookup = patients.set_index("subject_id").to_dict(orient="index")

    X_list: List[np.ndarray] = []
    events: List[int] = []
    ttes: List[int] = []
    metadata: List[Dict] = []

    for _, stay in icustays.iterrows():
        stay_id = int(stay["stay_id"])
        hadm_id = int(stay["hadm_id"])
        subject_id = int(stay["subject_id"])
        intime = stay["intime"]

        if pd.isna(intime):
            continue
        if hadm_id not in adm_lookup or subject_id not in pat_lookup:
            continue

        g = chartevents[chartevents["stay_id"] == stay_id].copy()
        if g.empty:
            continue

        g = g[g["charttime"] >= intime]
        g["hour_idx"] = ((g["charttime"] - intime).dt.total_seconds() // 3600).astype(int)
        g = g[(g["hour_idx"] >= 0) & (g["hour_idx"] < max_hours)]
        if g.empty:
            continue

        hourly = np.full((max_hours, len(itemids)), np.nan, dtype=np.float32)
        agg = g.groupby(["hour_idx", "itemid"])["valuenum"].mean().reset_index()
        for _, row in agg.iterrows():
            h = int(row["hour_idx"])
            itemid = int(row["itemid"])
            hourly[h, itemid_to_idx[itemid]] = float(row["valuenum"])

        if np.isfinite(hourly).sum() < min_feature_count:
            continue

        df_hourly = pd.DataFrame(hourly, columns=feature_names)
        df_hourly = df_hourly.ffill()
        df_hourly = df_hourly.fillna(df_hourly.mean(skipna=True))
        df_hourly = df_hourly.fillna(0.0)
        hourly = df_hourly.values.astype(np.float32)

        adm = adm_lookup[hadm_id]
        pat = pat_lookup[subject_id]

        death_candidates = []
        if pd.notna(adm.get("deathtime")):
            death_candidates.append(adm["deathtime"])
        if pd.notna(pat.get("dod")):
            death_candidates.append(pat["dod"])

        event = 0
        tte = max_hours + PRED_HORIZON + 1

        if death_candidates:
            death_time = min(death_candidates)
            delta_hours = int((death_time - intime).total_seconds() // 3600)
            if delta_hours >= 0:
                event = 1
                tte = delta_hours

        X_list.append(hourly)
        events.append(event)
        ttes.append(tte)
        metadata.append(
            {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "event": event,
                "tte_hours": tte,
            }
        )

    if not X_list:
        raise RuntimeError("No usable ICU stays found. Try lowering MIN_FEATURE_COUNT.")

    X_all = np.stack(X_list, axis=0)
    mean = X_all.mean(axis=(0, 1), keepdims=True)
    std = X_all.std(axis=(0, 1), keepdims=True) + 1e-6
    X_all = (X_all - mean) / std

    return list(X_all), events, ttes, metadata


class BaselineEEPDataset(Dataset):
    def __init__(self, X, events, ttes, horizon):
        self.X = X
        self.events = events
        self.ttes = ttes
        self.horizon = horizon

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = 1.0 if (self.events[idx] == 1 and self.ttes[idx] <= self.horizon) else 0.0
        return {"x": x, "y": torch.tensor(y, dtype=torch.float32)}

class DummyDataset:
    def __init__(self, samples):
        self.samples = samples
        self.input_schema = {
            "x": "timeseries",
            "hazard_y": "vector",
            "hazard_mask": "vector",
        }
        self.output_schema = {
            "event_within_h": "binary",
        }

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
class DynamicSurvivalDataset(Dataset):
    def __init__(self, X, events, ttes, horizon):
        self.X = X
        self.events = events
        self.ttes = ttes
        self.horizon = horizon

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        event = self.events[idx]
        tte = self.ttes[idx]

        hazard_y = np.zeros(self.horizon, dtype=np.float32)
        hazard_mask = np.ones(self.horizon, dtype=np.float32)

        if event == 1 and 1 <= tte <= self.horizon:
            hazard_y[tte - 1] = 1.0

        event_within_h = 1.0 if (event == 1 and 1 <= tte <= self.horizon) else 0.0
        return {
            "x": x,
            "hazard_y": torch.tensor(hazard_y, dtype=torch.float32),
            "hazard_mask": torch.tensor(hazard_mask, dtype=torch.float32),
            "event_within_h": torch.tensor(event_within_h, dtype=torch.float32),
        }


class BaselineGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, h = self.gru(x)
        last_hidden = h[-1]
        logits = self.head(last_hidden).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs}


@dataclass
class Metrics:
    auroc: float
    auprc: float


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Metrics:
    if len(np.unique(y_true)) < 2:
        return Metrics(float("nan"), float("nan"))
    return Metrics(
        auroc=roc_auc_score(y_true, y_score),
        auprc=average_precision_score(y_true, y_score),
    )


def train_baseline(model, train_loader, val_loader, epochs=20, lr=1e-3):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            out = model(x)
            loss = criterion(out["logits"], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(DEVICE)
                y = batch["y"].cpu().numpy()
                p = model(x)["probs"].cpu().numpy()
                ys.extend(y.tolist())
                ps.extend(p.tolist())
        m = compute_binary_metrics(np.array(ys), np.array(ps))
        print(
            f"[Baseline] epoch={epoch:02d} "
            f"loss={np.mean(losses):.4f} val_auprc={m.auprc:.4f} val_auroc={m.auroc:.4f}"
        )


def train_survival(model, train_loader, val_loader, prevalence, epochs=20, lr=1e-3):
    model.to(DEVICE)
    model.initialize_bias_from_prevalence(prevalence)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auprc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            hazard_y = batch["hazard_y"].to(DEVICE)
            hazard_mask = batch["hazard_mask"].to(DEVICE)
            event_within_h = batch["event_within_h"].to(DEVICE)

            ret = model(
                x=x,
                hazard_y=hazard_y,
                hazard_mask=hazard_mask,
                event_within_h=event_within_h,
            )
            loss = ret["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(DEVICE)
                hazard_y = batch["hazard_y"].to(DEVICE)
                hazard_mask = batch["hazard_mask"].to(DEVICE)
                event_within_h = batch["event_within_h"].to(DEVICE)

                ret = model(
                    x=x,
                    hazard_y=hazard_y,
                    hazard_mask=hazard_mask,
                    event_within_h=event_within_h,
                )
                ys.extend(event_within_h.cpu().numpy().tolist())
                ps.extend(ret["y_prob"].squeeze(-1).cpu().numpy().tolist())

        m = compute_binary_metrics(np.array(ys), np.array(ps))
        print(
            f"[Survival] epoch={epoch:02d} "
            f"loss={np.mean(losses):.4f} val_auprc={m.auprc:.4f} val_auroc={m.auroc:.4f}"
        )

        if not np.isnan(m.auprc) and m.auprc > best_val_auprc:
            best_val_auprc = m.auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)


def evaluate_baseline(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].cpu().numpy()
            p = model(x)["probs"].cpu().numpy()
            ys.extend(y.tolist())
            ps.extend(p.tolist())
    return compute_binary_metrics(np.array(ys), np.array(ps))


def evaluate_survival(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            hazard_y = batch["hazard_y"].to(DEVICE)
            hazard_mask = batch["hazard_mask"].to(DEVICE)
            event_within_h = batch["event_within_h"].to(DEVICE)

            ret = model(
                x=x,
                hazard_y=hazard_y,
                hazard_mask=hazard_mask,
                event_within_h=event_within_h,
            )
            ys.extend(event_within_h.cpu().numpy().tolist())
            ps.extend(ret["y_prob"].squeeze(-1).cpu().numpy().tolist())
    return compute_binary_metrics(np.array(ys), np.array(ps))


def estimate_hazard_prevalence(dataset):
    ys = [dataset[i]["hazard_y"].numpy() for i in range(len(dataset))]
    ys = np.stack(ys, axis=0)
    prevalence = ys.mean(axis=0)
    return np.clip(prevalence, 1e-4, 1 - 1e-4)


def subset(lst, idxs):
    return [lst[i] for i in idxs]


def run_one_experiment(hidden_dim: int):
    print(f"\n=== Running experiment with hidden_dim={hidden_dim} ===")
    admissions, patients, icustays, chartevents, d_items = load_tables()

    chosen_items = choose_itemids(d_items)
    X, events, ttes, metadata = build_icu_hourly_sequences(
        admissions=admissions,
        patients=patients,
        icustays=icustays,
        chartevents=chartevents,
        chosen_items=chosen_items,
        max_hours=MAX_HOURS,
        min_feature_count=MIN_FEATURE_COUNT,
    )

    labels_within_h = [1 if (e == 1 and t <= PRED_HORIZON) else 0 for e, t in zip(events, ttes)]

    idx = np.arange(len(X))
    strat = safe_stratify(labels_within_h)
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=SEED, stratify=strat
    )

    train_labels = np.array([labels_within_h[i] for i in train_idx])
    strat2 = safe_stratify(train_labels)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=SEED, stratify=strat2
    )

    X_train, X_val, X_test = subset(X, train_idx), subset(X, val_idx), subset(X, test_idx)
    e_train, e_val, e_test = subset(events, train_idx), subset(events, val_idx), subset(events, test_idx)
    t_train, t_val, t_test = subset(ttes, train_idx), subset(ttes, val_idx), subset(ttes, test_idx)

    baseline_train = BaselineEEPDataset(X_train, e_train, t_train, PRED_HORIZON)
    baseline_val = BaselineEEPDataset(X_val, e_val, t_val, PRED_HORIZON)
    baseline_test = BaselineEEPDataset(X_test, e_test, t_test, PRED_HORIZON)

    survival_train = DynamicSurvivalDataset(X_train, e_train, t_train, PRED_HORIZON)
    survival_val = DynamicSurvivalDataset(X_val, e_val, t_val, PRED_HORIZON)
    survival_test = DynamicSurvivalDataset(X_test, e_test, t_test, PRED_HORIZON)

    train_loader_b = DataLoader(baseline_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_b = DataLoader(baseline_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_b = DataLoader(baseline_test, batch_size=BATCH_SIZE, shuffle=False)

    train_loader_s = DataLoader(survival_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_s = DataLoader(survival_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_s = DataLoader(survival_test, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train[0].shape[1]
    baseline_model = BaselineGRU(input_dim=input_dim, hidden_dim=hidden_dim)

    toy_sample_dataset = DummyDataset(
        [
            {
                "x": X_train[0],
                "hazard_y": survival_train[0]["hazard_y"].numpy(),
                "hazard_mask": survival_train[0]["hazard_mask"].numpy(),
                "event_within_h": float(survival_train[0]["event_within_h"].item()),
            }
        ]
    )

    prevalence = estimate_hazard_prevalence(survival_train)

    survival_model = DynamicSurvivalRNN(
        dataset=toy_sample_dataset,
        feature_key="x",
        label_key="event_within_h",
        hazard_label_key="hazard_y",
        hazard_mask_key="hazard_mask",
        hidden_dim=hidden_dim,
        horizon=PRED_HORIZON,
        bias_init_prevalence=prevalence,
    )

    print("Training baseline model...")
    train_baseline(baseline_model, train_loader_b, val_loader_b, epochs=EPOCHS, lr=LR)

    print("Training survival model...")
    train_survival(survival_model, train_loader_s, val_loader_s, prevalence, epochs=EPOCHS, lr=LR)

    baseline_model.to(DEVICE)
    survival_model.to(DEVICE)

    mb = evaluate_baseline(baseline_model, test_loader_b)
    ms = evaluate_survival(survival_model, test_loader_s)

    print("\n===== Final test metrics =====")
    print(f"[hidden_dim={hidden_dim}] Baseline EEP  AUROC={mb.auroc:.4f} AUPRC={mb.auprc:.4f}")
    print(f"[hidden_dim={hidden_dim}] DSA Survival  AUROC={ms.auroc:.4f} AUPRC={ms.auprc:.4f}")

    return {
        "hidden_dim": hidden_dim,
        "baseline_auroc": mb.auroc,
        "baseline_auprc": mb.auprc,
        "survival_auroc": ms.auroc,
        "survival_auprc": ms.auprc,
    }


def main():
    print("Using device:", DEVICE)
    results = []
    for hidden_dim in [32, 64]:
        results.append(run_one_experiment(hidden_dim))

    print("\n===== Ablation Summary =====")
    for r in results:
        print(
            f"hidden_dim={r['hidden_dim']}: "
            f"baseline AUROC={r['baseline_auroc']:.4f}, "
            f"baseline AUPRC={r['baseline_auprc']:.4f}, "
            f"survival AUROC={r['survival_auroc']:.4f}, "
            f"survival AUPRC={r['survival_auprc']:.4f}"
        )


if __name__ == "__main__":
    main()

