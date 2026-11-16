"""Tabular ICU mortality on eICU with XGBoost: task ablation and synthetic data.

Contributors:
    Alex Castillo (acast23@illinois.edu)
    Srikar Lunke (srikarl2@illinois.edu)
    Maheswara Vatti (mvatti2@illinois.edu)

Paper:
    Lin, Yu and Lee, "A Case Study Exploring the Current Landscape of
    Synthetic Medical Record Generation with Commercial LLMs", CHIL 2025
    (PMLR 287), https://proceedings.mlr.press/v287/lin25a.html

What it shows, with :class:`~pyhealth.tasks.TabularMortalityPredictionEICU`
(one row of early features per ICU stay, label = death in the ICU) and
:class:`~pyhealth.models.XGBoostModel`:

1. Task-configuration ablation. The task's two settings are varied: the early
   window (``window_hours`` of 6, 12 and 24) and the feature set (demographics
   only, first lab values, last lab values, all features, and the task's
   default ten). Each configuration is scored by AUROC and AUPRC on held-out
   patients. The paper varies only how many features it keeps and how many
   rows it generates.

2. Real vs. synthetic training data. The paper scores synthetic records by
   the average marginal KL divergence from the real data (fidelity) and by
   training XGBoost on them and testing on real patients (utility). Three
   references the paper does not report are added: the same model trained on
   real data, on a real sample of the same size as the synthetic sets, and on
   an independent-marginal baseline, where every column is drawn on its own
   from the real data. That baseline has a KL divergence near zero and no
   predictive value, which shows that marginal KL alone cannot certify
   synthetic data. The sample and the baseline are drawn ten times and
   averaged, because a model fitted to 1,000 rows varies from draw to draw.
   With ``--synthetic-dir`` the LLM-generated tables written by
   ``examples/eicu_tabular_mortality_llm_generation.py`` are scored the same
   way.

Data:
    eICU-CRD is credentialed (https://physionet.org/content/eicu-crd/). Pass
    its folder with ``--root``. Without ``--root`` the script runs on the small
    eICU demo bundled with PyHealth, which needs no credentials. The demo has
    174 stays and 5 deaths, so its scores only show that the script runs.

Needs the optional extra: ``pip install "pyhealth[xgboost]"``. On macOS,
XGBoost and PyTorch each load an OpenMP runtime and the fit can crash; see the
macOS note in the :class:`~pyhealth.models.XGBoostModel` docs.

Usage:
    Try it without credentials (bundled demo)::

        python examples/eicu_tabular_mortality_prediction_xgboost.py

    Full eICU, and score LLM-generated tables as well::

        python examples/eicu_tabular_mortality_prediction_xgboost.py \\
            --root /path/to/eicu-crd/2.0 --synthetic-dir ~/eicu_synthetic

Findings (eICU-CRD 2.0, every ICU stay, patient-level 80/20 split, seed 0;
160,625 training stays, 5.4% mortality; scores on the held-out patients):

    Part 1. AUROC (AUPRC) by early window and feature set:

    =====================  =============  =============  =============
    feature set (number)   6 hours        12 hours       24 hours
    =====================  =============  =============  =============
    first lab values (20)  0.738 (0.238)  0.783 (0.275)  0.831 (0.307)
    last lab values (20)   0.738 (0.242)  0.788 (0.300)  0.840 (0.352)
    all features (42)      0.770 (0.267)  0.808 (0.312)  0.851 (0.377)
    default ten (10)       0.759 (0.247)  0.790 (0.291)  0.838 (0.359)
    =====================  =============  =============  =============

    Age and sex alone give 0.603 (0.075) at any window. A longer window helps
    every feature set. Last values overtake first values as the window grows.
    The default ten features stay within 0.013 AUROC of all 42.

    Part 2. Default ten features, 24-hour window; AUROC and AUPRC of a
    model trained on each table:

    =================================  ======  ===================  ===================
    training data                      avg KL  AUROC                AUPRC
    =================================  ======  ===================  ===================
    real train pool (160,625 rows)          -  0.838                0.359
    real sample (1,000 rows)                -  0.770 (0.746-0.792)  0.212 (0.145-0.254)
    independent-marginal (1,000 rows)   0.009  0.535 (0.376-0.692)  0.063 (0.040-0.103)
    LLM, naive prompt (1,000 rows)      0.734  0.666                0.131
    LLM, group prompt (1,000 rows)      0.718  0.702                0.131
    =================================  ======  ===================  ===================

    The sample and the baseline are means of ten draws, with the smallest and
    largest draw in brackets. The independent-marginal baseline has by far
    the best KL of the generated tables, yet a model trained on it is at
    chance on average, so marginal KL alone cannot certify synthetic data.
    The LLM tables (one run each, from qwen2.5:14b run locally with Ollama,
    as the eICU data use agreement requires) fall between that baseline and a
    real sample of the same size. The paper, with o1 and a filtered cohort,
    reports an average KL of 0.155 and an AUROC of 0.813 for its group-based
    prompt.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import create_sample_dataset, eICUDataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn, bootstrap_ci
from pyhealth.models import XGBoostModel
from pyhealth.tasks import TabularMortalityPredictionEICU

LABEL = "mortality"
DEMO_ROOT = Path(__file__).resolve().parents[1] / "test-resources/core/eicudemo"

DEFAULT_FEATURES = list(TabularMortalityPredictionEICU.DEFAULT_FEATURES)
ALL_FEATURES = list(TabularMortalityPredictionEICU.SUPPORTED_FEATURES)
DEMOGRAPHICS = ["age", "is_female"]
# The feature sets of the ablation, each a choice of the task's ``features``.
FEATURE_SETS: dict[str, list[str]] = {
    "demographics only": DEMOGRAPHICS,
    "first lab values": [f for f in ALL_FEATURES if f.endswith("_first_early")],
    "last lab values": [f for f in ALL_FEATURES if f.endswith("_last_early")],
    "all features": ALL_FEATURES,
    "default ten": DEFAULT_FEATURES,
}


# ---------------------------------------------------------------------------
# Data: run the task and flatten its samples to one table row per ICU stay
# ---------------------------------------------------------------------------
def build_table(
    root: str,
    features: Sequence[str],
    window_hours: float = 24.0,
    dev: bool = False,
) -> pd.DataFrame:
    """Runs the task on eICU and returns one row per ICU stay.

    Args:
        root: Folder with the eICU CSV files (``.csv`` or ``.csv.gz``).
        features: Feature names passed to the task.
        window_hours: Early window passed to the task.
        dev: Load only the first 1,000 patients, for a quick trial run.

    Returns:
        A table with ``patient_id``, ``stay_id``, one column per feature (NaN
        where missing) and ``mortality``.
    """
    dataset = eICUDataset(root=root, tables=["lab"], dev=dev)
    task = TabularMortalityPredictionEICU(
        features=list(features), window_hours=window_hours
    )
    samples = dataset.set_task(task)
    rows = []
    for index in range(len(samples)):
        sample = samples[index]
        row = {"patient_id": sample["patient_id"], "stay_id": int(sample["stay_id"])}
        row.update(zip(task.features, sample["features"].tolist()))
        row[LABEL] = int(float(sample[LABEL]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["patient_id", "stay_id", *task.features, LABEL])


def split_table_by_patient(
    table: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a table so that no patient is on both sides.

    Args:
        table: A table with a ``patient_id`` column.
        test_fraction: Share of patients held out.
        seed: Seed of the shuffle.

    Returns:
        The train pool and the held-out test set.
    """
    patients = np.sort(table["patient_id"].astype(str).unique())
    np.random.default_rng(seed).shuffle(patients)
    n_test = max(1, round(len(patients) * test_fraction))
    held_out = table["patient_id"].astype(str).isin(patients[:n_test])
    return table[~held_out].reset_index(drop=True), table[held_out].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Metrics: the paper's fidelity (KL divergence) and utility (XGBoost) scores
# ---------------------------------------------------------------------------
def average_marginal_kl(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: Sequence[str],
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Average KL divergence between real and synthetic feature marginals.

    For each feature, the real values set the bin edges (quantiles, so each
    bin holds about the same share of real rows); synthetic values beyond the
    real range fall in the end bins. The divergence is KL(P || Q) with P the
    real and Q the synthetic bin shares, each smoothed by ``epsilon``. Missing
    values are left out. A feature with no values on either side is skipped.

    Args:
        real: The real table (P).
        synthetic: The synthetic table (Q).
        features: Names of the feature columns to compare.
        bins: Largest number of bins per feature.
        epsilon: Added to every bin share so empty bins do not give infinity.

    Returns:
        The mean divergence over the features, 0 for identical tables and
        larger for worse fidelity; NaN if no feature could be compared.

    Examples:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"age": [40.0, 50.0, 60.0, 70.0, 80.0]})
        >>> average_marginal_kl(real, real, ["age"])
        0.0
        >>> average_marginal_kl(real, real + 25, ["age"]) > 0.5
        True
    """
    divergences = []
    for name in features:
        p_values = real[name].dropna().to_numpy(dtype=float)
        q_values = synthetic[name].dropna().to_numpy(dtype=float)
        if len(p_values) == 0 or len(q_values) == 0:
            continue
        edges = np.unique(np.quantile(p_values, np.linspace(0, 1, bins + 1)))
        # Interior edges only, so values beyond the real range land in the
        # first or last bin.
        inner = edges[1:-1]
        n_bins = len(inner) + 1
        p = np.bincount(
            np.searchsorted(inner, p_values, side="right"), minlength=n_bins
        )
        q = np.bincount(
            np.searchsorted(inner, q_values, side="right"), minlength=n_bins
        )
        p = (p / p.sum() + epsilon) / (1 + n_bins * epsilon)
        q = (q / q.sum() + epsilon) / (1 + n_bins * epsilon)
        divergences.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(divergences)) if divergences else float("nan")


def train_synthetic_test_real(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Fits XGBoost on one table and scores mortality prediction on another.

    Pass a synthetic table as ``train`` and a real held-out table as ``test``
    for the paper's "across" setting, or a real train pool as ``train`` for
    the real-data reference. Missing feature values stay NaN, which XGBoost
    reads as missing.

    Args:
        train: Table to fit on; needs both classes in ``mortality``.
        test: Table to score on.
        features: Names of the feature columns, in model input order.
        n_boot: Bootstrap resamples for the confidence intervals; 0 skips
            them and leaves the bounds NaN.
        seed: Seed of the model and of the bootstrap.

    Returns:
        ``{"roc_auc": {...}, "pr_auc": {...}}``, each with ``estimate``,
        ``lower`` and ``upper`` (95% interval). All NaN if ``train`` or
        ``test`` holds a single class.

    Examples:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=200)
        >>> table = pd.DataFrame({"x": x, "mortality": (x > 0.5).astype(int)})
        >>> scores = train_synthetic_test_real(table, table, ["x"], n_boot=20)
        >>> scores["roc_auc"]["estimate"] > 0.9
        True
    """
    nan = {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan")}
    if train[LABEL].nunique() < 2 or test[LABEL].nunique() < 2:
        return {"roc_auc": dict(nan), "pr_auc": dict(nan)}

    def to_samples(table: pd.DataFrame) -> list[dict]:
        matrix = table[list(features)].to_numpy(dtype=float)
        labels = table[LABEL].to_numpy(dtype=int)
        return [
            {"patient_id": str(i), "features": row.tolist(), LABEL: int(label)}
            for i, (row, label) in enumerate(zip(matrix, labels))
        ]

    # The same schema the task declares, so the model sees task-shaped samples.
    schema = (
        TabularMortalityPredictionEICU.input_schema,
        TabularMortalityPredictionEICU.output_schema,
    )
    train_set = create_sample_dataset(to_samples(train), *schema)
    test_set = create_sample_dataset(
        to_samples(test),
        *schema,
        input_processors=train_set.input_processors,
        output_processors=train_set.output_processors,
    )
    # n_jobs=1: on macOS, XGBoost and PyTorch each load an OpenMP runtime, and
    # a multithreaded fit can crash with both in one process.
    model = XGBoostModel(
        train_set,
        feature_names={"features": list(features)},
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_set)
    y_true, y_prob = [], []
    for batch in get_dataloader(test_set, batch_size=1024, shuffle=False):
        output = model(**batch)
        y_true.append(output["y_true"].numpy().reshape(-1))
        y_prob.append(output["y_prob"].numpy().reshape(-1))
    y_true, y_prob = np.concatenate(y_true), np.concatenate(y_prob)
    if n_boot == 0:
        point = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])
        return {
            metric: {**nan, "estimate": float(point[metric])}
            for metric in ("roc_auc", "pr_auc")
        }
    scores = {}
    for metric in ("roc_auc", "pr_auc"):
        ci = bootstrap_ci(y_true, y_prob, metric, n_boot=n_boot, seed=seed)
        scores[metric] = {key: float(ci[key]) for key in ("estimate", "lower", "upper")}
    return scores


def _score_columns(scores: dict[str, dict[str, float]]) -> dict[str, object]:
    """Formats AUROC and AUPRC with their intervals as table cells."""
    cells: dict[str, object] = {}
    for metric, label in (("roc_auc", "AUROC"), ("pr_auc", "AUPRC")):
        cells[label] = scores[metric]["estimate"]
        cells[f"{label} 95% CI"] = (
            f"[{scores[metric]['lower']:.3f}, {scores[metric]['upper']:.3f}]"
        )
    return cells


# ---------------------------------------------------------------------------
# Extension: the independent-marginal baseline
# ---------------------------------------------------------------------------
def independent_marginal_baseline(
    table: pd.DataFrame, columns: Sequence[str], n_rows: int, seed: int = 0
) -> pd.DataFrame:
    """Draws every column on its own from a real table.

    Each column keeps its real distribution, missing values included, but
    the columns are independent of each other and of the label. It is the
    weakest generator that still matches every marginal.

    Args:
        table: The real table.
        columns: Columns to draw, the label among them.
        n_rows: Number of rows to draw.
        seed: Seed of the draws.

    Returns:
        A table with ``columns`` and ``n_rows`` rows.

    Examples:
        >>> import pandas as pd
        >>> real = pd.DataFrame({"x": [1.0, 2.0, 3.0], "mortality": [0, 0, 1]})
        >>> fake = independent_marginal_baseline(real, ["x", "mortality"], 50)
        >>> fake.shape, set(fake["x"]) <= {1.0, 2.0, 3.0}
        ((50, 2), True)
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            column: rng.choice(table[column].to_numpy(), size=n_rows, replace=True)
            for column in columns
        }
    )


# ---------------------------------------------------------------------------
# Part 1: task-configuration ablation
# ---------------------------------------------------------------------------
def run_ablation(
    root: str,
    windows: Sequence[float],
    test_fraction: float = 0.2,
    n_boot: int = 200,
    seed: int = 0,
    dev: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scores XGBoost for every early window and feature set.

    For each window the task is run once with every supported feature, and
    each feature set is read from that table. This gives the same rows as
    running ``TabularMortalityPredictionEICU(features=<set>, window_hours=<w>)``
    for each set, because the task computes every feature on its own.

    Args:
        root: Folder with the eICU CSV files.
        windows: Values of ``window_hours`` to compare.
        test_fraction: Share of patients held out.
        n_boot: Bootstrap resamples for the confidence intervals.
        seed: Seed of the split, the model and the bootstrap.
        dev: Load only the first 1,000 patients.

    Returns:
        The ablation table (one row per configuration), and the train pool
        and held-out set of the last window, for reuse.
    """
    rows = []
    train = test = pd.DataFrame()
    for position, window_hours in enumerate(windows):
        table = build_table(root, ALL_FEATURES, window_hours, dev)
        train, test = split_table_by_patient(table, test_fraction, seed)
        for name, features in FEATURE_SETS.items():
            window_free = features == DEMOGRAPHICS
            if window_free and position > 0:
                continue  # age and sex do not depend on the window
            scores = train_synthetic_test_real(train, test, features, n_boot, seed)
            rows.append(
                {
                    "window_hours": "any" if window_free else f"{window_hours:g}",
                    "feature set": name,
                    "features": len(features),
                    **_score_columns(scores),
                }
            )
    return pd.DataFrame(rows), train, test


# ---------------------------------------------------------------------------
# Part 2: real vs. synthetic training data
# ---------------------------------------------------------------------------
def _score_training_table(
    table: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    generated: bool,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Scores one training table against the real train pool and test set.

    Args:
        table: The table to train on.
        train: The real train pool, the reference for the KL divergence.
        test: The real held-out set.
        generated: Whether ``table`` is generated; adds the KL divergence and
            the "within" scores (fit on 80% of the table, scored on the rest).
        n_boot: Bootstrap resamples for the "across" intervals; 0 skips them.
        seed: Seed of the models, the bootstrap and the within split.

    Returns:
        The table's mortality rate and scores, by column name.
    """
    features = DEFAULT_FEATURES
    cells = {"mortality": float(table[LABEL].mean())}
    if generated:
        cells["avg KL"] = average_marginal_kl(train, table, features)
        shuffled = table.sample(frac=1.0, random_state=seed)
        cut = int(len(shuffled) * 0.8)
        within = train_synthetic_test_real(
            shuffled[:cut], shuffled[cut:], features, 0, seed
        )
        cells["AUROC within"] = within["roc_auc"]["estimate"]
        cells["AUPRC within"] = within["pr_auc"]["estimate"]
    across = train_synthetic_test_real(table, test, features, n_boot, seed)
    for metric, label in (("roc_auc", "AUROC"), ("pr_auc", "AUPRC")):
        cells[f"{label} across"] = across[metric]["estimate"]
        cells[f"{label} lower"] = across[metric]["lower"]
        cells[f"{label} upper"] = across[metric]["upper"]
    return cells


def run_real_vs_synthetic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: dict[str, pd.DataFrame] | None = None,
    n_rows: int = 1000,
    n_draws: int = 10,
    n_boot: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Scores real, baseline and LLM-generated training tables.

    The real sample and the independent-marginal baseline are random draws,
    and a model fitted to 1,000 rows changes from draw to draw, so both are
    drawn ``n_draws`` times: the table reports the mean over the draws and
    the smallest and largest value. The other rows are single tables, with a
    95% bootstrap interval over the test patients.

    Args:
        train: The real train pool.
        test: The real held-out set.
        synthetic: LLM-generated tables by name, if any.
        n_rows: Size of the real sample and of the baseline; the paper's
            smallest synthetic set has 1,000 rows.
        n_draws: Number of draws of the real sample and of the baseline.
        n_boot: Bootstrap resamples for the confidence intervals.
        seed: Seed of the draws, the models and the bootstrap.

    Returns:
        One row per training table: its size and mortality rate, the average
        KL divergence from the real train pool, AUROC and AUPRC "across"
        (fit on the table, scored on the real held-out set) with their
        interval and, for generated tables, "within" (fit on 80% of the
        table, scored on the other 20%, as the paper also reports).
    """
    n_rows = min(n_rows, len(train))
    columns = [*DEFAULT_FEATURES, LABEL]
    rows = []

    def add(name: str, n: int, cells: dict[str, float], interval: str) -> None:
        row: dict[str, object] = {"training data": name, "rows": n, **cells}
        for label in ("AUROC", "AUPRC"):
            low, high = row.pop(f"{label} lower"), row.pop(f"{label} upper")
            row[f"{label} interval"] = f"[{low:.3f}, {high:.3f}]"
        row["interval"] = interval
        rows.append(row)

    bootstrap = "95% bootstrap over test patients"
    add(
        "real: train pool",
        len(train),
        _score_training_table(train, train, test, False, n_boot, seed),
        bootstrap,
    )

    # References drawn several times: mean, with the range over the draws.
    def real_sample(draw_seed: int) -> pd.DataFrame:
        return train.sample(n=n_rows, random_state=draw_seed)

    def baseline(draw_seed: int) -> pd.DataFrame:
        return independent_marginal_baseline(train, columns, n_rows, draw_seed)

    draws = {
        f"real: {n_rows}-row sample": real_sample,
        "independent-marginal baseline": baseline,
    }
    for name, draw in draws.items():
        generated = not name.startswith("real")
        scored = pd.DataFrame(
            [
                _score_training_table(draw(seed + i), train, test, generated, 0, seed)
                for i in range(n_draws)
            ]
        )
        cells = scored.mean().to_dict()
        for label in ("AUROC", "AUPRC"):
            cells[f"{label} lower"] = scored[f"{label} across"].min()
            cells[f"{label} upper"] = scored[f"{label} across"].max()
        add(name, n_rows, cells, f"min and max of {n_draws} draws")

    for name, table in (synthetic or {}).items():
        cells = _score_training_table(table, train, test, True, n_boot, seed)
        add(f"LLM: {name}", len(table), cells, bootstrap)

    order = [
        "training data",
        "rows",
        "mortality",
        "avg KL",
        "AUROC across",
        "AUROC interval",
        "AUPRC across",
        "AUPRC interval",
        "AUROC within",
        "AUPRC within",
        "interval",
    ]
    return pd.DataFrame(rows).reindex(columns=order)


def main() -> None:
    """Parses the command line and prints both tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", help="eICU folder; the bundled demo if omitted.")
    parser.add_argument(
        "--synthetic-dir",
        help="Folder written by eicu_tabular_mortality_llm_generation.py; its "
        "synthetic tables are scored against its saved real split.",
    )
    parser.add_argument("--output-dir", help="Also save both tables here as CSV.")
    parser.add_argument("--windows", type=float, nargs="+", default=[6.0, 12.0, 24.0])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument(
        "--n-draws",
        type=int,
        default=10,
        help="Draws of the real sample and of the baseline in part 2.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dev", action="store_true", help="Load only 1,000 patients (trial run)."
    )
    parser.add_argument(
        "--skip-ablation", action="store_true", help="Run only the second part."
    )
    args = parser.parse_args()
    root = args.root or str(DEMO_ROOT)
    if not args.root:
        print("No --root given: using the bundled eICU demo. Its 174 stays and 5")
        print("deaths are too few for meaningful scores.\n")

    def show(title: str, table: pd.DataFrame, file_name: str) -> None:
        print(f"\n===== {title} =====")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(output_dir / file_name, index=False)

    # Part 1: how the task's window and feature set change model performance.
    train = test = None
    if not args.skip_ablation:
        ablation, train, test = run_ablation(
            root, args.windows, args.test_fraction, args.n_boot, args.seed, args.dev
        )
        show("Part 1: task-configuration ablation", ablation, "ablation.csv")

    # Part 2: real vs. synthetic training data, on the task's default setting.
    synthetic = {}
    if args.synthetic_dir:
        # Use the split saved by the generation script, so the rows the LLM
        # was shown can never be test rows.
        folder = Path(args.synthetic_dir).expanduser()
        train = pd.read_csv(folder / "real_train.csv")
        test = pd.read_csv(folder / "real_test.csv")
        for name in ("naive", "group"):
            path = folder / f"synthetic_{name}.csv"
            if path.exists():
                synthetic[name] = pd.read_csv(path)
    elif train is None or args.windows[-1] != 24.0:
        table = build_table(root, DEFAULT_FEATURES, 24.0, args.dev)
        train, test = split_table_by_patient(table, args.test_fraction, args.seed)
    results = run_real_vs_synthetic(
        train,
        test,
        synthetic,
        n_draws=args.n_draws,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    show("Part 2: real vs. synthetic training data", results, "results.csv")


if __name__ == "__main__":
    main()
