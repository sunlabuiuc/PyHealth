"""Generate synthetic eICU mortality records with an LLM.

Contributors:
    Alex Castillo (acast23@illinois.edu)
    Srikar Lunke (srikarl2@illinois.edu)
    Maheswara Vatti (mvatti2@illinois.edu)

Paper:
    Lin, Yu and Lee, "A Case Study Exploring the Current Landscape of
    Synthetic Medical Record Generation with Commercial LLMs", CHIL 2025
    (PMLR 287), https://proceedings.mlr.press/v287/lin25a.html

AI assistance:
    Generated with Claude Fable 5.2 (Anthropic).

This script reproduces two of the paper's prompting strategies on the table
that :class:`~pyhealth.tasks.TabularMortalityPredictionEICU` builds (one row
per ICU stay: ten early features, sex, ethnicity and in-ICU mortality):

- ``naive``: the LLM sees a small sample of real rows and is asked for more
  rows like them.
- ``group``: the same, but each request names one sex and ethnicity group and
  every group gets the same number of rows.

What it does:

1. Builds the real table from eICU and splits it by patient into a train pool
   and a held-out test set (``real_train.csv`` and ``real_test.csv``).
2. Shows the LLM fewer than 100 rows drawn from the train pool only, with
   deaths at the pool's own rate (a plain random draw of so few rows can hold
   almost none, and the LLM then generates survivors only).
3. Asks for rows in batches, parses the replies, and drops rows that are
   malformed, out of range, outside the requested group, or copies of a row
   the LLM was shown.
4. Writes ``synthetic_naive.csv`` and ``synthetic_group.csv``.

The prompts are adapted from the paper's Appendix A.1 to this table. Each
request also spells out which properties of the sample to keep (the share of
deaths, the spread of each column and the share of missing values), because a
small local model otherwise returns only typical survivors.

Data use:
    The prompts contain real patient rows. eICU-CRD is credentialed data:
    send it only to a model running on your own machine, or to a service you
    have verified keeps no data and trains on none. Keep the output folder
    outside the repository and never commit its files. The demo bundled with
    PyHealth is open data, so it is used when ``--root`` is not given.

The LLM is any OpenAI-compatible chat endpoint, set by environment variables:
    ``LLM_BASE_URL`` (default ``http://localhost:11434/v1``, a local Ollama),
    ``LLM_MODEL`` (for example ``qwen2.5:14b``) and ``LLM_API_KEY`` (optional).

Usage:
    Print the prompts without calling an LLM (runs on the bundled demo)::

        python examples/eicu_tabular_mortality_llm_generation.py \\
            --output-dir /tmp/eicu_synthetic --dry-run

    Generate 1,000 rows per strategy from full eICU with a local model::

        LLM_MODEL=qwen2.5:14b \\
        python examples/eicu_tabular_mortality_llm_generation.py \\
            --root /path/to/eicu-crd/2.0 --output-dir ~/eicu_synthetic
"""

import argparse
import csv
import io
import json
import math
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.tasks import TabularMortalityPredictionEICU

FEATURES = list(TabularMortalityPredictionEICU.DEFAULT_FEATURES)
LABEL = "mortality"
GROUP_COLUMNS = ["gender", "ethnicity"]
# Columns of every real and synthetic row, in order.
COLUMNS = [*FEATURES, LABEL, *GROUP_COLUMNS]
DEMO_ROOT = Path(__file__).resolve().parents[1] / "test-resources/core/eicudemo"

# Loose physiological limits; a generated value outside them drops the row.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "lactate_last_early": (0.0, 50.0),  # mmol/L
    "bun_last_early": (0.0, 300.0),  # mg/dL
    "age": (0.0, 90.0),  # years; eICU caps age at 90
    "albumin_first_early": (0.0, 10.0),  # g/dL
    "bilirubin_last_early": (0.0, 100.0),  # mg/dL
    "wbc_last_early": (0.0, 500.0),  # K/mcL
    "bicarbonate_last_early": (0.0, 80.0),  # mmol/L
    "creatinine_last_early": (0.0, 50.0),  # mg/dL
    "platelet_first_early": (0.0, 3000.0),  # K/mcL
    "bg_ph_last_early": (6.0, 8.0),
}
_MISSING = {"", "nan", "na", "n/a", "null", "none"}

NAIVE_PROMPT = """\
You are an AI model that generates realistic synthetic Electronic Health \
Records (EHR) while protecting patient privacy.

Study the sample below, which is a structured extract of the eICU dataset. \
Generate synthetic patient records that keep the statistical and structural \
properties of the original data and do not copy any real patient's record.

Return the synthetic records as CSV with the same columns as the sample."""

GROUP_PROMPT = """\
You are an AI model that generates realistic synthetic Electronic Health \
Records (EHR) with a group-based strategy: the records must keep the \
statistical properties of each demographic group (sex and race) and stay \
coherent within the group.

Process:
1. Start from the group named in the request.
2. Generate each feature in turn, conditioned on that group.
3. Keep group-specific medical patterns: disease risk, lab values and \
treatment patterns should vary between groups the way they do clinically.

For example, for a 65-year-old woman, laboratory values should follow the \
distributions seen in older female patients.

Return the synthetic records as CSV with the same columns as the sample. \
They must reflect realistic group-level trends without bias or inconsistency."""

PROMPTS = {"naive": NAIVE_PROMPT, "group": GROUP_PROMPT}


def build_real_table(
    root: str, window_hours: float = 24.0, dev: bool = False
) -> pd.DataFrame:
    """Runs the task on eICU and flattens its samples to one row per ICU stay.

    Args:
        root: Folder with the eICU CSV files (``.csv`` or ``.csv.gz``).
        window_hours: Early window passed to the task.
        dev: Load only the first 1,000 patients, for a quick trial run.

    Returns:
        A table with ``patient_id``, ``stay_id`` and :data:`COLUMNS`.
    """
    from pyhealth.datasets import eICUDataset

    dataset = eICUDataset(root=root, tables=["lab"], dev=dev)
    samples = dataset.set_task(
        TabularMortalityPredictionEICU(features=FEATURES, window_hours=window_hours)
    )
    rows = []
    for index in range(len(samples)):
        sample = samples[index]
        row = {"patient_id": sample["patient_id"], "stay_id": int(sample["stay_id"])}
        row.update(zip(FEATURES, sample["features"].tolist()))
        row[LABEL] = int(float(sample[LABEL]))
        row.update({column: sample[column] for column in GROUP_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=["patient_id", "stay_id", *COLUMNS])


def split_by_patient(
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


def read_table(path: str | Path) -> pd.DataFrame:
    """Reads a real or synthetic table written by this script.

    Args:
        path: CSV file with at least :data:`COLUMNS`.

    Returns:
        The table, with empty group cells as empty strings.
    """
    table = pd.read_csv(path, dtype={column: str for column in GROUP_COLUMNS})
    table[GROUP_COLUMNS] = table[GROUP_COLUMNS].fillna("")
    return table


def to_csv_text(table: pd.DataFrame) -> str:
    """Formats rows as the CSV text shown to the LLM (empty cell = missing).

    Args:
        table: Rows holding :data:`COLUMNS`.

    Returns:
        CSV text with a header line.
    """
    return table[COLUMNS].to_csv(index=False, float_format="%.6g").strip()


def build_messages(
    strategy: str,
    examples: pd.DataFrame,
    n_rows: int,
    group: tuple[str, str] | None = None,
) -> list[dict[str, str]]:
    """Builds the chat messages of one generation request.

    Args:
        strategy: ``"naive"`` or ``"group"``.
        examples: Real rows shown to the LLM.
        n_rows: Number of rows to ask for.
        group: The (gender, ethnicity) to generate, for the group strategy.

    Returns:
        A system and a user message.
    """
    request = f"Generate {n_rows} new synthetic records"
    if group is not None:
        request += f" for the group gender={group[0]}, ethnicity={group[1]}"
    user = (
        f"Dataset sample (CSV, an empty cell is a missing value):\n\n"
        f"{to_csv_text(examples)}\n\n"
        f"{request}. Reply with CSV rows only, no header and no commentary, "
        f"in this column order: {','.join(COLUMNS)}. {LABEL} is 1 if the "
        f"patient died in the ICU and 0 otherwise. Leave a cell empty where "
        f"the value would be missing. Match the sample's statistics: about "
        f"the same share of rows with {LABEL}=1, the same spread in every "
        f"column, including unusually low and high values, and about the same "
        f"share of empty cells per column."
    )
    return [
        {"role": "system", "content": PROMPTS[strategy]},
        {"role": "user", "content": user},
    ]


def sample_examples(
    train: pd.DataFrame, n_examples: int, seed: int = 0
) -> pd.DataFrame:
    """Draws the real rows shown to the LLM, with deaths at the pool's rate.

    Args:
        train: The train pool.
        n_examples: Number of rows to draw (fewer if the pool is smaller).
        seed: Seed of the draw.

    Returns:
        The rows, shuffled. If the pool holds both outcomes, so do the rows:
        the number of deaths is the pool's rate times ``n_examples``, rounded,
        and at least one.
    """
    n_examples = min(n_examples, len(train))
    died = train[train[LABEL] == 1]
    survived = train[train[LABEL] != 1]
    n_died = min(len(died), max(1, round(n_examples * len(died) / len(train))))
    if len(survived) and n_examples > 1:
        n_died = min(n_died, n_examples - 1)
    n_survived = min(len(survived), n_examples - n_died)
    examples = pd.concat(
        [
            died.sample(n=n_died, random_state=seed),
            survived.sample(n=n_survived, random_state=seed),
        ]
    )
    return examples.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def list_groups(examples: pd.DataFrame) -> list[tuple[str, str]]:
    """Lists the (gender, ethnicity) groups present in the real sample.

    Args:
        examples: Real rows shown to the LLM.

    Returns:
        The groups with both values known, sorted.

    Raises:
        ValueError: If no row has both a gender and an ethnicity.
    """
    pairs = examples[GROUP_COLUMNS].drop_duplicates()
    groups = sorted(
        (gender, ethnicity)
        for gender, ethnicity in pairs.itertuples(index=False)
        if gender and ethnicity
    )
    if not groups:
        raise ValueError("No example row has both a gender and an ethnicity.")
    return groups


def make_llm(
    base_url: str,
    model: str,
    api_key: str = "",
    temperature: float | None = None,
    timeout: float = 600.0,
) -> Callable[[list[dict[str, str]]], str]:
    """Returns a function that sends chat messages to an OpenAI-compatible API.

    Args:
        base_url: API root, for example ``http://localhost:11434/v1``.
        model: Model name.
        api_key: Bearer token, if the endpoint needs one.
        temperature: Sampling temperature; left to the endpoint if ``None``.
        timeout: Seconds to wait for one reply.

    Returns:
        A function from chat messages to the reply text.
    """

    def chat(messages: list[dict[str, str]]) -> str:
        payload: dict[str, object] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.load(response)
        return reply["choices"][0]["message"]["content"]

    return chat


def _row_key(row: dict[str, object]) -> tuple[str, ...]:
    """Returns a row's values as text, for spotting copies of real rows."""
    return tuple(
        format(value, ".6g") if isinstance(value, float) else str(value)
        for value in (row[column] for column in COLUMNS)
    )


def parse_rows(
    text: str,
    genders: set[str],
    ethnicities: set[str],
    group: tuple[str, str] | None = None,
    shown: set[tuple[str, ...]] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """Parses an LLM reply into valid rows.

    A line is dropped if it does not have one field per column, a feature is
    not a number or is outside :data:`VALID_RANGES`, the label is not 0 or 1,
    the sex or ethnicity is not one seen in the real sample (or is not the
    requested ``group``), or the row copies a row the LLM was shown.

    Args:
        text: The reply, CSV lines possibly wrapped in other text.
        genders: Sex values accepted.
        ethnicities: Ethnicity values accepted.
        group: The (gender, ethnicity) every row must have, if any.
        shown: Keys (see ``_row_key``) of the rows shown to the LLM.

    Returns:
        The valid rows and the number of dropped lines. Header, blank and
        code-fence lines count as neither.

    Examples:
        >>> rows, dropped = parse_rows(
        ...     "1.5,20,61,3.2,0.8,9,24,1.1,200,7.4,0,Male,Asian\\nnot a row",
        ...     genders={"Male"},
        ...     ethnicities={"Asian"},
        ... )
        >>> rows[0]["age"], rows[0]["mortality"], dropped
        (61.0, 0, 1)
    """
    rows: list[dict[str, object]] = []
    dropped = 0
    for fields in csv.reader(io.StringIO(text)):
        fields = [field.strip() for field in fields]
        if not any(fields) or fields[0].startswith("```") or fields == COLUMNS:
            continue
        row = _validate(fields, genders, ethnicities, group)
        if row is None or (shown and _row_key(row) in shown):
            dropped += 1
        else:
            rows.append(row)
    return rows, dropped


def _validate(
    fields: list[str],
    genders: set[str],
    ethnicities: set[str],
    group: tuple[str, str] | None,
) -> dict[str, object] | None:
    """Converts one CSV line to a row, or returns ``None`` if it is invalid."""
    if len(fields) != len(COLUMNS):
        return None
    raw = dict(zip(COLUMNS, fields))
    row: dict[str, object] = {}
    for name in FEATURES:
        if raw[name].lower() in _MISSING:
            row[name] = math.nan
            continue
        try:
            value = float(raw[name])
        except ValueError:
            return None
        low, high = VALID_RANGES[name]
        if not low <= value <= high:
            return None
        row[name] = value
    if raw[LABEL] not in {"0", "1", "0.0", "1.0"}:
        return None
    row[LABEL] = int(float(raw[LABEL]))
    gender, ethnicity = raw["gender"], raw["ethnicity"]
    if group is not None and (gender, ethnicity) != group:
        return None
    if gender not in genders or ethnicity not in ethnicities:
        return None
    row["gender"], row["ethnicity"] = gender, ethnicity
    return row


def generate(
    strategy: str,
    examples: pd.DataFrame,
    n_rows: int,
    batch_size: int,
    llm: Callable[[list[dict[str, str]]], str],
    seed: int = 0,
) -> pd.DataFrame:
    """Generates synthetic rows with one prompting strategy.

    The group strategy splits ``n_rows`` evenly over the (gender, ethnicity)
    groups found in ``examples``. Requests are repeated until each target is
    met, up to three times the number of requests a perfect LLM would need.

    Args:
        strategy: ``"naive"`` or ``"group"``.
        examples: Real rows shown to the LLM.
        n_rows: Number of rows wanted.
        batch_size: Rows asked for per request.
        llm: Function from chat messages to the reply text.
        seed: Seed of the final shuffle.

    Returns:
        Up to ``n_rows`` valid rows with :data:`COLUMNS`; fewer if the LLM
        kept returning invalid rows.
    """
    genders = set(examples["gender"])
    ethnicities = set(examples["ethnicity"])
    shown = {_row_key(row) for row in examples[COLUMNS].to_dict("records")}
    targets: list[tuple[tuple[str, str] | None, int]] = [(None, n_rows)]
    if strategy == "group":
        groups = list_groups(examples)
        targets = [(group, math.ceil(n_rows / len(groups))) for group in groups]

    rows: list[dict[str, object]] = []
    for group, target in targets:
        kept: list[dict[str, object]] = []
        requests_left = 3 * math.ceil(target / batch_size)
        while len(kept) < target and requests_left > 0:
            requests_left -= 1
            # Ask for a few spare rows near the end: some will be dropped.
            ask = min(batch_size, max(target - len(kept), 5))
            reply = llm(build_messages(strategy, examples, ask, group))
            valid, dropped = parse_rows(reply, genders, ethnicities, group, shown)
            kept.extend(valid)
            label = "all" if group is None else "/".join(group)
            print(
                f"[{strategy}] {label}: +{len(valid)} valid, {dropped} dropped "
                f"({min(len(kept), target)}/{target})"
            )
        rows.extend(kept[:target])

    table = pd.DataFrame(rows, columns=COLUMNS)
    table = table.sample(frac=1.0, random_state=seed).head(n_rows)
    if len(table) < n_rows:
        print(f"[{strategy}] warning: only {len(table)} of {n_rows} rows are valid.")
    return table.reset_index(drop=True)


def main() -> None:
    """Parses the command line and runs the generation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", help="eICU folder; the bundled demo if omitted.")
    parser.add_argument(
        "--train-csv",
        help="Use this table as the train pool instead of building it from eICU.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the CSV files; keep it outside the repository.",
    )
    parser.add_argument(
        "--strategy", choices=["naive", "group", "both"], default="both"
    )
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument(
        "--n-examples", type=int, default=99, help="Real rows shown to the LLM (<100)."
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument(
        "--dev", action="store_true", help="Load only 1,000 patients (trial run)."
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the prompts; call no LLM."
    )
    args = parser.parse_args()
    if not 1 <= args.n_examples < 100:
        parser.error("--n-examples must be between 1 and 99.")
    if args.n_rows < 1 or args.batch_size < 1:
        parser.error("--n-rows and --batch-size must be at least 1.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.train_csv:
        train = read_table(args.train_csv)
    else:
        table = build_real_table(
            args.root or str(DEMO_ROOT), args.window_hours, args.dev
        )
        train, test = split_by_patient(table, args.test_fraction, args.seed)
        train.to_csv(output_dir / "real_train.csv", index=False, float_format="%.6g")
        test.to_csv(output_dir / "real_test.csv", index=False, float_format="%.6g")
        print(
            f"Real table: {len(table)} stays, mortality {table[LABEL].mean():.2%}; "
            f"{len(train)} train, {len(test)} held out."
        )
        train = read_table(output_dir / "real_train.csv")
    examples = sample_examples(train, args.n_examples, args.seed)
    print(
        f"Showing the LLM {len(examples)} real rows, "
        f"{int(examples[LABEL].sum())} of them deaths."
    )

    strategies = ["naive", "group"] if args.strategy == "both" else [args.strategy]
    if args.dry_run:
        for strategy in strategies:
            group = list_groups(examples)[0] if strategy == "group" else None
            messages = build_messages(strategy, examples, args.batch_size, group)
            print(f"\n===== {strategy} prompt =====")
            for message in messages:
                print(f"--- {message['role']} ---\n{message['content']}")
        return

    model = os.environ.get("LLM_MODEL")
    if not model:
        parser.error("Set the LLM_MODEL environment variable (or use --dry-run).")
    llm = make_llm(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=model,
        api_key=os.environ.get("LLM_API_KEY", ""),
        temperature=args.temperature,
    )
    for strategy in strategies:
        synthetic = generate(
            strategy, examples, args.n_rows, args.batch_size, llm, args.seed
        )
        path = output_dir / f"synthetic_{strategy}.csv"
        synthetic.to_csv(path, index=False, float_format="%.6g")
        print(
            f"[{strategy}] wrote {len(synthetic)} rows to {path} "
            f"(mortality {synthetic[LABEL].mean():.2%})."
        )


if __name__ == "__main__":
    main()
