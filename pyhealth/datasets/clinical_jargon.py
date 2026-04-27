"""Clinical jargon benchmark dataset for PyHealth.

This module exposes a public clinical jargon benchmark derived from the
released MedLingo and CASI assets from Jia et al. (CHIL 2025). The dataset is
normalized into a single CSV file that PyHealth can load as an `examples`
table.
"""

import csv
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset


MEDLINGO_URL = (
    "https://raw.githubusercontent.com/Flora-jia-jfr/diagnosing_our_datasets/"
    "main/datasets/MedLingo/questions.csv"
)
CASI_RELEASE_INDEX_URL = (
    "https://api.github.com/repos/Flora-jia-jfr/diagnosing_our_datasets/contents/"
    "datasets/casi/cleaned_dataset_subset?ref=main"
)
MEDLINGO_ONESHOT_PREFIX = (
    "In a clinical note that mentions a high creat, creat stands for creatine. "
)
DOWNLOAD_TIMEOUT_SECONDS = 10
PAPER59_EXCLUSIONS = frozenset(
    {
        ("AB", "blood group in ABO system"),
        ("US", "United States"),
        ("IB", "international baccalaureate"),
        ("MS", "master of science"),
        ("MP", "military police"),
        ("PD", "police department"),
        ("MP", "metatarsophalangeal/metacarpophalangeal"),
        ("OP", "oblique presentation/occiput posterior"),
        ("SA", "slow acting/sustained action"),
        ("C&S", "conjunctivae and sclerae"),
        ("C&S", "culture and sensitivity"),
        ("C&S", "protein C and protein S"),
    }
)


def split_aliases(answer: str) -> list[str]:
    """Split released answer strings into canonical aliases.

    Args:
        answer: The released answer string. Some MedLingo answers contain
            multiple acceptable expansions joined by ``or``.

    Returns:
        A non-empty list of acceptable expansion strings.
    """
    pieces = re.split(r"\s+or\s+", answer.strip())
    aliases = [piece.strip() for piece in pieces if piece.strip()]
    return aliases or [answer.strip()]


def surface_form_group(abbreviation: str) -> str:
    """Assign a surface-form bucket to a jargon token.

    Args:
        abbreviation: The shorthand token being evaluated.

    Returns:
        One of ``all_caps``, ``lowercase``, ``mixed_case``, or
        ``digit_or_symbol``.
    """
    if any(character.isdigit() or not character.isalpha() for character in abbreviation):
        return "digit_or_symbol"
    if abbreviation.isupper():
        return "all_caps"
    if abbreviation.islower():
        return "lowercase"
    return "mixed_case"


def strip_medlingo_oneshot(question: str) -> str:
    """Remove the released MedLingo one-shot demonstration when present.

    Args:
        question: A released MedLingo question string.

    Returns:
        The same question without the built-in one-shot example prefix.
    """
    if question.startswith(MEDLINGO_ONESHOT_PREFIX):
        return question[len(MEDLINGO_ONESHOT_PREFIX) :].strip()
    return question.strip()


def dedupe(values: list[str]) -> list[str]:
    """Preserve order while removing duplicate strings.

    Args:
        values: Ordered candidate strings.

    Returns:
        The input values with duplicates removed in first-seen order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def token_length(text: str) -> int:
    """Count alphanumeric tokens in a string.

    Args:
        text: The input text.

    Returns:
        The number of regex word tokens in ``text``.
    """
    return len(re.findall(r"\w+", text))


def choose_medlingo_distractors(
    records: list[dict],
    current_record: dict,
    distractor_count: int = 3,
) -> list[str]:
    """Select distractor expansions for a MedLingo item.

    The ranking favors candidate expansions with similar token length, first
    within the same surface-form group and then globally if more negatives are
    needed.

    Args:
        records: All normalized MedLingo records.
        current_record: The record whose distractors are being chosen.
        distractor_count: Number of negative candidates to return.

    Returns:
        A list of distractor expansions ordered from closest to farthest match.
    """
    gold = current_record["gold_expansion"]
    goal_length = token_length(gold)

    def rank(pool: list[str]) -> list[str]:
        return sorted(
            dedupe([value for value in pool if value != gold]),
            key=lambda value: (abs(token_length(value) - goal_length), value.lower()),
        )

    same_group = [
        record["gold_expansion"]
        for record in records
        if record["sample_id"] != current_record["sample_id"]
        and record["surface_form_group"] == current_record["surface_form_group"]
    ]
    global_pool = [
        record["gold_expansion"]
        for record in records
        if record["sample_id"] != current_record["sample_id"]
    ]
    negatives = rank(same_group)
    if len(negatives) < distractor_count:
        for candidate in rank(global_pool):
            if candidate not in negatives:
                negatives.append(candidate)
            if len(negatives) == distractor_count:
                break
    return negatives[:distractor_count]


class ClinicalJargonDataset(BaseDataset):
    """Public clinical jargon benchmark dataset for PyHealth.

    The dataset downloads the public MedLingo and CASI benchmark assets,
    normalizes them into a single ``clinical_jargon_examples.csv`` file, and
    exposes the result through the PyHealth dataset API.

    The default task is :class:`pyhealth.tasks.ClinicalJargonVerification`,
    which converts each benchmark item into paired-text binary verification
    samples over candidate expansions.

    Args:
        root: Root directory used to store normalized benchmark files.
        dataset_name: Optional dataset name. Defaults to ``clinical_jargon``.
        config_path: Optional path to the dataset config file.
        download: Whether to download and normalize the public source assets
            when ``clinical_jargon_examples.csv`` is missing. Defaults to
            ``False``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import ClinicalJargonDataset
        >>> dataset = ClinicalJargonDataset(
        ...     root="/tmp/clinical_jargon",
        ...     download=True,
        ... )
        >>> task = dataset.default_task
        >>> samples = dataset.set_task(task)
        >>> print(samples[0]["paired_text"])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the public clinical jargon dataset.

        Args:
            root: Root directory used to cache normalized files.
            dataset_name: Optional dataset name override.
            config_path: Optional dataset config path override.
            download: Whether to fetch and normalize the released benchmark
                assets when the normalized CSV is missing.
            **kwargs: Additional keyword arguments passed to ``BaseDataset``.
        """
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "clinical_jargon.yaml"
        normalized_csv = root_path / "clinical_jargon_examples.csv"
        if not normalized_csv.exists():
            if not download:
                raise FileNotFoundError(
                    f"Missing normalized metadata at {normalized_csv}. "
                    "Pass download=True to fetch the public MedLingo and CASI "
                    "assets and generate this CSV."
                )
            self.prepare_metadata(root_path)
        super().__init__(
            root=str(root_path),
            tables=["examples"],
            dataset_name=dataset_name or "clinical_jargon",
            config_path=str(config_path),
            **kwargs,
        )

    @staticmethod
    def _download_text(url: str, destination: Path) -> str:
        """Download text content unless it is already cached locally.

        Args:
            url: Source URL.
            destination: Cache path for the downloaded content.

        Returns:
            The downloaded or cached text payload.
        """
        if destination.exists():
            return destination.read_text(encoding="utf-8", errors="replace")
        request = urllib.request.Request(url)
        with urllib.request.urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            payload = response.read().decode("utf-8", errors="replace")
        destination.write_text(payload, encoding="utf-8", errors="replace")
        return payload

    @staticmethod
    def _validated_file_name(file_name: str) -> str:
        """Validate a remotely provided cache filename."""
        candidate = Path(file_name)
        if (
            not file_name
            or candidate.is_absolute()
            or candidate.name != file_name
            or candidate.parent != Path(".")
        ):
            raise ValueError(f"Invalid cache file name: {file_name}")
        return file_name

    @classmethod
    def _fetch_medlingo_rows(cls, cache_dir: Path) -> list[dict]:
        """Load raw MedLingo rows from the released public CSV.

        Args:
            cache_dir: Cache directory for downloaded assets.

        Returns:
            Raw MedLingo rows as dictionaries.
        """
        csv_text = cls._download_text(MEDLINGO_URL, cache_dir / "medlingo_questions.csv")
        return list(csv.DictReader(csv_text.splitlines()))

    @classmethod
    def _fetch_casi_rows(cls, cache_dir: Path) -> list[dict]:
        """Load raw CASI rows from the released public subset.

        Args:
            cache_dir: Cache directory for downloaded assets.

        Returns:
            Raw CASI rows as dictionaries with source-file metadata.
        """
        index_path = cache_dir / "casi_release_index.json"
        entries = json.loads(cls._download_text(CASI_RELEASE_INDEX_URL, index_path))
        rows: list[dict] = []
        for entry in entries:
            file_name = cls._validated_file_name(entry["name"])
            file_text = cls._download_text(entry["download_url"], cache_dir / file_name)
            for row in csv.DictReader(file_text.splitlines()):
                row["source_file"] = file_name
                rows.append(row)
        return rows

    @classmethod
    def prepare_metadata(cls, root: Path) -> None:
        """Normalize public MedLingo and CASI assets into one CSV file.

        Args:
            root: Root directory where the normalized file and cache should be
                written.
        """
        cache_dir = root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        medlingo_rows: list[dict] = []
        for index, row in enumerate(cls._fetch_medlingo_rows(cache_dir), start=1):
            aliases = split_aliases(row["answer"])
            medlingo_rows.append(
                {
                    "sample_id": f"medlingo_{index:03d}",
                    "benchmark": "medlingo",
                    "abbreviation": row["word1"],
                    "context": "",
                    "question": row["question"].strip(),
                    "question_zero_shot": strip_medlingo_oneshot(row["question"]),
                    "gold_expansion": aliases[0],
                    "gold_aliases_json": json.dumps(aliases),
                    "surface_form_group": surface_form_group(row["word1"]),
                    "paper59_included": "true",
                    "source_file": "medlingo_questions.csv",
                }
            )

        medlingo_candidate_map = {}
        for row in medlingo_rows:
            negatives = choose_medlingo_distractors(medlingo_rows, row, distractor_count=3)
            medlingo_candidate_map[row["sample_id"]] = [row["gold_expansion"], *negatives]

        for row in medlingo_rows:
            row["candidate_expansions_json"] = json.dumps(
                medlingo_candidate_map[row["sample_id"]]
            )
            row["candidate_expansions_paper59_json"] = row["candidate_expansions_json"]

        casi_rows = cls._fetch_casi_rows(cache_dir)
        release_expansions: dict[str, list[str]] = defaultdict(list)
        paper59_expansions: dict[str, list[str]] = defaultdict(list)
        for row in casi_rows:
            abbreviation = row["sf"]
            expansion = row["target_lf"]
            release_expansions[abbreviation].append(expansion)
            if (abbreviation, expansion) not in PAPER59_EXCLUSIONS:
                paper59_expansions[abbreviation].append(expansion)

        release_expansions = {
            key: dedupe(sorted(values, key=str.lower))
            for key, values in release_expansions.items()
        }
        paper59_expansions = {
            key: dedupe(sorted(values, key=str.lower))
            for key, values in paper59_expansions.items()
        }

        normalized_rows = list(medlingo_rows)
        for index, row in enumerate(casi_rows, start=1):
            abbreviation = row["sf"]
            expansion = row["target_lf"]
            question = f"{row['context'].strip()} In this sentence, {abbreviation} means:"
            normalized_rows.append(
                {
                    "sample_id": f"casi_{index:04d}",
                    "benchmark": "casi",
                    "abbreviation": abbreviation,
                    "context": row["context"].strip(),
                    "question": question,
                    "question_zero_shot": question,
                    "gold_expansion": expansion,
                    "gold_aliases_json": json.dumps([expansion]),
                    "surface_form_group": surface_form_group(abbreviation),
                    "paper59_included": str(
                        (abbreviation, expansion) not in PAPER59_EXCLUSIONS
                    ).lower(),
                    "source_file": row["source_file"],
                    "candidate_expansions_json": json.dumps(
                        release_expansions[abbreviation]
                    ),
                    "candidate_expansions_paper59_json": json.dumps(
                        paper59_expansions.get(abbreviation, [])
                    ),
                }
            )

        output_path = root / "clinical_jargon_examples.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(normalized_rows[0].keys()))
            writer.writeheader()
            writer.writerows(normalized_rows)

    @property
    def default_task(self):
        """Return the default task for the dataset.

        Returns:
            ClinicalJargonVerification: The default binary verification task.
        """
        from ..tasks import ClinicalJargonVerification

        return ClinicalJargonVerification()
