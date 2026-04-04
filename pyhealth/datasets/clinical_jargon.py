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
    pieces = re.split(r"\s+or\s+", answer.strip())
    aliases = [piece.strip() for piece in pieces if piece.strip()]
    return aliases or [answer.strip()]


def surface_form_group(abbreviation: str) -> str:
    if any(character.isdigit() or not character.isalpha() for character in abbreviation):
        return "digit_or_symbol"
    if abbreviation.isupper():
        return "all_caps"
    if abbreviation.islower():
        return "lowercase"
    return "mixed_case"


def strip_medlingo_oneshot(question: str) -> str:
    if question.startswith(MEDLINGO_ONESHOT_PREFIX):
        return question[len(MEDLINGO_ONESHOT_PREFIX) :].strip()
    return question.strip()


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def token_length(text: str) -> int:
    return len(re.findall(r"\w+", text))


def choose_medlingo_distractors(
    records: list[dict],
    current_record: dict,
    distractor_count: int = 3,
) -> list[str]:
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
    """Public clinical jargon benchmark dataset for PyHealth."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "clinical_jargon.yaml"
        normalized_csv = root_path / "clinical_jargon_examples.csv"
        if not normalized_csv.exists():
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
        if destination.exists():
            return destination.read_text()
        payload = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
        destination.write_text(payload)
        return payload

    @classmethod
    def _fetch_medlingo_rows(cls, cache_dir: Path) -> list[dict]:
        csv_text = cls._download_text(MEDLINGO_URL, cache_dir / "medlingo_questions.csv")
        return list(csv.DictReader(csv_text.splitlines()))

    @classmethod
    def _fetch_casi_rows(cls, cache_dir: Path) -> list[dict]:
        index_path = cache_dir / "casi_release_index.json"
        if index_path.exists():
            entries = json.loads(index_path.read_text())
        else:
            entries = json.loads(
                urllib.request.urlopen(CASI_RELEASE_INDEX_URL)
                .read()
                .decode("utf-8", errors="replace")
            )
            index_path.write_text(json.dumps(entries, indent=2))
        rows: list[dict] = []
        for entry in entries:
            file_name = entry["name"]
            file_text = cls._download_text(entry["download_url"], cache_dir / file_name)
            for row in csv.DictReader(file_text.splitlines()):
                row["source_file"] = file_name
                rows.append(row)
        return rows

    @classmethod
    def prepare_metadata(cls, root: Path) -> None:
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
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(normalized_rows[0].keys()))
            writer.writeheader()
            writer.writerows(normalized_rows)

    @property
    def default_task(self):
        from ..tasks import ClinicalJargonVerification

        return ClinicalJargonVerification()
