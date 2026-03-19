"""Dataset loader for Path-VQA style medical visual question answering data."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _find_first(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _normalize_split(split_value: Optional[str]) -> str:
    if split_value is None:
        return "unspecified"
    split = str(split_value).strip().lower()
    if split in {"val", "valid", "dev"}:
        return "validation"
    if split in {"train", "training"}:
        return "train"
    if split in {"test", "testing"}:
        return "test"
    return split


def _resolve_image_path(raw_path: Any, images_dir: Optional[Path]) -> str:
    if raw_path is None:
        raise ValueError("missing image path")

    if isinstance(raw_path, dict):
        raw_path = raw_path.get("path") or raw_path.get("src")

    if raw_path is None:
        raise ValueError("missing image path")

    path = Path(str(raw_path))
    if path.is_absolute() or images_dir is None:
        return str(path)
    return str((images_dir / path).resolve())


def _read_annotation_records(annotation_path: Path) -> List[Dict[str, Any]]:
    suffix = annotation_path.suffix.lower()

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(annotation_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    if suffix == ".json":
        with open(annotation_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            records: List[Dict[str, Any]] = []
            for split_name, split_records in payload.items():
                if isinstance(split_records, list):
                    for record in split_records:
                        if isinstance(record, dict):
                            row = dict(record)
                            row.setdefault("split", split_name)
                            records.append(row)
            if records:
                return records
            return [payload]

        raise ValueError(f"Unsupported JSON annotation structure: {annotation_path}")

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(annotation_path, sep=sep).to_dict("records")

    raise ValueError(
        "Unsupported annotation file type. Expected one of "
        ".json, .jsonl, .csv, or .tsv"
    )


def _extract_row(
    record: Dict[str, Any],
    row_idx: int,
    dataset_name: str,
    images_dir: Optional[Path],
    default_split: Optional[str],
) -> Dict[str, str]:
    image_raw = _find_first(
        record,
        (
            "image",
            "image_path",
            "img_path",
            "img",
            "image_name",
            "filename",
            "file_name",
            "path",
        ),
    )
    question = _find_first(record, ("question", "query", "q"))
    answer = _find_first(record, ("answer", "label", "a"))
    split = _find_first(record, ("split", "set", "subset", "partition"))
    question_id = _find_first(record, ("question_id", "qid", "qa_id", "id"))
    image_id = _find_first(record, ("image_id", "img_id", "study_id"))

    missing_keys: List[str] = []
    if question is None:
        missing_keys.append("question")
    if answer is None:
        missing_keys.append("answer")
    if image_raw is None:
        missing_keys.append("image")

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ValueError(
            f"missing required field(s): {missing_str}. row_index={row_idx}"
        )

    image_path = _resolve_image_path(image_raw, images_dir)
    image_basename = Path(image_path).stem

    if question_id is None:
        question_id = f"{dataset_name}_q_{row_idx}"
    if image_id is None:
        image_id = image_basename

    resolved_split = _normalize_split(str(split) if split is not None else default_split)

    return {
        "path": image_path,
        "question": str(question),
        "answer": str(answer),
        "split": resolved_split,
        "question_id": str(question_id),
        "image_id": str(image_id),
        "patient_id": str(image_id),
        "dataset": dataset_name,
    }


class PathVQADataset(BaseDataset):
    """Base dataset wrapper for Path-VQA style VQA data."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
        annotation_path: Optional[str] = None,
        images_dir: Optional[str] = None,
        split: Optional[str] = None,
        clean_split_path: Optional[str] = None,
        enable_dedup: bool = False,
        dedup_method: str = "auto",
        refresh_metadata: bool = False,
        annotation_candidates: Optional[List[str]] = None,
    ) -> None:
        self.annotation_path = annotation_path
        self.images_dir = images_dir
        self.split = _normalize_split(split) if split else None
        self.clean_split_path = clean_split_path
        self.enable_dedup = enable_dedup
        self.dedup_method = dedup_method.strip().lower()
        self.refresh_metadata = refresh_metadata
        self.annotation_candidates = annotation_candidates

        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "path_vqa.yaml")

        metadata_path = root_path / "path_vqa-metadata-pyhealth.csv"
        if not metadata_path.exists() or refresh_metadata:
            self.prepare_metadata(root_path)

        super().__init__(
            root=str(root_path),
            tables=["path_vqa"],
            dataset_name=dataset_name or "PathVQADataset",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self):
        from pyhealth.tasks import GenerativeMedicalVQA

        return GenerativeMedicalVQA()

    def _detect_annotation_path(self, root: Path) -> Path:
        if self.annotation_path is not None:
            path = Path(self.annotation_path)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                raise FileNotFoundError(f"Annotation file not found: {path}")
            return path

        candidates = self.annotation_candidates or [
            "path_vqa.json",
            "path-vqa.json",
            "annotations.json",
            "train.json",
            "validation.json",
            "val.json",
            "test.json",
            "data.json",
            "data.jsonl",
            "qa_pairs.json",
            "path_vqa.csv",
            "annotations.csv",
        ]
        if self.split:
            split_prefix = [
                f"{self.split}.json",
                f"{self.split}.jsonl",
                f"{self.split}.csv",
            ]
            candidates = split_prefix + candidates

        for candidate in candidates:
            path = root / candidate
            if path.exists() and path.is_file():
                logger.info("Using Path-VQA annotation file: %s", path)
                return path

        raise FileNotFoundError(
            "Could not auto-detect Path-VQA annotations. Set annotation_path or "
            "provide annotation_candidates."
        )

    def _resolve_images_dir(self, root: Path) -> Optional[Path]:
        if self.images_dir is not None:
            path = Path(self.images_dir)
            if not path.is_absolute():
                path = root / path
            return path

        candidates = [
            root / "images",
            root / "imgs",
            root / "PathVQA images",
            root,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _load_clean_split_maps(self, root: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        if self.clean_split_path is None:
            return {}, {}

        path = Path(self.clean_split_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"clean_split_path not found: {path}")

        records: List[Dict[str, Any]] = []
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            records = pd.read_csv(path, sep=sep).to_dict("records")
        else:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                records = [
                    {"question_id": str(key), "split": str(value)}
                    for key, value in payload.items()
                ]
            elif isinstance(payload, list):
                records = [r for r in payload if isinstance(r, dict)]
            else:
                raise ValueError(
                    "Unsupported clean_split_path format. Expected CSV/TSV/JSON."
                )

        qid_map: Dict[str, str] = {}
        image_map: Dict[str, str] = {}
        for record in records:
            split = _find_first(record, ("split", "set", "subset", "partition"))
            if split is None:
                continue
            split = _normalize_split(str(split))
            question_id = _find_first(record, ("question_id", "qid", "qa_id", "id"))
            image_id = _find_first(record, ("image_id", "img_id", "study_id"))
            if question_id is not None:
                qid_map[str(question_id)] = split
            if image_id is not None:
                image_map[str(image_id)] = split

        return qid_map, image_map

    def _apply_clean_split(self, rows: List[Dict[str, str]], root: Path) -> None:
        qid_map, image_map = self._load_clean_split_maps(root)
        if not qid_map and not image_map:
            return

        for row in rows:
            question_id = row.get("question_id")
            image_id = row.get("image_id")
            if question_id in qid_map:
                row["split"] = qid_map[question_id]
            elif image_id in image_map:
                row["split"] = image_map[image_id]

    def _phash(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return hashlib.sha1(image_path.encode("utf-8")).hexdigest()

        try:
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("L").resize((8, 8))
                pixels = list(image.getdata())
            mean = sum(pixels) / max(len(pixels), 1)
            bits = "".join("1" if value > mean else "0" for value in pixels)
            return hashlib.sha1(bits.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha1(image_path.encode("utf-8")).hexdigest()

    def _dedup_with_phash(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        train_hashes = {
            self._phash(row["path"])
            for row in rows
            if row.get("split") == "train"
        }
        deduped_rows: List[Dict[str, str]] = []
        removed = 0
        for row in rows:
            if row.get("split") == "test" and self._phash(row["path"]) in train_hashes:
                removed += 1
                continue
            deduped_rows.append(row)

        if removed:
            logger.warning("Dedup removed %d likely leaked test samples", removed)
        return deduped_rows

    def _apply_dedup(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not self.enable_dedup:
            return rows

        if self.dedup_method in {"none", "off"}:
            return rows

        if self.dedup_method == "phash":
            return self._dedup_with_phash(rows)

        if self.dedup_method in {"auto", "faiss"}:
            has_faiss = importlib.util.find_spec("faiss") is not None
            if not has_faiss:
                warnings.warn(
                    "FAISS is unavailable. Skipping dedup in MVP mode. "
                    "Set dedup_method='phash' for deterministic fallback.",
                    UserWarning,
                    stacklevel=2,
                )
                return rows

            warnings.warn(
                "FAISS dedup hook is available but not implemented in this MVP. "
                "Skipping dedup.",
                UserWarning,
                stacklevel=2,
            )
            return rows

        raise ValueError(
            "Invalid dedup_method. Expected one of: auto, faiss, phash, none."
        )

    def _validate_rows(self, rows: List[Dict[str, str]]) -> None:
        missing_paths: List[str] = []
        required_keys = {"path", "question", "answer", "question_id", "image_id"}

        for index, row in enumerate(rows):
            missing_keys = sorted(key for key in required_keys if key not in row)
            if missing_keys:
                missing = ", ".join(missing_keys)
                raise ValueError(
                    f"Metadata row {index} is missing required keys: {missing}"
                )

            row_path = str(row["path"])
            if row_path.startswith(("http://", "https://")):
                continue
            if not Path(row_path).exists():
                missing_paths.append(row_path)

        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            suffix = "\n..." if len(missing_paths) > 10 else ""
            raise FileNotFoundError(
                "Found metadata rows with non-existent image paths:\n"
                f"{preview}{suffix}"
            )

    def prepare_metadata(self, root: Path) -> None:
        if self.clean_split_path is None and not self.enable_dedup:
            warnings.warn(
                "No clean_split_path and dedup disabled. Path-VQA split may contain "
                "near-duplicate leakage; results can be optimistic.",
                UserWarning,
                stacklevel=2,
            )

        annotation_path = self._detect_annotation_path(root)
        images_dir = self._resolve_images_dir(root)

        records = _read_annotation_records(annotation_path)
        rows: List[Dict[str, str]] = []
        for row_idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            try:
                row = _extract_row(
                    record=record,
                    row_idx=row_idx,
                    dataset_name="path_vqa",
                    images_dir=images_dir,
                    default_split=self.split,
                )
            except ValueError as exc:
                logger.warning("Skipping malformed Path-VQA row %d: %s", row_idx, exc)
                continue
            rows.append(row)

        self._apply_clean_split(rows, root)

        if self.split:
            rows = [row for row in rows if row.get("split") == self.split]

        rows = self._apply_dedup(rows)

        if not rows:
            raise ValueError("No valid Path-VQA records found after preprocessing")

        self._validate_rows(rows)

        metadata = pd.DataFrame(rows)
        metadata_path = root / "path_vqa-metadata-pyhealth.csv"
        metadata.to_csv(metadata_path, index=False)
        logger.info("Saved Path-VQA metadata to %s", metadata_path)
