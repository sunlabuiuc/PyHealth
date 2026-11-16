"""Authors: Aritra Saha and Ugochukwu Njoku.

Identify benchmark fits and verify saved run records and artifacts.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyhealth

SPLITS = ("train", "valid", "test")


def input_hashes(
    root: Path | None, released_dir: Path | None, dataset: str, split: int
) -> dict[str, str]:
    """Identify the cohort and released files used by one fit.

    Args:
        root: CSV directory, required for real datasets.
        released_dir: Local released NPZ directory for real datasets.
        dataset: One of gbsg, support2 or synthetic.
        split: Positive released split number or synthetic partition seed.

    Returns:
        Input file fingerprints, or the synthetic generation recipe identity.
    """
    if dataset == "synthetic":
        return {"synthetic": "six-patient-survival-features7-generation-seed0-v2"}
    paths = {
        "cohort": Path(root) / ("gbsg.csv" if dataset == "gbsg" else "support2.csv")
    }
    paths.update({name: Path(released_dir) / f"{name}_{split}.npz" for name in SPLITS})
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def validate_resume(
    record: dict[str, Any], request: dict[str, Any], run_dir: Path | None = None
) -> None:
    """Reject changed requests or missing/altered completed outputs.

    Args:
        record: Previously saved fit record.
        request: Settings, input and source identity of the requested fit.
        run_dir: Optional fit directory whose recorded files must match.

    Raises:
        ValueError: If identities differ or recorded output files are invalid.
    """
    if not isinstance(record, dict) or record.get("request") != request:
        raise ValueError(
            "Saved run identity does not match the requested fit; preserve "
            "the existing files and inspect the record before rerunning"
        )

    if run_dir is not None:
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, dict) or "predictions.npz" not in artifacts:
            raise ValueError(
                f"Incomplete run at {run_dir}: no verified predictions; "
                "preserve this directory before explicitly restarting the fit"
            )
        for name, expected in artifacts.items():
            path = run_dir / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(
                    f"Missing or changed result artifact: {path}. Preserve this "
                    "run and inspect the files before explicitly restarting; "
                    "automatic repair is disabled"
                )


def resolve_run(
    parent: Path, request: dict[str, Any]
) -> tuple[Path, dict[str, Any] | None]:
    """Create a versioned fit directory or verify an identical completed run.

    Args:
        parent: Readable model/split/seed directory below the output root.
        request: JSON settings, input, source and runtime identity of one fit.

    Returns:
        The fingerprint-named directory and its verified record, or None
        for a newly created run. Existing files are never overwritten.

    Raises:
        ValueError: If an existing run is incomplete, unreadable or altered.
    """
    encoded = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    run_id = hashlib.sha256(encoded).hexdigest()[:16]
    run_dir = parent / f"run_{run_id}"
    parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        record_path = run_dir / "record.json"
        if not record_path.is_file():
            raise ValueError(
                f"Incomplete run at {run_dir}: no completed record. Another "
                "process may still be running. Preserve this directory before "
                "explicitly restarting the fit; automatic repair is disabled"
            ) from None
        try:
            record = json.loads(record_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Unreadable completed record at {record_path}; preserve the "
                "run and inspect it before explicitly restarting"
            ) from error
        validate_resume(record, request, run_dir)
        return run_dir, record
    (run_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    return run_dir, None


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    """Hash completed outputs before recording a resumable fit.

    Args:
        run_dir: Completed fit directory, excluding its record.json.

    Returns:
        Relative file paths mapped to SHA-256 fingerprints.
    """
    paths = [run_dir / "predictions.npz"]
    paths += [
        run_dir / name
        for name in ("request.json", "split_ids.json")
        if (run_dir / name).is_file()
    ]
    paths += sorted((run_dir / "checkpoint").glob("*"))
    paths += sorted((run_dir / "trainer").glob("*.ckpt"))
    if not paths[0].is_file():
        raise ValueError("A completed fit requires predictions.npz")
    return {
        str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def implementation_hashes() -> dict[str, str]:
    """Identify example and native source used for training and scoring.

    Returns:
        Source identifiers mapped to SHA-256 fingerprints. Installed source
        is read directly, so an editable checkout and a source wheel work.
    """
    paths = {p.name: p for p in Path(__file__).parent.glob("*.py")}
    package = Path(pyhealth.__file__).parent
    for name in [
        "trainer.py",
        "models/survival_mdn.py",
        "metrics/survival_curve.py",
        "tasks/survival_time_to_event.py",
        "datasets/gbsg.py",
        "datasets/sample_dataset.py",
        "datasets/support2.py",
        "datasets/utils.py",
    ]:
        paths["pyhealth/" + name] = package / name
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def prepare_output(out: Path, dataset: str, runner: str) -> None:
    """Keep one dataset and runner type per output root.

    Args:
        out: Benchmark output root.
        dataset: Real dataset name or synthetic.
        runner: MDN or baseline runner identifier.

    Raises:
        ValueError: If another dataset or runner already owns this folder.
    """
    identity = {"dataset": dataset, "runner": runner}
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "benchmark.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Use separate output folders for each dataset and runner")
    marker.write_text(json.dumps(identity, indent=2) + "\n")


def git_commit(path: Path) -> str | None:
    """Read the checkout revision for provenance, without modifying Git.

    Args:
        path: Directory inside a Git checkout.

    Returns:
        HEAD, with a dirty suffix for modified tracked files, or None when Git
        metadata or the executable is unavailable.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")
