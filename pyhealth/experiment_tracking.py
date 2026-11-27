"""Optional experiment manifests with explicit local or MLflow storage.

Only pass patient-independent metadata and aggregate metrics. This module does
not extract or redact patient data from values supplied by the caller.
"""

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path


def _write_local(directory, record):
    """Write one manifest atomically with private file permissions."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / (record["record_id"] + ".json")
    fd, temp = tempfile.mkstemp(prefix=".run-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temp, dest)
    except Exception:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
    return str(dest)


def record_experiment(
    *,
    params,
    metrics,
    prompt=None,
    backend="local",
    local_dir=None,
    tracking_uri=None,
    experiment_name="PyHealth",
    fallback_dir=None,
):
    """Record patient-independent run metadata and aggregate metrics.

    Args:
        params: Flat string/number/bool parameter values, without patient data.
        metrics: Flat numeric aggregate metrics, without patient data.
        prompt: Optional patient-independent template metadata, including its
            text and SHA-256; omit for non-prompt experiments.
        backend: ``local`` or ``mlflow``; no MLflow dependency is required for local.
        local_dir: Required output directory for the local backend.
        tracking_uri: Required explicit MLflow tracking URI for the MLflow backend.
        experiment_name: MLflow experiment name; created when absent.
        fallback_dir: Explicit local directory to use if MLflow fails. If omitted,
            the original failure is raised rather than silently falling back.

    Returns:
        A dictionary reporting the actual backend, record ID, and local path or
        MLflow run ID. A fallback reports the error type and any partial run ID.

    Examples:
        >>> import hashlib
        >>> import tempfile
        >>> digest = hashlib.sha256(b"template").hexdigest()
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     result = record_experiment(params={"epochs": 1}, metrics={"f1": 0.5},
        ...         prompt={"template": "template", "template_sha256": digest},
        ...         local_dir=directory)
        ...     result["backend"]
        'local'
    """
    if backend not in {"local", "mlflow"}:
        raise ValueError("backend must be 'local' or 'mlflow'")
    if backend == "local" and not local_dir:
        raise ValueError("local_dir is required for local tracking")
    if backend == "mlflow" and not tracking_uri:
        raise ValueError("tracking_uri is required for MLflow tracking")
    for name, values in (("params", params), ("metrics", metrics)):
        if not isinstance(values, dict) or not all(isinstance(k, str) for k in values):
            raise ValueError(f"{name} must be a dictionary with string keys")
    if prompt is not None:
        if not isinstance(prompt, dict) or not all(isinstance(k, str) for k in prompt):
            raise ValueError("prompt must be a dictionary with string keys")
        template = prompt.get("template")
        if (
            not isinstance(template, str)
            or prompt.get("template_sha256")
            != hashlib.sha256(template.encode("utf-8")).hexdigest()
        ):
            raise ValueError("prompt template and SHA-256 must match")
    if any(not isinstance(v, (str, int, float, bool)) for v in params.values()):
        raise ValueError("params must contain scalar values")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in metrics.values()
    ):
        raise ValueError("metrics must contain numeric values")
    record = {
        "schema_version": 1,
        "record_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "params": params,
        "metrics": metrics,
        "prompt": prompt,
    }
    # Fail before creating an external run for non-serializable or nonfinite data.
    json.dumps(record, allow_nan=False)
    if backend == "local":
        record["tracking"] = {"backend": "local", "status": "complete"}
        return {
            "backend": "local",
            "record_id": record["record_id"],
            "path": _write_local(local_dir, record),
        }

    client = None
    run_id = None
    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(experiment_name)
        )
        run_id = client.create_run(experiment_id).info.run_id
        for key, value in params.items():
            client.log_param(run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run_id, key, float(value))
        record["tracking"] = {
            "backend": "mlflow",
            "status": "complete",
            "run_id": run_id,
        }
        client.log_text(
            run_id, json.dumps(record, sort_keys=True, indent=2), "run_manifest.json"
        )
        client.set_terminated(run_id, status="FINISHED")
        return {"backend": "mlflow", "record_id": record["record_id"], "run_id": run_id}
    except Exception as exc:
        termination_failed = False
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:  # noqa: BLE001 - preserve the original tracking failure
                termination_failed = True
        if fallback_dir is None:
            raise
        # Do not include exception text: it could contain credentials or data.
        record["tracking"] = {
            "backend": "local",
            "status": "fallback",
            "error_type": type(exc).__name__,
            "partial_mlflow_run_id": run_id,
            "partial_run_termination_failed": termination_failed,
        }
        return {
            "backend": "local",
            "record_id": record["record_id"],
            "path": _write_local(fallback_dir, record),
            "fallback": True,
            "error_type": type(exc).__name__,
            "partial_mlflow_run_id": run_id,
            "partial_run_termination_failed": termination_failed,
        }
