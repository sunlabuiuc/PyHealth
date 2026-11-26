"""Tests for optional experiment tracking and explicit local fallback."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from pyhealth.experiment_tracking import record_experiment


@pytest.fixture
def run_data():
    return {
        "params": {"epochs": 2},
        "metrics": {"f1": 0.5},
        "prompt": {
            "template": "safe {boundary}",
            "template_sha256": hashlib.sha256(b"safe {boundary}").hexdigest(),
        },
    }


def test_local_manifest_and_private_permissions(tmp_path, run_data):
    result = record_experiment(**run_data, local_dir=tmp_path)
    path = tmp_path / (result["record_id"] + ".json")
    saved = json.loads(path.read_text())
    assert result["backend"] == "local" and result["path"] == str(path)
    assert (
        saved["params"] == run_data["params"]
        and saved["metrics"] == run_data["metrics"]
    )
    assert saved["tracking"]["status"] == "complete"
    assert os.stat(path).st_mode & 0o077 == 0


def test_mlflow_requires_explicit_fallback_and_excludes_exception_text(
    tmp_path, monkeypatch, run_data
):
    class Broken:
        def __init__(self, tracking_uri):
            raise RuntimeError("sensitive-secret")

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=Broken))
    with pytest.raises(RuntimeError):
        record_experiment(**run_data, backend="mlflow", tracking_uri="file:///tmp/test")
    result = record_experiment(
        **run_data,
        backend="mlflow",
        tracking_uri="file:///tmp/test",
        fallback_dir=tmp_path,
    )
    assert result["fallback"] is True and result["error_type"] == "RuntimeError"
    saved = json.loads((tmp_path / (result["record_id"] + ".json")).read_text())
    assert saved["tracking"]["status"] == "fallback"
    assert "sensitive-secret" not in json.dumps(saved)


def test_mlflow_success_and_partial_failure_are_visible(
    tmp_path, monkeypatch, run_data
):
    class Client:
        events = []
        fail = False

        def __init__(self, tracking_uri):
            self.events.append(("uri", tracking_uri))

        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="42")

        def create_run(self, experiment_id):
            return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

        def log_param(self, run_id, key, value):
            self.events.append(("param", key, value))

        def log_metric(self, run_id, key, value):
            if self.fail:
                raise OSError("down")
            self.events.append(("metric", key, value))

        def log_text(self, run_id, text, artifact_file):
            self.events.append(("artifact", json.loads(text), artifact_file))

        def set_terminated(self, run_id, status):
            self.events.append(("status", status))

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=Client))
    result = record_experiment(
        **run_data, backend="mlflow", tracking_uri="sqlite:///test.db"
    )
    assert result["backend"] == "mlflow" and result["run_id"] == "run-1"
    assert ("status", "FINISHED") in Client.events
    Client.fail = True
    result = record_experiment(
        **run_data,
        backend="mlflow",
        tracking_uri="sqlite:///test.db",
        fallback_dir=tmp_path,
    )
    assert result["partial_mlflow_run_id"] == "run-1"
    assert ("status", "FAILED") in Client.events


def test_invalid_tracking_inputs_fail_before_storage(tmp_path, run_data):
    with pytest.raises(ValueError):
        record_experiment(**run_data, backend="mlflow")
    with pytest.raises(ValueError):
        record_experiment(**run_data, backend="local")
    run_data["metrics"] = {"f1": float("nan")}
    with pytest.raises(ValueError):
        record_experiment(**run_data, local_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_wrong_template_hash_is_rejected(tmp_path, run_data):
    run_data["prompt"]["template_sha256"] = "not-the-template-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        record_experiment(**run_data, local_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_manifest_contains_only_caller_supplied_metadata(tmp_path, monkeypatch, run_data):
    monkeypatch.setenv("PRIVATE_PATIENT", "PRIVATE_DIAGNOSIS")
    saved = record_experiment(**run_data, local_dir=tmp_path)
    text = (tmp_path / (saved["record_id"] + ".json")).read_text()
    assert "PRIVATE_PATIENT" not in text and "PRIVATE_DIAGNOSIS" not in text
    assert json.loads(text)["prompt"] == run_data["prompt"]


def test_non_prompt_experiment_can_be_recorded(tmp_path):
    result = record_experiment(
        params={"epochs": 1}, metrics={"loss": 0.2}, local_dir=tmp_path
    )
    saved = json.loads((tmp_path / (result["record_id"] + ".json")).read_text())
    assert saved["prompt"] is None
    assert saved["metrics"]["loss"] == 0.2
