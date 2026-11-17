"""Actual CSV loader, processors, and LSTM integration on invented patients."""

from project.person_c.checkpoint import run


def test_csv_to_lstm(tmp_path):
    result = run(tmp_path)
    assert result["status"] == "passed"
    assert result["parameters_changed"]
