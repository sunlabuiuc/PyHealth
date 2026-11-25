from types import SimpleNamespace
from unittest.mock import patch

import torch

from pyhealth.tasks import PTBXLAllDiagnosticClassification


def test_ptbxl_71_scp_task_call(tmp_path):
    scp_csv = tmp_path / "scp_statements.csv"
    scp_csv.write_text(
        ",diagnostic\nNORM,1\nMI,1\nNONDIAG,0\n",
        encoding="utf-8",
    )

    task = PTBXLAllDiagnosticClassification(scp_statements_path=str(scp_csv))
    assert task.task_name == "PTBXLAllDiagnosticClassification"
    assert "NORM" in task.valid_codes
    assert "MI" in task.valid_codes
    assert "NONDIAG" not in task.valid_codes

    events = [
        SimpleNamespace(
            record_id=101,
            scp_codes="{'NORM': 100.0, 'NONDIAG': 50.0, 'MI': 80.0}",
            signal_file="rec1.dat",
        ),
        SimpleNamespace(
            record_id=102,
            scp_codes="{'NONDIAG': 50.0}",
            signal_file="rec2.dat",
        ),
    ]
    patient = SimpleNamespace(
        patient_id="p1",
        get_events=lambda event_type="records": events,
    )

    fake_signal = torch.zeros((12, 1000))
    with (
        patch("pyhealth.datasets.ptbxl.parse_scp_codes", side_effect=lambda s: eval(s)),
        patch("pyhealth.datasets.ptbxl.load_ptbxl_record", return_value=fake_signal),
    ):
        samples = task(patient)

    assert len(samples) == 1
    assert samples[0]["patient_id"] == "p1"
    assert samples[0]["record_id"] == "101"
    assert samples[0]["labels"] == ["MI", "NORM"]
    assert samples[0]["signal"].shape == (12, 1000)