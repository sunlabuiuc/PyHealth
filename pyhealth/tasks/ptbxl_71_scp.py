from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


class PTBXLAllDiagnosticClassification(BaseTask):
    """71-class multi-label classification on PTB-XL (All diagnostic SCP codes).

    Unlike the 5-superclass task, this extracts all granular SCP codes
    flagged as diagnostic (diagnostic == 1) in scp_statements.csv.

    >>> task = PTBXLAllDiagnosticClassification(
    ...     scp_statements_path="tests/data/ptbxl_synthetic/scp_statements.csv"
    ... )
    >>> task.task_name
    'PTBXLAllDiagnosticClassification'
    """
    
    task_name: str = "PTBXLAllDiagnosticClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(self, scp_statements_path: str):
        self.scp_statements_path = Path(scp_statements_path)
        
        df = pd.read_csv(self.scp_statements_path, index_col=0)
        self.valid_codes = set(df[df["diagnostic"] == 1].index.astype(str))
        super().__init__()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        from pyhealth.datasets.ptbxl import load_ptbxl_record, parse_scp_codes

        samples = []
        for event in patient.get_events(event_type="records"):
            codes = parse_scp_codes(event.scp_codes)
            
            labels = sorted([code for code in codes if code in self.valid_codes])
            
            if not labels:
                continue
                
            samples.append({
                "patient_id": patient.patient_id,
                "record_id": str(event.record_id),
                "signal": load_ptbxl_record(event.signal_file),
                "labels": labels,
            })
        return samples