import os
import re
import glob
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.configs.config import load_yaml_config


class MIMICCXRReportsDataset(BaseDataset):
    """
    PyHealth Dataset: MIMIC-CXR Radiology Reports

    Sample format:
    {
        "patient_id": ...,
        "study_id": ...,
        "report_text": ...,
        "findings": ...,
        "impression": ...,
        "path": ...
    }
    """

    def __init__(
        self,
        root: str,
        patients: Optional[List[str]] = None,
        dev_mode: bool = False,
        limit: int = 200,
        **kwargs,
    ):
        # Resolve YAML config in a portable way
        if "__file__" in globals():
            current_dir = Path(__file__).parent
        else:
            # assume sys.path[0] contains the fork root
            import sys
            current_dir = Path(sys.path[0])

        yaml_path = current_dir / "configs" / "mimic_cxr_reports.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found at {yaml_path}")
       
        config = load_yaml_config(str(yaml_path))

        # Call BaseDataset (tables positional, root positional, config keyword)
        super().__init__(tables=["reports"], root=root, **kwargs)
        self.config = config  # store locally if needed

        # Dataset-specific attributes
        self.patients = patients
        self.dev_mode = dev_mode
        self.limit = limit

        # Extract ZIP and load samples
        self._extract_zip()
        self.samples = self._load_samples()

    # -----------------------------
    # Extraction
    # -----------------------------
    def _extract_zip(self):
        zip_path = os.path.join(self.root, "mimic-cxr-reports.zip")
        out_dir = os.path.join(self.root, "mimic-cxr-reports")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"{zip_path} not found.")
        if not os.path.exists(out_dir):
            print("Extracting mimic-cxr-reports.zip ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)

    # -----------------------------
    # Parsing
    # -----------------------------
    @staticmethod
    def extract_section(text: str, section: str) -> Optional[str]:
        pattern = rf"{section}:(.*?)(?=\n[A-Z ]+:|$)"
        m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def parse_report(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r") as f:
            text = f.read()
        return {
            "report_text": text,
            "findings": self.extract_section(text, "FINDINGS"),
            "impression": self.extract_section(text, "IMPRESSION"),
        }

    # -----------------------------
    # Load samples
    # -----------------------------
    def _load_samples(self) -> List[Dict[str, Any]]:
        base = os.path.join(self.root, "mimic-cxr-reports")
        patient_dirs = sorted(glob.glob(os.path.join(base, "p*")))
        if self.patients:
            patient_dirs = [p for p in patient_dirs if os.path.basename(p) in self.patients]

        samples = []
        count = 0

        for patient_path in patient_dirs:
            patient_id = os.path.basename(patient_path)
            study_dirs = sorted(glob.glob(os.path.join(patient_path, "*")))
            for study_path in study_dirs:
                study_id = os.path.basename(study_path)
                txt_files = glob.glob(os.path.join(study_path, "*.txt"))
                for txt_path in txt_files:
                    parsed = self.parse_report(txt_path)
                    sample = {
                        "patient_id": patient_id,
                        "study_id": study_id,
                        "report_text": parsed["report_text"],
                        "findings": parsed["findings"],
                        "impression": parsed["impression"],
                        "path": txt_path,
                    }
                    samples.append(sample)
                    count += 1
                    if self.dev_mode and count >= self.limit:
                        return samples
        return samples

    # -----------------------------
    # PyHealth API
    # -----------------------------
    def get_samples(self):
        return self.samples