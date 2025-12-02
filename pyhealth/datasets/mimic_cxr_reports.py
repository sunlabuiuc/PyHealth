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

        # Load YAML config for dataset metadata (but do not let BaseDataset
        # attempt to load table CSVs — this dataset handles its own loading).
        config = load_yaml_config(str(yaml_path))

        # Do NOT call BaseDataset.__init__ here. The MIMIC-CXR reports
        # dataset manages its own data extraction and parsing (ZIP of
        # text files) and the BaseDataset implementation expects CSV/TSV
        # table files. Calling the base initializer leads to attempts to
        # scan CSV files (which don't exist for this dataset) and raises
        # FileNotFoundError. Instead, set the minimal attributes that are
        # expected by other code paths and proceed with dataset-specific
        # loading below.
        self.root = root
        self.tables = ["reports"]
        self.dataset_name = self.__class__.__name__
        self.config = config
        self.dev = dev_mode

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

        # Primary pattern: top-level patient directories named like p*
        patient_dirs = sorted(glob.glob(os.path.join(base, "p*")))

        # Fallback 1: recursive glob in case layout is nested (e.g. files/.../p19/...)
        if not patient_dirs:
            patient_dirs = sorted(glob.glob(os.path.join(base, "**", "p*"), recursive=True))

        # Fallback 2: os.walk to catch any directories starting with 'p'
        if not patient_dirs:
            found = []
            for dirpath, dirnames, _ in os.walk(base):
                for d in dirnames:
                    if d.startswith("p"):
                        found.append(os.path.join(dirpath, d))
            patient_dirs = sorted(set(found))

        if self.patients:
            patient_dirs = [p for p in patient_dirs if os.path.basename(p) in self.patients]

        samples = []
        count = 0

        for patient_path in patient_dirs:
            # patient_path may be like .../p10 or .../p19; inside it there are
            # subfolders per-patient (study folders) e.g. p10000032. We want the
            # patient_id to be the subfolder (p10000032) and the study_id to be
            # the txt filename without extension (s53189527).
            study_dirs = sorted(glob.glob(os.path.join(patient_path, "*")))
            for study_path in study_dirs:
                # Use the subfolder name as patient_id
                patient_id = os.path.basename(study_path)
                txt_files = glob.glob(os.path.join(study_path, "*.txt"))
                for txt_path in txt_files:
                    # Use the txt filename (without extension) as study_id
                    study_id = os.path.splitext(os.path.basename(txt_path))[0]
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