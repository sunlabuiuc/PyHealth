"""
ECG Diagnosis task for the PTB-XL dataset.

Supports both multilabel (one ECG → multiple diagnostic classes) and
multiclass (one ECG → single dominant class) classification, matching the
experimental setup in:

    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis. Proceedings of Machine Learning
    Research, 149, 414-424.

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import ast
import logging
from typing import Any, Dict, List, Optional

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Superdiagnostic label set used in the Nonaka & Seita (2021) benchmark.
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Mapping from SCP code prefix → superdiagnostic class.
# Based on the scp_statements.csv shipped with PTB-XL.
SCP_TO_SUPER: Dict[str, str] = {
    "NORM": "NORM",
    "IMI": "MI",
    "ASMI": "MI",
    "ILMI": "MI",
    "AMI": "MI",
    "ALMI": "MI",
    "INJAS": "MI",
    "LMI": "MI",
    "INJAL": "MI",
    "IPLMI": "MI",
    "IPMI": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    "RMI": "MI",
    "INJIL": "MI",
    "STD_": "STTC",
    "ISCA": "STTC",
    "ISCI": "STTC",
    "ISC_": "STTC",
    "IVCTE": "STTC",
    "STTC": "STTC",
    "NST_": "STTC",
    "STE_": "STTC",
    "LNGQT": "STTC",
    "TAB_": "STTC",
    "INVT": "STTC",
    "LVOLT": "HYP",
    "HVOLT": "HYP",
    "HYP": "HYP",
    "RVH": "HYP",
    "LVH": "HYP",
    "LAO/LAE": "HYP",
    "RAO/RAE": "HYP",
    "SEHYP": "HYP",
    "LAFB/LPFB": "CD",
    "IRBBB": "CD",
    "ILBBB": "CD",
    "CRBBB": "CD",
    "CLBBB": "CD",
    "IVCD": "CD",
    "LBBB": "CD",
    "RBBB": "CD",
    "WPW": "CD",
    "LPFB": "CD",
    "LAFB": "CD",
    "CD": "CD",
}


def _scp_to_superclasses(scp_codes_str: str) -> List[str]:
    """Convert a raw SCP-ECG codes string to a list of superdiagnostic labels.

    Args:
        scp_codes_str: String representation of a dict mapping SCP code to
            likelihood, e.g. ``"{'NORM': 100.0, 'SR': 0.0}"``.

    Returns:
        Sorted list of unique superdiagnostic class names present in the
        record (likelihood > 0).
    """
    try:
        codes: Dict[str, float] = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return []

    supers = set()
    for code, likelihood in codes.items():
        if likelihood > 0 and code in SCP_TO_SUPER:
            supers.add(SCP_TO_SUPER[code])
    return sorted(supers)


class PTBXLDiagnosis(BaseTask):
    """ECG multilabel diagnosis task for the PTB-XL dataset.

    Each ECG recording is mapped to one or more of five superdiagnostic
    classes: NORM, MI, STTC, CD, HYP — following the benchmark setup of
    Nonaka & Seita (2021).

    The task returns the path to the WFDB signal file so that downstream
    processors or model code can load the raw signal on demand.

    Attributes:
        task_name (str): ``"PTBXLDiagnosis"``.
        input_schema (Dict[str, str]): ``{"signal_file": "signal_file"}``.
        output_schema (Dict[str, str]): ``{"labels": "multilabel"}``.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLDiagnosis
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl")
        >>> samples = dataset.set_task(PTBXLDiagnosis())
        >>> print(samples[0])
        {'patient_id': '...', 'signal_file': '...', 'labels': ['NORM']}
    """

    task_name: str = "PTBXLDiagnosis"
    input_schema: Dict[str, str] = {"signal_file": "signal_file"}
    output_schema: Dict[str, str] = {"labels": "multilabel"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate diagnosis samples for a single patient.

        Args:
            patient: A PyHealth patient object containing ``ptbxl`` events.

        Returns:
            List of sample dicts, one per ECG recording, each containing:
                - ``patient_id`` (str): Patient identifier.
                - ``record_id`` (int): ECG record identifier.
                - ``signal_file`` (str): Relative path to the WFDB .hea file.
                - ``labels`` (List[str]): Superdiagnostic class labels.
        """
        events = patient.get_events(event_type="ptbxl")
        samples = []
        for event in events:
            labels = _scp_to_superclasses(str(event.get("scp_codes", "{}")))
            if not labels:
                # Skip records with no mappable superdiagnostic label.
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.get("record_id"),
                    "signal_file": str(event.get("signal_file", "")),
                    "labels": labels,
                }
            )
        return samples


class PTBXLMulticlassDiagnosis(BaseTask):
    """ECG multiclass diagnosis task for the PTB-XL dataset.

    Assigns each ECG recording to a single superdiagnostic class by selecting
    the class with the highest aggregate SCP likelihood score. Records with
    ties or no mappable label are skipped.

    Attributes:
        task_name (str): ``"PTBXLMulticlassDiagnosis"``.
        input_schema (Dict[str, str]): ``{"signal_file": "signal_file"}``.
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLMulticlassDiagnosis
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl")
        >>> samples = dataset.set_task(PTBXLMulticlassDiagnosis())
        >>> print(samples[0])
        {'patient_id': '...', 'signal_file': '...', 'label': 'NORM'}
    """

    task_name: str = "PTBXLMulticlassDiagnosis"
    input_schema: Dict[str, str] = {"signal_file": "signal_file"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate single-label diagnosis samples for a single patient.

        Args:
            patient: A PyHealth patient object containing ``ptbxl`` events.

        Returns:
            List of sample dicts, one per ECG recording, each containing:
                - ``patient_id`` (str): Patient identifier.
                - ``record_id`` (int): ECG record identifier.
                - ``signal_file`` (str): Relative path to the WFDB .hea file.
                - ``label`` (str): Dominant superdiagnostic class label.
        """
        events = patient.get_events(event_type="ptbxl")
        samples = []
        for event in events:
            scp_str = str(event.get("scp_codes", "{}"))
            try:
                codes: Dict[str, float] = ast.literal_eval(scp_str)
            except (ValueError, SyntaxError):
                continue

            # Aggregate likelihood per superclass.
            scores: Dict[str, float] = {}
            for code, likelihood in codes.items():
                if likelihood > 0 and code in SCP_TO_SUPER:
                    sup = SCP_TO_SUPER[code]
                    scores[sup] = scores.get(sup, 0.0) + likelihood

            if not scores:
                continue

            dominant = max(scores, key=lambda k: scores[k])
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.get("record_id"),
                    "signal_file": str(event.get("signal_file", "")),
                    "label": dominant,
                }
            )
        return samples
