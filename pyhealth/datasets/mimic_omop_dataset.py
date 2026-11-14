from pathlib import Path

from .omop import OMOPDataset


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "omop.yaml"


class MIMICOMOPDataset(OMOPDataset):
    """
    MIMIC-IV OMOP dataset built on top of PyHealth OMOPDataset.
    """

    def __init__(
        self,
        root,
        dev=False,
        cache_dir=".pyhealth_cache",
    ):
        super().__init__(
            root=str(root),
            tables=["condition_occurrence"],
            dataset_name="mimic4_hosp_omop",
            config_path=str(CONFIG_PATH),
            dev=dev,
            cache_dir=str(cache_dir),
        )

    def get_condition_code(self, condition_event):
        """
        Returns condition concept id if present,
        otherwise source code.
        """

        concept = getattr(
            condition_event,
            "condition_concept_id",
            None,
        )

        if concept not in (
            None,
            "",
            0,
            "0",
        ):
            return str(concept)

        source = getattr(
            condition_event,
            "condition_source_value",
            None,
        )

        if source not in (
            None,
            "",
        ):
            return str(source)

        return None

    def get_visit_count(self, patient):
        return len(
            patient.get_events(
                event_type="visit_occurrence"
            )
        )
