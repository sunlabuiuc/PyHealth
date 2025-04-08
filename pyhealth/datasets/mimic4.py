import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC4Dataset(BaseDataset):
    """
    A dataset class for handling MIMIC-IV data.

    This class is responsible for loading and managing the MIMIC-IV dataset,
    which includes tables such as patients, admissions, and icustays.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the MIMIC4Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "mimic4".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic4.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic4",
            config_path=config_path,
        )
        return


# TODO: Utilize the following methods (by John Wu) in the YAML config file
#   # Fix for parse_notes method in MIMIC4Dataset
    # def parse_notes(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
    #     """Parse clinical notes from the note_root directory.
        
    #     Args:
    #         patients: a dict of `Patient` objects indexed by patient_id.
            
    #     Returns:
    #         The updated patients dict.
    #     """
    #     if not self.note_root or not os.path.exists(self.note_root):
    #         print("Note root directory not found, skipping notes parsing.")
    #         return patients
            
    #     print("Reading discharge notes...")
    #     note_path = os.path.join(self.note_root, "discharge.csv")
    #     if not os.path.exists(note_path):
    #         print(f"Note file {note_path} not found, skipping notes parsing.")
    #         return patients
            
    #     note_df = pd.read_csv(note_path)
    #     # Make all column names lowercase
    #     note_df.columns = note_df.columns.str.lower()
        
    #     note_df = note_df.dropna(subset=["subject_id", "text", "charttime"])
    #     print(f"Read {len(note_df)} note events.")
    #     note_df = note_df.sort_values(["subject_id", "charttime"], ascending=True)
        
    #     note_df['subject_id'] = note_df['subject_id'].astype(str)
    #     note_df = self._dev_mode(note_df)
        
    #     # Define function to extract sections from notes
    #     def get_section(text, section_header):
    #         pattern = re.escape(section_header) + "(.*?)(?=\n[A-Za-z ]+:|$)"
    #         match = re.search(pattern, text, flags=re.DOTALL)
    #         if match:
    #             return match.group(1).strip()
    #         return ""
        
    #     # Group by patient and process notes
    #     for patient_id, patient_notes in note_df.groupby("subject_id"):
    #         if patient_id not in patients:
    #             continue
                
    #         for _, row in patient_notes.iterrows():
    #             text = row['text']
                
    #             # Process note sections based on configuration
    #             if self.note_sections[0] == "all":
    #                 # Add entire note as a single event
    #                 event = Event(
    #                     type="clinical_note",
    #                     timestamp=strptime(row['charttime']),
    #                     attr_dict={
    #                         "value": text,
    #                         "section": "all",
    #                         "patient_id": patient_id
    #                     }
    #                 )
    #                 patients[patient_id].add_event(event)
    #             else:
    #                 if self.concatenate_notes:
    #                     # Concatenate all specified sections
    #                     combined_text = ""
    #                     for section in self.note_sections:
    #                         section_text = get_section(text.lower(), section.lower())
    #                         if section_text:
    #                             combined_text += f"<{section}> {section_text} </{section}> "
                                
    #                     if combined_text:
    #                         event = Event(
    #                             type="clinical_note",
    #                             timestamp=strptime(row['charttime']),
    #                             attr_dict={
    #                                 "value": combined_text.strip(),
    #                                 "section": "combined",
    #                                 "patient_id": patient_id
    #                             }
    #                         )
    #                         patients[patient_id].add_event(event)
    #                 else:
    #                     # Add each section as a separate event
    #                     for section in self.note_sections:
    #                         section_text = get_section(text.lower(), section.lower())
    #                         if section_text:
    #                             event = Event(
    #                                 type="clinical_note",
    #                                 timestamp=strptime(row['charttime']),
    #                                 attr_dict={
    #                                     "value": section_text,
    #                                     "section": section,
    #                                     "patient_id": patient_id
    #                                 }
    #                             )
    #                             patients[patient_id].add_event(event)
        
    #     return patients

    # # Fix for parse_xrays method in MIMIC4Dataset
    # def parse_xrays(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
    #     """Parse X-ray metadata from the cxr_root directory.
        
    #     Args:
    #         patients: a dict of `Patient` objects indexed by patient_id.
            
    #     Returns:
    #         The updated patients dict.
    #     """
    #     if not self.cxr_root or not os.path.exists(self.cxr_root):
    #         print("CXR root directory not found, skipping X-ray parsing.")
    #         return patients
            
    #     print("Reading CXR metadata...")
    #     metadata_path = os.path.join(self.cxr_root, "mimic-cxr-2.0.0-metadata.csv")
    #     if not os.path.exists(metadata_path):
    #         print(f"X-ray metadata file {metadata_path} not found, skipping X-ray parsing.")
    #         return patients
            
    #     cxr_jpg_meta_df = pd.read_csv(metadata_path)
    #     # Make all column names lowercase
    #     cxr_jpg_meta_df.columns = cxr_jpg_meta_df.columns.str.lower()
        
    #     # Process datetime columns
    #     cxr_jpg_meta_df.studydate = cxr_jpg_meta_df.studydate.astype(str)
    #     cxr_jpg_meta_df.studytime = cxr_jpg_meta_df.studytime.astype(str).str.split(".").str[0]
    #     cxr_jpg_meta_df["studydatetime"] = pd.to_datetime(
    #         cxr_jpg_meta_df.studydate + cxr_jpg_meta_df.studytime,
    #         format="%Y%m%d%H%M%S",
    #         errors="coerce"
    #     )
        
    #     # Filter and prepare dataframe
    #     cxr_df = cxr_jpg_meta_df[["subject_id", "study_id", "dicom_id", "studydatetime"]]
    #     cxr_df = cxr_df.dropna(subset=["subject_id", "dicom_id", "studydatetime"])
    #     cxr_df = cxr_df.sort_values(["subject_id", "studydatetime"], ascending=True)
    #     print(f"Read {len(cxr_df)} x-ray events.")
        
    #     cxr_df['subject_id'] = cxr_df['subject_id'].astype(str)
    #     cxr_df = self._dev_mode(cxr_df)
        
    #     # Process each patient's X-rays
    #     for patient_id, patient_xrays in cxr_df.groupby("subject_id"):
    #         if patient_id not in patients:
    #             continue
                
    #         for _, row in patient_xrays.iterrows():
    #             dicom_id = row['dicom_id']
    #             image_path = os.path.join(self.cxr_root, "images", f"{dicom_id}.jpg")
                
    #             event = Event(
    #                 type="xray",
    #                 timestamp=row['studydatetime'],
    #                 attr_dict={
    #                     "dicom_id": dicom_id,
    #                     "study_id": row['study_id'],
    #                     "image_path": image_path,
    #                     "patient_id": patient_id
    #                 }
    #             )
    #             patients[patient_id].add_event(event)
                
    #     return patients

    # def _add_events_to_patient_dict(
    #     self,
    #     patient_dict: Dict[str, Patient],
    #     group_df: pd.DataFrame,
    # ) -> Dict[str, Patient]:
    #     """Helper function which adds the events column of a df.groupby object to the patient dict."""
    #     for _, events in group_df.items():
    #         for event in events:
    #             patient_dict = self._add_event_to_patient_dict(patient_dict, event)
    #     return patient_dict

    # @staticmethod
    # def _add_event_to_patient_dict(
    #     patient_dict: Dict[str, Patient],
    #     event: Event,
    # ) -> Dict[str, Patient]:
    #     """Helper function which adds an event to the patient dict."""
    #     patient_id = event.attr_dict["patient_id"]
    #     try:
    #         patient_dict[patient_id].add_event(event)
    #     except KeyError:
    #         pass
    #     return patient_dict

    # def stat(self) -> str:
    #     """Returns some statistics of the base dataset."""
    #     lines = list()
    #     lines.append("")
    #     lines.append(f"Statistics of base dataset (dev={self.dev}):")
    #     lines.append(f"\t- Dataset: {self.dataset_name}")
    #     lines.append(f"\t- Number of patients: {len(self.patients)}")
    #     num_visits = [len(p.get_events_by_type("admissions")) for p in
    #                   self.patients.values()]
    #     lines.append(f"\t- Number of visits: {sum(num_visits)}")
    #     lines.append(
    #         f"\t- Number of visits per patient: {sum(num_visits) / len(num_visits):.4f}"
    #     )
    #     for table in self.tables:
    #         num_events = [
    #             len(p.get_events_by_type(table)) for p in self.patients.values()
    #         ]
    #         lines.append(
    #             f"\t- Number of events per patient in {table}: "
    #             f"{sum(num_events) / len(num_events):.4f}"
    #         )
            
    #     # Add stats for multimodal data if available
    #     if hasattr(self, 'note_root') and self.note_root:
    #         num_notes = [
    #             len(p.get_events_by_type("clinical_note")) for p in self.patients.values()
    #         ]
    #         if sum(num_notes) > 0:
    #             lines.append(
    #                 f"\t- Number of clinical notes per patient: "
    #                 f"{sum(num_notes) / len(num_notes):.4f}"
    #             )
                
    #     if hasattr(self, 'cxr_root') and self.cxr_root:
    #         num_xrays = [
    #             len(p.get_events_by_type("xray")) for p in self.patients.values()
    #         ]
    #         if sum(num_xrays) > 0:
    #             lines.append(
    #                 f"\t- Number of X-rays per patient: "
    #                 f"{sum(num_xrays) / len(num_xrays):.4f}"
    #             )
                
    #     lines.append("")
    #     print("\n".join(lines))
    #     return "\n".join(lines)
