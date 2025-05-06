import pandas as pd
import os
from pyhealth.datasets import BaseEHRDataset

class SimpleMIMICCXRDataset(BaseEHRDataset):
    def __init__(
        self,
        root: str,
        metadata_path: str,
        image_dir: str,
        dev: bool = False,
        refresh_cache: bool = False,
    ):
        """Simple MIMIC-CXR dataset for view-specific X-ray generation task"""
        print("DEBUG: Entering SimpleMIMICCXRDataset.__init__")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        
        # Initialize patients to avoid None
        self.patients = {}
        print(f"DEBUG: Initialized self.patients as {self.patients}")
        
        # Preprocess the CSV data
        print("DEBUG: Preprocessing CSV data")
        cols = ['dicom_id', 'subject_id', 'study_id', 'view', 'count']
        try:
            # Read the first few rows to inspect the CSV
            print("DEBUG: Reading first few rows of CSV to inspect")
            df_preview = pd.read_csv(self.metadata_path, nrows=5)
            print("DEBUG: First few rows of CSV (raw):\n", df_preview)
            
            # Check if the first row is a header by comparing with expected columns
            first_row = pd.read_csv(self.metadata_path, nrows=1)
            expected_cols = set(cols)
            first_row_cols = set(first_row.columns)
            has_header = first_row_cols == expected_cols or first_row.iloc[0, 1] == 'subject_id'
            print(f"DEBUG: Does CSV have a header? {has_header}")
            
            # Load CSV based on header presence
            if has_header:
                print("DEBUG: Loading CSV with default header parsing")
                df = pd.read_csv(
                    self.metadata_path,
                    sep=',',
                    dtype={'study_id': str, 'subject_id': str, 'dicom_id': str},
                )
            else:
                print("DEBUG: Loading CSV without header")
                df = pd.read_csv(
                    self.metadata_path,
                    sep=',',
                    header=None,
                    names=cols,
                    dtype={'study_id': str, 'subject_id': str, 'dicom_id': str},
                    skiprows=0,
                )
            
            print("DEBUG: Loaded DataFrame shape:", df.shape)
            print("DEBUG: Loaded DataFrame columns:", df.columns.tolist())
            print("DEBUG: First few rows of loaded DataFrame:\n", df.head())
            
            # Validate required columns
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. Found columns: {df.columns.tolist()}")
            
            if df.empty:
                raise ValueError(f"CSV file {self.metadata_path} is empty")
            
            # Add image paths
            print("DEBUG: Adding image paths to DataFrame")
            df['image_path'] = df.apply(self._construct_image_path, axis=1)
            
            # Parse basic info
            print("DEBUG: Parsing basic info")
            self.parse_basic_info(df)
            
            # Store preprocessed patients before super().__init__
            preprocessed_patients = self.patients
            print(f"DEBUG: Preprocessed patients before super().__init__: {preprocessed_patients}")
            
        except Exception as e:
            print(f"DEBUG: Error preprocessing CSV: {str(e)}")
            raise ValueError(f"Error preprocessing CSV: {str(e)}")
        
        # Initialize BaseEHRDataset with empty tables to prevent re-parsing
        super().__init__(
            root=root,
            tables=[],  # Disable default table parsing
            dataset_name="simple_mimiccxr",
            dev=dev,
            refresh_cache=refresh_cache,
        )
        
        # Restore preprocessed patients
        self.patients = preprocessed_patients
        print(f"DEBUG: After super().__init__, self.patients is {self.patients}")

    def parse_tables(self):
        """Override parse_tables (not used since tables=[])"""
        print("DEBUG: Entering parse_tables (note: should not be called)")
        return {}

    def parse_basic_info(self, df):
        """Parse basic patient information from the DataFrame"""
        print("DEBUG: Entering parse_basic_info with DataFrame")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame first few rows:\n{df.head()}")
        print(f"DEBUG: self.patients before structuring: {self.patients}")
        
        self.patients = {}
        for (subject_id, study_id), group in df.groupby(["subject_id", "study_id"]):
            print(f"DEBUG: Processing subject_id: {subject_id}, study_id: {study_id}")
            # Skip invalid subject_ids (e.g., 'subject_id' from header)
            if not subject_id.isdigit() or subject_id == 'subject_id':
                print(f"DEBUG: Skipping invalid subject_id: {subject_id}")
                continue
            patient_id = f"p{subject_id}"
            events = self._create_events(group)
            if not events:
                print(f"DEBUG: No events for subject_id: {subject_id}, study_id: {study_id}")
                continue
            if patient_id not in self.patients:
                self.patients[patient_id] = {"patient_id": patient_id, "visits": []}
            self.patients[patient_id]["visits"].append({
                "visit_id": f"s{study_id}",
                "events": events
            })
            print(f"DEBUG: Added patient {patient_id} with visit s{study_id}")
        
        if not self.patients:
            raise ValueError("No patients with valid events were created.")
        
        print(f"DEBUG: self.patients after structuring: {self.patients}")
        return self.patients

    def _construct_image_path(self, row):
        """Construct the image path based on subject_id and study_id"""
        first_two_digits = row['subject_id'][:2]  # e.g., '12' for '12470349'
        path = os.path.join(
            self.image_dir,
            f"files/mimic-cxr-jpg/2.0.0/files/p{first_two_digits}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg"
        )
        return path

    def _create_events(self, group):
        """Create event entries"""
        events = [{
            "dicom_id": row["dicom_id"],
            "view_position": row.get("view", "UNKNOWN"),
            "image_path": row["image_path"],
            "study_time": None
        } for _, row in group.iterrows()]
        print(f"DEBUG: Created events for group:\n{group}\nEvents: {events}")
        return events

    def _convert_code_in_patient_dict(self, patients):
        """Override to skip code mapping since this dataset doesn't use clinical codes"""
        print(f"DEBUG: Entering _convert_code_in_patient_dict, self.patients is {self.patients}")
        print(f"DEBUG: Argument patients is {patients}")
        return patients

    def _set_statistics(self):
        """Override to skip statistics computation since this dataset doesn't use clinical codes"""
        print("DEBUG: Entering _set_statistics")
        print(f"DEBUG: self.patients before setting statistics: {self.patients}")
        self.statistics = {}
        return self

    def _set_table_parsers(self):
        """Override to debug table parser setup"""
        print("DEBUG: Entering _set_table_parsers")
        print(f"DEBUG: self.patients before setting table parsers: {self.patients}")
        super()._set_table_parsers()
        print(f"DEBUG: self.patients after setting table parsers: {self.patients}")

    def set_task(self, task_fn):
        """Override to debug task setup"""
        print("DEBUG: Entering set_task")
        samples = task_fn(self.patients)
        print(f"DEBUG: Samples generated by task_fn: {samples}")
        return super().set_task(task_fn)