

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import wfdb

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class LUDBDataset(BaseDataset):
    """Lobachevsky University Electrocardiography Database (LUDB)

    Dataset is available at https://physionet.org/files/ludb/1.0.1/

    The LUDB dataset includes 200 ECG recordings collected from 2017-2018. Each recording:
    - Is 10 seconds long, sampled at 500 Hz (5000 samples)
    - Contains 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    - Has manually annotated boundaries and peaks of P, T waves and QRS complexes
      independently annotated by cardiologists for each of 12 leads
    - Includes diagnostic labels (rhythm type, electric axis, conduction abnormalities,
      hypertrophies, ischemia, etc.)
    
    The dataset contains 16797 P waves, 21966 QRS complexes, and 19666 T waves across
    all recordings. Data is stored in WFDB-compatible format (.hea header files and .dat
    signal files). Version 1.0.1 includes a ludb.csv file with aggregated patient metadata
    (gender, age, rhythm type, electric axis, pacemaker presence, etc.).

    Args:
        root: Root directory of the raw data. Should contain the LUDB dataset files
            downloaded from PhysioNet. The directory should contain .hea and .dat files
            for each ECG recording.
        dataset_name: Name of the dataset. Default is None (uses class name).
        dev: Whether to enable dev mode (only use a small subset of the data).
            Default is False.
        config_path: Optional path to configuration file. Defaults to ludb.yaml.

    Attributes:
        task: Optional[str], name of the task (e.g., "ECG delineation").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> # First, download LUDB dataset from PhysioNet:
        >>> # Option 1: wget
        >>> #   wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/
        >>> # Option 2: Manual download from https://physionet.org/files/ludb/1.0.1/data/#files-panel
        >>> #
        >>> # Then provide the path to the downloaded directory
        >>> dataset = LUDBDataset(
        ...     root="/srv/local/data/physionet.org/files/ludb/1.0.1/",
        ...     dev=True,
        ... )
        >>> dataset.stats()

    Note:
        The dataset expects files to be organized in WFDB-compatible format with:
        - .hea files: Header files containing metadata, annotations, and diagnostic labels
          (parsed using wfdb.rdheader)
        - .dat files: Signal data files in WFDB binary format
          (parsed using wfdb.rdsamp)
        - ludb.csv (optional): Metadata file with patient information (gender, age, rhythm,
          electric axis, pacemaker, etc.) - available in version 1.0.1+

        Each recording should have files named like: 1.hea, 1.dat
        where the number ranges from 1 to 200 (total of 200 recordings).

        The annotations include:
        - P wave boundaries and peaks (onset/offset points)
        - QRS complex boundaries and peaks (onset/offset points)
        - T wave boundaries and peaks (onset/offset points)
        
        All annotations are provided independently for each of the 12 leads.
    """

    # ECG lead names in standard 12-lead ECG order
    LEAD_NAMES = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    
    # Metadata fields that may be available from ludb.csv
    METADATA_FIELDS = [
        "sex",
        "age",
        "rhythm",
        "electric_axis",
        "conduction_abnormalities",
        "extrasystolies",
        "hypertrophies",
        "cardiac_pacing",
        "ischemia",
        "non_specific_repolarization",
        "other_states",
    ]

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
    ):
        """Initialize the LUDB dataset.

        Args:
            root (str): Root directory containing the LUDB dataset files.
            dataset_name (Optional[str]): Optional name for the dataset.
                Defaults to None (uses class name).
            config_path (Optional[str]): Optional configuration file path.
                Defaults to ludb.yaml.
            dev (bool): Enable dev mode (use subset of data). Defaults to False.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ludb.yaml"
        
        metadata_file = Path(root) / "ludb-pyhealth.csv"
        
        if not os.path.exists(metadata_file):
            logger.info(f"{metadata_file} does not exist, preparing metadata...")
            self.prepare_metadata(root)
        
        default_tables = ["ludb_ecg"]
        
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "ludb",
            config_path=config_path,
            dev=dev,
        )

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata CSV file for the LUDB dataset.
        
        This method processes the LUDB dataset files and creates a CSV file
        with patient metadata and file paths for BaseDataset to use. Called
        automatically by __init__ if ludb-pyhealth.csv does not exist.
        
        Args:
            root (str): Root directory containing the LUDB dataset files.
        """
        output_path = Path(root) / "ludb-pyhealth.csv"
        
        # Find all .hea files in the root directory
        hea_files = [
            f
            for f in os.listdir(root)
            if f.endswith(".hea")
        ]
        
        if not hea_files:
            raise FileNotFoundError(
                f"No LUDB .hea files found in {root}. "
                "Please ensure the directory contains LUDB dataset files."
            )
        
        # Load CSV metadata file if available
        csv_path = os.path.join(root, "ludb.csv")
        if not os.path.exists(csv_path):
            # Try parent directory (common structure)
            parent_dir = os.path.dirname(root.rstrip(os.sep))
            csv_path = os.path.join(parent_dir, "ludb.csv")
        
        metadata_dict = {}
        if os.path.exists(csv_path):
            metadata_dict = self._load_metadata_csv(csv_path)
        
        # Extract recording numbers and sort
        recording_numbers = []
        for hea_file in hea_files:
            # Extract number from filename (e.g., 1.hea, 10.hea)
            match = re.search(r"^(\d+)\.hea$", hea_file)
            if match:
                rec_num = int(match.group(1))
            else:
                continue
            
            # Valid recording numbers are 1-200
            if 1 <= rec_num <= 200:
                recording_numbers.append(rec_num)
        
        recording_numbers.sort()
        
        # Build records list
        records = []
        for rec_num in recording_numbers:
            # File naming format: number.hea, number.dat (e.g., 1.hea, 1.dat)
            hea_file = f"{rec_num}.hea"
            dat_file = f"{rec_num}.dat"
            
            # Verify both files exist
            if not os.path.exists(os.path.join(root, hea_file)) or \
               not os.path.exists(os.path.join(root, dat_file)):
                continue
            
            # Create record
            patient_id = f"rec_{rec_num:03d}"
            record = {
                "patient_id": patient_id,
                "recording_number": rec_num,
                "signal_file": os.path.join(root, dat_file),
                "label_file": os.path.join(root, hea_file),
            }
            
            # Add metadata from CSV if available, otherwise use None
            # This ensures all config columns exist even if metadata is missing
            if rec_num in metadata_dict:
                meta = metadata_dict[rec_num]
                record.update(
                    {field: meta.get(field) for field in LUDBDataset.METADATA_FIELDS}
                    )
            else:
                record.update({field: None for field in LUDBDataset.METADATA_FIELDS})
            
            records.append(record)
        
        # Create DataFrame and save
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"Created metadata file: {output_path} with {len(df)} records")

    @staticmethod
    def _load_metadata_csv(csv_path: str) -> Dict[int, Dict]:
        """Load metadata from ludb.csv file.
        
        The CSV file contains patient metadata including sex, age, rhythm type,
        electric axis, conduction abnormalities, and other diagnostic 
        information. The CSV may have multi-line values in quoted fields.
        Called by prepare_metadata() to load patient metadata.
        
        Args:
            csv_path (str): Path to the ludb.csv file.
            
        Returns:
            Dict[int, Dict]: Mapping recording ID (int) to metadata dictionary.
                Keys include:
                - sex: Gender (M/F)
                - age: Age in years (may be ">89" for ages over 89)
                - rhythm: Heart rhythm type
                - electric_axis: Electrical axis of the heart
                - conduction_abnormalities: Conduction abnormality types
                - extrasystolies: Extrasystole types
                - hypertrophies: Hypertrophy types
                - cardiac_pacing: Pacemaker/pacing information
                - ischemia: Ischemia-related findings
                - non_specific_repolarization: Non-specific repolarization abnormalities
                - other_states: Other clinical states
        """
        metadata = {}
        
        try:
            # Read CSV with pandas, handling multi-line quoted fields
            # The CSV has quoted fields that may contain newlines
            df = pd.read_csv(
                csv_path,
                quotechar='"',
                skipinitialspace=True,
                on_bad_lines='skip',
                engine='python',
                quoting=1,  # QUOTE_ALL
            )
        except Exception as e:
            # If pandas fails, return empty dict
            import warnings
            warnings.warn(
                f"Could not parse CSV metadata file {csv_path}: {e}. "
                "Metadata will not be available for this dataset.",
                UserWarning
            )
            return metadata
        
        # Process each row and extract metadata
        for _, row in df.iterrows():
            try:
                # Get recording ID - try different column name variations
                rec_id = None
                for id_col in ['ID', 'id', 'Id', 'recording_id', 'Recording ID']:
                    if id_col in df.columns:
                        rec_id_val = row[id_col]
                        if pd.notna(rec_id_val):
                            try:
                                rec_id = int(rec_id_val)
                                break
                            except (ValueError, TypeError):
                                continue
                
                if rec_id is None or rec_id < 1 or rec_id > 200:
                    continue
                
                # Build metadata dictionary
                meta = {}
                
                # Extract metadata fields
                if 'Sex' in df.columns or 'sex' in df.columns:
                    sex_col = 'Sex' if 'Sex' in df.columns else 'sex'
                    if pd.notna(row[sex_col]):
                        meta['sex'] = str(row[sex_col]).strip().replace('\n', ' ')
                    else:
                        meta['sex'] = None
                
                if 'Age' in df.columns or 'age' in df.columns:
                    age_col = 'Age' if 'Age' in df.columns else 'age'
                    if pd.notna(row[age_col]):
                        age_val = str(row[age_col]).strip().replace('\n', ' ')
                    else:
                        age_val = None
                    if age_val:
                        if age_val.startswith('>'):
                            meta['age'] = age_val  # Keep as string for ">89"
                        else:
                            try:
                                meta['age'] = int(age_val)
                            except (ValueError, TypeError):
                                meta['age'] = age_val
                
                # Extract other fields, cleaning up newlines and whitespace
                field_mappings = {
                    'rhythm': ['Rhythms', 'rhythms', 'Rhythm', 'rhythm'],
                    'electric_axis': ['Electric axis of the heart', 'electric axis of the heart'],
                    'conduction_abnormalities': ['Conduction abnormalities', 'conduction abnormalities'],
                    'extrasystolies': ['Extrasystolies', 'extrasystolies'],
                    'hypertrophies': ['Hypertrophies', 'hypertrophies'],
                    'cardiac_pacing': ['Cardiac pacing', 'cardiac pacing'],
                    'ischemia': ['Ischemia', 'ischemia'],
                    'non_specific_repolarization': [
                        'Non-specific repolarization abnormalities',
                        'non-specific repolarization abnormalities',
                    ],
                    'other_states': ['Other states', 'other states'],
                }
                
                for meta_key, col_names in field_mappings.items():
                    for col_name in col_names:
                        if col_name in df.columns:
                            value = row[col_name]
                            if pd.notna(value):
                                # Clean up multi-line values
                                cleaned_value = str(value).strip().replace('\n', '; ')
                                if cleaned_value:
                                    meta[meta_key] = cleaned_value
                            break
                
                # Store metadata for this recording
                if meta:  # Only store if we have some metadata
                    metadata[rec_id] = meta
                
            except Exception as e:
                # Skip rows that can't be parsed
                continue
        
        return metadata

    @staticmethod
    def _split_path(file_path: str) -> tuple[str, str]:
        """Split a file path into directory and filename.
        
        Helper method used by parse_hea_file, parse_dat_file, and 
        parse_annotation_file to extract directory and filename components.
        
        Args:
            file_path (str): Path to a file (can be relative or absolute).
            
        Returns:
            tuple[str, str]: Tuple of (directory, filename) where both are 
                absolute paths.
        """
        abs_path = os.path.abspath(file_path)
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return directory, filename

    @staticmethod
    def parse_hea_file(hea_path: str) -> Dict:
        """Parse a LUDB .hea header file to extract metadata and annotations.
        
        Uses WFDB library to parse header files. Called by prepare_metadata()
        during dataset initialization or can be called directly for parsing
        individual header files.

        Args:
            hea_path (str): Path to the .hea header file.

        Returns:
            Dict: Dictionary containing:
                - record_name: Name of the recording
                - sampling_rate: Sampling frequency in Hz
                - num_leads: Number of ECG leads
                - lead_names: List of lead names
                - signal_length: Length of signal in samples
                - age: Patient age (int or ">89" string)
                - sex: Patient sex (M/F)
                - rhythm: Heart rhythm label (if available)
                - electric_axis: Electrical axis label (if available)
                - diagnoses: List of all diagnostic labels
                - annotations: Dictionary with annotations for each lead
                  (Currently empty - P/QRS/T wave annotations are in separate binary files)
                  
        Note:
            P/QRS/T wave annotations are stored in separate binary annotation files
            (one per lead: .i, .ii, .iii, .avr, .avl, .avf, .v1, .v2, .v3, .v4, .v5, .v6).
            These need to be loaded separately using parse_annotation_file() or similar method.
        """
        annotations = {lead: {} for lead in LUDBDataset.LEAD_NAMES}
        metadata = {}
        
        # Extract record name from path (wfdb.rdheader expects path without .hea extension)
        directory, filename = LUDBDataset._split_path(hea_path)
        record_name = os.path.join(directory, os.path.splitext(filename)[0])
        
        # Read header file using wfdb
        record = wfdb.rdheader(record_name)
        
        # Extract basic metadata from wfdb record
        metadata["record_name"] = record.record_name
        metadata["num_leads"] = record.n_sig
        metadata["sampling_rate"] = int(record.fs) if record.fs else 500
        metadata["signal_length"] = record.sig_len
        metadata["lead_names"] = record.sig_name[:12] if record.sig_name else []
        
        # Parse diagnostic metadata from comments (LUDB-specific format)
        # wfdb.rdheader provides comments as a list of strings (without '#' prefix)
        comment_lines = record.comments
        
        # Parse diagnostic metadata from comments
        # Format: <age>: <value>, <sex>: <M/F>, <diagnoses>: followed by diagnostic labels
        # Note: wfdb strips the '#' prefix, so comments come as '<age>: 51' not '#<age>: 51'
        in_diagnoses_section = False
        diagnoses_list = []
        
        for line in comment_lines:
            # Handle both string lines and list items
            if isinstance(line, str):
                line = line.strip()
            else:
                line = str(line).strip()
            
            if not line:
                continue
            
            # Parse age
            if line.startswith("<age>:"):
                age_str = line.split(":", 1)[1].strip()
                try:
                    # Handle ">89" format
                    if age_str.startswith(">"):
                        metadata["age"] = age_str  # Keep as string
                    else:
                        metadata["age"] = int(age_str)
                except (ValueError, TypeError):
                    metadata["age"] = age_str  # Keep as string if parsing fails
            
            # Parse sex
            elif line.startswith("<sex>:"):
                sex_str = line.split(":", 1)[1].strip()
                metadata["sex"] = sex_str.upper()  # Normalize to uppercase
            
            # Start of diagnoses section
            elif line == "<diagnoses>:":
                in_diagnoses_section = True
            
            # Parse diagnostic labels (in diagnoses section)
            elif in_diagnoses_section:
                # Remove trailing period
                diag_line = line.rstrip(".")
                
                # Parse rhythm
                if diag_line.startswith("Rhythm:"):
                    rhythm = diag_line.split(":", 1)[1].strip()
                    metadata["rhythm"] = rhythm
                    diagnoses_list.append(f"Rhythm: {rhythm}")
                
                # Parse electric axis
                elif diag_line.startswith("Electric axis of the heart:"):
                    axis = diag_line.split(":", 1)[1].strip()
                    metadata["electric_axis"] = axis
                    diagnoses_list.append(f"Electric axis: {axis}")
                
                # Other diagnoses (hypertrophy, ischemia, etc.)
                else:
                    # Store all other diagnoses
                    diagnoses_list.append(diag_line)
        
        # Store all diagnoses as a list
        if diagnoses_list:
            metadata["diagnoses"] = diagnoses_list
        
        # Note: P/QRS/T wave annotations are stored in separate binary files per lead
        # (e.g., 1.i, 1.ii, 1.v1, etc.) and need to be loaded separately
        # See parse_annotation_file() method for loading binary annotation files
        metadata["annotations"] = annotations
        return metadata

    @staticmethod
    def load_all_annotations(recording_base_path: str, recording_number: int) -> Dict[str, Dict]:
        """Load all annotation files for a single recording (all 12 leads).
        
        LUDB stores annotations in separate binary files, one per lead:
        - Recording 1, Lead I: 1.i
        - Recording 1, Lead II: 1.ii
        - Recording 1, Lead V1: 1.v1
        - etc.
        Can be called directly to load all annotations for a recording.
        
        Args:
            recording_base_path (str): Base directory path containing 
                annotation files.
            recording_number (int): Recording number (1-200).
            
        Returns:
            Dict[str, Dict]: Mapping lead names to their annotations:
                {
                    "i": {"p_waves": [...], "qrs_complexes": [...], "t_waves": [...]},
                    "ii": {...},
                    ...
                }
        """
        # Map lead names to file suffixes
        lead_to_suffix = {
            "i": "i",
            "ii": "ii",
            "iii": "iii",
            "avr": "avr",
            "avl": "avl",
            "avf": "avf",
            "v1": "v1",
            "v2": "v2",
            "v3": "v3",
            "v4": "v4",
            "v5": "v5",
            "v6": "v6",
        }
        
        annotations = {}
        for lead_name, suffix in lead_to_suffix.items():
            # File naming format: number.suffix (e.g., 1.i, 1.v1)
            annotation_path = os.path.join(recording_base_path, f"{recording_number}.{suffix}")
            
            if os.path.exists(annotation_path):
                annotations[lead_name] = (
                    LUDBDataset.parse_annotation_file(annotation_path)
                )
            else:
                logger.warning(
                    f"Annotation file not found for recording "
                    f"{recording_number}, lead {lead_name}"
                )
                annotations[lead_name] = {
                    "p_waves": [],
                    "qrs_complexes": [],
                    "t_waves": []
                }
        
        return annotations

    @staticmethod
    def parse_annotation_file(annotation_path: str) -> Dict:
        """Parse a LUDB binary annotation file for a single lead.
        
        Uses WFDB library to parse annotation files. LUDB stores annotations 
        in separate binary files, one per lead:
        - Recording 1, Lead I: 1.i
        - Recording 1, Lead II: 1.ii
        - Recording 1, Lead V1: 1.v1
        - etc.
        
        These binary files contain P wave, QRS complex, and T wave boundaries
        and peaks (onset/offset/peak points) for that specific lead.
        Called by load_all_annotations() for each lead.
        
        Args:
            annotation_path (str): Path to the binary annotation file 
                (e.g., "1.i", "1.v1").
            
        Returns:
            Dict: Dictionary containing annotations for the lead:
                - p_waves: List of dicts with keys: onset, peak, offset
                - qrs_complexes: List of dicts with keys: onset, peak, offset
                - t_waves: List of dicts with keys: onset, peak, offset
                
        Note:
            This method uses the WFDB library to parse annotation files. The wfdb
            library is required for this dataset.
            
            The WFDB annotation format uses specific symbols for different wave types:
            - P wave: 'p' (onset), 'P' (peak), '}' (offset)
            - QRS complex: '(' (onset), 'N' or 'Q' or 'R' or 'S' (peak), ')' (offset)
            - T wave: 't' (onset), 'T' (peak), 'u' (offset)
        """
        if not os.path.exists(annotation_path):
            logger.warning(f"Annotation file not found: {annotation_path}")
            return {
                "p_waves": [],
                "qrs_complexes": [],
                "t_waves": []
            }
        
        try:
            # Extract record name and extension from path
            # e.g., "/path/to/1.i" -> record_name="/path/to/1", extension="i"
            directory, filename = LUDBDataset._split_path(annotation_path)
            
            # File naming format: number.extension (e.g., 1.i, 1.v1)
            record_name_with_num = filename.rsplit(".", 1)[0]  # "1"
            extension = filename.rsplit(".", 1)[1]  # "i"
            record_name = os.path.join(directory, record_name_with_num)
            
            # Read annotation file using wfdb
            annotation = wfdb.rdann(record_name, extension)
            
            # Parse annotations by symbol type
            # WFDB annotation symbols for ECG waves:
            # P wave: 'p' (onset), 'P' (peak), '}' (offset)
            # QRS: '(' (onset), 'N'/'Q'/'R'/'S' (peak), ')' (offset)
            # T wave: 't' (onset), 'T' (peak), 'u' (offset)
            
            p_waves = []
            qrs_complexes = []
            t_waves = []
            
            # Track current wave boundaries
            current_p = {"onset": None, "peak": None, "offset": None}
            current_qrs = {"onset": None, "peak": None, "offset": None}
            current_t = {"onset": None, "peak": None, "offset": None}
            
            # Map symbols to wave types
            # P wave symbols
            P_ONSET_SYMBOLS = {'p', 'P'}
            P_PEAK_SYMBOLS = {'P'}
            P_OFFSET_SYMBOLS = {'}', 'p'}
            
            # QRS symbols
            QRS_ONSET_SYMBOLS = {'(', '['}
            QRS_PEAK_SYMBOLS = {'N', 'Q', 'R', 'S', 'V', 'E'}
            QRS_OFFSET_SYMBOLS = {')', ']'}
            
            # T wave symbols
            T_ONSET_SYMBOLS = {'t', 'T'}
            T_PEAK_SYMBOLS = {'T'}
            T_OFFSET_SYMBOLS = {'u', 'U', 't'}
            
            # Process annotations in order
            for sample, symbol in zip(annotation.sample, annotation.symbol):
                symbol_upper = symbol.upper()
                
                # Process P waves
                if symbol in P_ONSET_SYMBOLS or symbol_upper in P_ONSET_SYMBOLS:
                    if current_p["onset"] is None:
                        current_p["onset"] = int(sample)
                elif symbol in P_PEAK_SYMBOLS or symbol_upper in P_PEAK_SYMBOLS:
                    current_p["peak"] = int(sample)
                elif symbol in P_OFFSET_SYMBOLS or symbol_upper in P_OFFSET_SYMBOLS:
                    if current_p["offset"] is None:
                        current_p["offset"] = int(sample)
                        # Complete P wave
                        if current_p["onset"] is not None and current_p["peak"] is not None:
                            p_waves.append({
                                "onset": current_p["onset"],
                                "peak": current_p["peak"],
                                "offset": current_p["offset"]
                            })
                        current_p = {"onset": None, "peak": None, "offset": None}
                
                # Process QRS complexes
                if symbol in QRS_ONSET_SYMBOLS:
                    if current_qrs["onset"] is None:
                        current_qrs["onset"] = int(sample)
                elif symbol in QRS_PEAK_SYMBOLS or symbol_upper in QRS_PEAK_SYMBOLS:
                    current_qrs["peak"] = int(sample)
                elif symbol in QRS_OFFSET_SYMBOLS:
                    if current_qrs["offset"] is None:
                        current_qrs["offset"] = int(sample)
                        # Complete QRS complex
                        if current_qrs["onset"] is not None and current_qrs["peak"] is not None:
                            qrs_complexes.append({
                                "onset": current_qrs["onset"],
                                "peak": current_qrs["peak"],
                                "offset": current_qrs["offset"]
                            })
                        current_qrs = {"onset": None, "peak": None, "offset": None}
                
                # Process T waves
                if symbol in T_ONSET_SYMBOLS or symbol_upper in T_ONSET_SYMBOLS:
                    if current_t["onset"] is None:
                        current_t["onset"] = int(sample)
                elif symbol in T_PEAK_SYMBOLS or symbol_upper in T_PEAK_SYMBOLS:
                    current_t["peak"] = int(sample)
                elif symbol in T_OFFSET_SYMBOLS or symbol_upper in T_OFFSET_SYMBOLS:
                    if current_t["offset"] is None:
                        current_t["offset"] = int(sample)
                        # Complete T wave
                        if current_t["onset"] is not None and current_t["peak"] is not None:
                            t_waves.append({
                                "onset": current_t["onset"],
                                "peak": current_t["peak"],
                                "offset": current_t["offset"]
                            })
                        current_t = {"onset": None, "peak": None, "offset": None}
            
            return {
                "p_waves": p_waves,
                "qrs_complexes": qrs_complexes,
                "t_waves": t_waves
            }
            
        except Exception as e:
            logger.error(f"Error parsing annotation file {annotation_path}: {e}")
            return {
                "p_waves": [],
                "qrs_complexes": [],
                "t_waves": []
            }

    @staticmethod
    def parse_dat_file(dat_path: str):
        """Parse a LUDB .dat binary signal file.
        
        Uses WFDB library to parse signal files. Can be called directly to
        load ECG signal data from a .dat file.

        Args:
            dat_path (str): Path to the .dat file.

        Returns:
            numpy.ndarray: ECG signal data of shape (12, 5000) in float64 
                format. Data is in physical units (mV) as converted by 
                wfdb.rdsamp().
        """
        # Extract record name from path (wfdb.rdsamp expects path without .dat extension)
        directory, filename = LUDBDataset._split_path(dat_path)
        record_name = os.path.join(directory, os.path.splitext(filename)[0])
        
        signals, _ = wfdb.rdsamp(record_name)
        
        # wfdb.rdsamp returns signals as (samples, leads) shape
        # Transpose to get (leads, samples) shape
        signal = signals.T
        
        return signal


if __name__ == "__main__":
    # Example usage
    # Users need to download LUDB dataset from PhysioNet first.
    # The dataset is available at: https://physionet.org/files/ludb/1.0.1/
    #
    # To download using wfdb tools:
    #   pip install wfdb
    #   wfdb download ludb
    #
    # Then provide the path to the downloaded directory containing .hea and .dat files
    dataset = LUDBDataset(
        root="/srv/local/data/physionet.org/files/ludb/1.0.1/",
        dev=True,
    )
    dataset.stats()

