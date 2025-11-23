"""PromptEHR dataset for synthetic EHR generation.

This module provides the dataset class for training and generating with PromptEHR.
"""

from typing import Optional, List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from pyhealth.tokenizer import Tokenizer


def create_promptehr_tokenizer(diagnosis_codes: List[str]) -> Tokenizer:
    """Create a tokenizer for PromptEHR with special generation tokens.

    This function creates a PyHealth Tokenizer configured for PromptEHR,
    with 7 special tokens that are compatible with the pehr_scratch implementation.

    Special tokens (IDs 0-6):
        - <pad> (0): Padding token
        - <s> (1): Start of sequence (BOS)
        - </s> (2): End of sequence (EOS)
        - <unk> (3): Unknown token
        - <v> (4): Visit start marker
        - </v> (5): Visit end marker
        - <mask> (6): Masking token for corruption

    Medical diagnosis codes will start at ID 7 (code_offset=7).

    Args:
        diagnosis_codes: List of unique diagnosis code strings (e.g., ["401.9", "427.31", ...])

    Returns:
        Configured PyHealth Tokenizer with 1:1 code-to-token mapping.

    Example:
        >>> codes = ["401.9", "427.31", "250.00"]
        >>> tokenizer = create_promptehr_tokenizer(codes)
        >>> tokenizer.get_vocabulary_size()
        10  # 7 special tokens + 3 diagnosis codes
        >>> tokenizer.convert_tokens_to_indices(["<v>", "401.9", "</v>"])
        [4, 7, 5]  # <v>=4, first code=7, </v>=5

    Note:
        This maintains compatibility with pehr_scratch checkpoint token IDs.
        The order of special tokens MUST NOT be changed.
    """
    # Define special tokens in exact order (IDs will be 0-6)
    # CRITICAL: Order must match pehr_scratch for checkpoint compatibility
    special_tokens = [
        "<pad>",   # ID 0 - padding
        "<s>",     # ID 1 - start of sequence (BART BOS)
        "</s>",    # ID 2 - end of sequence (BART EOS)
        "<unk>",   # ID 3 - unknown token
        "<v>",     # ID 4 - visit start marker
        "</v>",    # ID 5 - visit end marker
        "<mask>",  # ID 6 - masking token for corruption
    ]

    # Create tokenizer with special tokens first, then diagnosis codes
    # PyHealth's Vocabulary adds special_tokens first, preserving order
    # This automatically creates code_offset=7 (len(special_tokens))
    tokenizer = Tokenizer(
        tokens=diagnosis_codes,
        special_tokens=special_tokens
    )

    return tokenizer


class PatientRecord:
    """Container for a single patient's EHR data.

    Stores demographics (age, gender) and visit history for PromptEHR.
    Note: Ethnicity removed from demographics for medical validity.
    """

    def __init__(
        self,
        subject_id: int,
        age: float,
        gender: str,
        visits: List[List[str]]
    ):
        """Initialize patient record.

        Args:
            subject_id: MIMIC-III subject ID
            age: Patient age at first admission
            gender: 'M' or 'F'
            visits: List of visits, each visit is list of ICD-9 codes
        """
        self.subject_id = subject_id
        self.age = age
        self.gender = gender
        self.visits = visits

        # Computed properties
        self.gender_id = 1 if gender == 'F' else 0  # 0=M, 1=F

    def to_dict(self) -> Dict:
        """Convert to dictionary format for dataset."""
        return {
            'subject_id': self.subject_id,
            'x_num': np.array([self.age], dtype=np.float32),
            'x_cat': np.array([self.gender_id], dtype=np.int64),
            'visits': self.visits,
            'num_visits': len(self.visits)
        }


def load_mimic_data(
    patients_path: str,
    admissions_path: str,
    diagnoses_path: str,
    logger: logging.Logger,
    num_patients: Optional[int] = None
) -> Tuple[List[PatientRecord], List[str]]:
    """Load MIMIC-III data and format into PatientRecord objects.

    Args:
        patients_path: Path to PATIENTS.csv file
        admissions_path: Path to ADMISSIONS.csv file
        diagnoses_path: Path to DIAGNOSES_ICD.csv file
        logger: Logger instance for output
        num_patients: Maximum number of patients to load (optional)

    Returns:
        Tuple of (patient_records, diagnosis_codes_list)
        where diagnosis_codes_list is all unique codes for building tokenizer
    """
    logger.info("Loading MIMIC-III data files")

    try:
        patients_df = pd.read_csv(patients_path, parse_dates=['DOB'])
        logger.info(f"Loaded {len(patients_df)} patients")

        admissions_df = pd.read_csv(admissions_path, parse_dates=['ADMITTIME'])
        logger.info(f"Loaded {len(admissions_df)} admissions")

        diagnoses_df = pd.read_csv(diagnoses_path)
        logger.info(f"Loaded {len(diagnoses_df)} diagnosis records")

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e.filename}")
        return [], []
    except Exception as e:
        logger.error(f"Unexpected error during file loading: {e}")
        return [], []

    # Calculate age at first admission
    first_admissions = admissions_df.loc[
        admissions_df.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
    ][['SUBJECT_ID', 'ADMITTIME']]

    demo_df = pd.merge(
        patients_df[['SUBJECT_ID', 'GENDER', 'DOB']],
        first_admissions,
        on='SUBJECT_ID',
        how='inner'
    )

    demo_df['AGE'] = (demo_df['ADMITTIME'].dt.year - demo_df['DOB'].dt.year)
    demo_df['AGE'] = np.where(demo_df['AGE'] > 89, 90, demo_df['AGE'])

    # Merge admissions with diagnoses
    admissions_info = admissions_df[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    merged_df = pd.merge(
        admissions_info,
        diagnoses_df[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    # Merge with demographics
    final_df = pd.merge(
        merged_df,
        demo_df[['SUBJECT_ID', 'AGE', 'GENDER']],
        on='SUBJECT_ID',
        how='left'
    )

    # Sort chronologically
    final_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM'], inplace=True)

    logger.info("Processing patient records")

    # Build patient records and collect unique codes
    patient_records = []
    all_codes = set()

    patient_groups = final_df.groupby('SUBJECT_ID')

    for subject_id, patient_data in patient_groups:
        # Extract demographics
        age = float(patient_data['AGE'].iloc[0])
        gender = patient_data['GENDER'].iloc[0]

        # Extract visits (grouped by HADM_ID)
        visits = []
        visit_groups = patient_data.groupby('HADM_ID', sort=False)

        for _, visit_data in visit_groups:
            # Get ICD-9 codes for this visit
            icd_codes = visit_data['ICD9_CODE'].astype(str).tolist()
            all_codes.update(icd_codes)
            visits.append(icd_codes)

        # Create patient record
        record = PatientRecord(
            subject_id=int(subject_id),
            age=age,
            gender=gender,
            visits=visits
        )
        patient_records.append(record)

        if num_patients is not None and len(patient_records) >= num_patients:
            break

    logger.info(f"Loaded {len(patient_records)} patient records")
    logger.info(f"Unique diagnosis codes: {len(all_codes)}")

    # Log statistics
    if len(patient_records) > 0:
        avg_visits = np.mean([len(r.visits) for r in patient_records])
        avg_codes_per_visit = np.mean([len(code_list) for r in patient_records for code_list in r.visits])

        logger.info(f"Average visits per patient: {avg_visits:.2f}")
        logger.info(f"Average codes per visit: {avg_codes_per_visit:.2f}")

        # Gender distribution
        gender_counts = pd.Series([r.gender for r in patient_records]).value_counts()
        logger.info(f"Gender distribution: {gender_counts.to_dict()}")

    return patient_records, sorted(list(all_codes))


class CorruptionFunctions:
    """Data corruption functions for robust EHR generation training.

    Implements three corruption strategies:
    1. Mask infilling: Replace code spans with <mask> token
    2. Token deletion: Randomly delete codes
    3. Token replacement: Replace codes with random alternatives
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        lambda_poisson: float = 3.0,
        del_probability: float = 0.15,
        rep_probability: float = 0.15
    ):
        """Initialize corruption functions.

        Args:
            tokenizer: PyHealth Tokenizer instance
            lambda_poisson: Poisson lambda for span masking length
            del_probability: Probability of deleting each token
            rep_probability: Probability of replacing each token
        """
        self.tokenizer = tokenizer
        self.lambda_poisson = lambda_poisson
        self.del_probability = del_probability
        self.rep_probability = rep_probability
        self.mask_token = "<mask>"
        self.vocab_size = tokenizer.get_vocabulary_size() - 7  # Exclude special tokens

    def mask_infill(
        self,
        visits: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """Apply Poisson-distributed span masking to diagnosis codes."""
        corrupted_visits = []
        label_masks = []

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                label_masks.append([])
                continue

            # Sample span length from Poisson distribution
            span_length = max(1, min(num_codes - 1,
                                    np.random.poisson(self.lambda_poisson)))

            # Randomly select start position
            max_start = num_codes - span_length
            start_idx = np.random.randint(0, max(1, max_start + 1))

            # Create corrupted visit
            corrupted_visit = (
                visit[:start_idx] +
                [self.mask_token] +
                visit[start_idx + span_length:]
            )

            # Create label mask (1 for masked positions)
            label_mask = [0] * num_codes
            for i in range(start_idx, min(start_idx + span_length, num_codes)):
                label_mask[i] = 1

            corrupted_visits.append(corrupted_visit)
            label_masks.append(label_mask)

        return corrupted_visits, label_masks

    def del_token(self, visits: List[List[str]]) -> List[List[str]]:
        """Apply binomial token deletion to diagnosis codes."""
        corrupted_visits = []

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                continue

            # Generate deletion mask (1 = delete, 0 = keep)
            deletion_mask = np.random.binomial(1, self.del_probability, num_codes)

            # Keep at least 1 code per visit
            if deletion_mask.sum() == num_codes:
                keep_idx = np.random.randint(0, num_codes)
                deletion_mask[keep_idx] = 0

            # Apply deletion
            corrupted_visit = [
                code for i, code in enumerate(visit)
                if deletion_mask[i] == 0
            ]

            corrupted_visits.append(corrupted_visit)

        return corrupted_visits

    def rep_token(self, visits: List[List[str]]) -> List[List[str]]:
        """Apply binomial token replacement with random codes."""
        corrupted_visits = []

        # Get all diagnosis codes (excluding special tokens at indices 0-6)
        all_codes = []
        for idx in range(7, self.tokenizer.get_vocabulary_size()):
            all_codes.append(self.tokenizer.vocabulary.idx2token[idx])

        for visit in visits:
            num_codes = len(visit)

            if num_codes == 0:
                corrupted_visits.append([])
                continue

            # Generate replacement mask (1 = replace, 0 = keep)
            replacement_mask = np.random.binomial(1, self.rep_probability, num_codes)

            # Generate random replacement codes
            random_codes = np.random.choice(all_codes, num_codes, replace=True)

            # Apply replacement
            corrupted_visit = []
            for i, code in enumerate(visit):
                if replacement_mask[i] == 1:
                    corrupted_visit.append(random_codes[i])
                else:
                    corrupted_visit.append(code)

            corrupted_visits.append(corrupted_visit)

        return corrupted_visits


class PromptEHRDataset(Dataset):
    """PyTorch Dataset for patient EHR data with separated demographics and codes.

    Args:
        patient_records: List of PatientRecord objects
        tokenizer: PyHealth Tokenizer configured for PromptEHR
        logger: Logger instance for debugging

    Example:
        >>> # Load MIMIC-III data
        >>> records, codes = load_mimic_data(..., logger)
        >>> # Create tokenizer
        >>> tokenizer = create_promptehr_tokenizer(codes)
        >>> # Create dataset
        >>> dataset = PromptEHRDataset(records, tokenizer, logger)
    """

    def __init__(
        self,
        patient_records: List[PatientRecord],
        tokenizer: Tokenizer,
        logger: logging.Logger
    ):
        """Initialize dataset."""
        self.patient_records = patient_records
        self.tokenizer = tokenizer
        self.logger = logger

        if len(patient_records) > 0:
            sample = patient_records[0].to_dict()
            self.logger.debug(f"Sample x_num shape: {sample['x_num'].shape}")
            self.logger.debug(f"Sample x_cat shape: {sample['x_cat'].shape}")
            self.logger.debug(f"Sample num_visits: {sample['num_visits']}")

    def __len__(self) -> int:
        return len(self.patient_records)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single patient record.

        Returns:
            Dict with:
                - x_num: [1] array with age
                - x_cat: [1] array with gender_id
                - visit_codes: List[List[str]] of diagnosis codes
                - token_ids: List[int] encoded visit sequence
                - subject_id: Patient identifier
        """
        record = self.patient_records[idx]
        record_dict = record.to_dict()

        # Encode visits to token IDs using PyHealth tokenizer
        # Build sequence: <s> + <v> codes </v> + <v> codes </v> + ... + </s>
        token_sequence = ["<s>"]  # Start token

        for visit in record.visits:
            token_sequence.append("<v>")  # Visit start
            token_sequence.extend(visit)  # Visit codes
            token_sequence.append("</v>")  # Visit end

        token_sequence.append("</s>")  # End token

        # Convert to indices
        token_ids = self.tokenizer.convert_tokens_to_indices(token_sequence)

        return {
            'x_num': record_dict['x_num'],
            'x_cat': record_dict['x_cat'],
            'visit_codes': record.visits,
            'token_ids': np.array(token_ids, dtype=np.int64),
            'subject_id': record.subject_id
        }
