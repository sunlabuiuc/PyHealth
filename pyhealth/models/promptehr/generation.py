"""
Generate synthetic patient sequences using trained PromptEHR model.

This module provides functions for generating realistic synthetic EHR data
using various conditioning strategies (demographics, visit structures, etc.).
"""
import json
import math
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Union, Dict


class DemographicSampler:
    """Sample patient demographics from empirical training distribution.

    Samples age and gender by directly drawing from the observed distribution
    in training data, ensuring synthetic patients match real population.
    """

    def __init__(self, patient_records: List, seed: int = 42):
        """Initialize sampler with empirical demographics from training data.

        Args:
            patient_records: List of patient records from training set.
                Each record should have 'age' and 'gender' attributes.
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)

        # Extract empirical demographics
        self.ages = []
        self.genders = []

        for patient in patient_records:
            # Handle both dict-like and object-like patient records
            if hasattr(patient, 'age') and hasattr(patient, 'gender'):
                age = patient.age
                gender = patient.gender
            elif isinstance(patient, dict) and 'age' in patient and 'gender' in patient:
                age = patient['age']
                gender = patient['gender']
            else:
                continue

            self.ages.append(float(age))
            # Convert gender to int: M=0, F=1
            if isinstance(gender, str):
                gender_int = 0 if gender == 'M' else 1
            else:
                gender_int = int(gender)
            self.genders.append(gender_int)

        # Convert to numpy arrays
        self.ages = np.array(self.ages)
        self.genders = np.array(self.genders)

        # Compute statistics
        self.stats = {
            'age_mean': np.mean(self.ages),
            'age_std': np.std(self.ages),
            'age_median': np.median(self.ages),
            'age_min': np.min(self.ages),
            'age_max': np.max(self.ages),
            'male_pct': (self.genders == 0).mean(),
            'female_pct': (self.genders == 1).mean(),
        }

    def sample(self) -> dict:
        """Sample demographics from empirical distribution.

        Returns:
            Dictionary with:
                - 'age': float (sampled from training ages)
                - 'sex': int (0=Male, 1=Female, sampled from training)
                - 'sex_str': str ('M' or 'F')
        """
        # Sample random index from training data
        idx = self.rng.randint(0, len(self.ages))

        age = self.ages[idx]
        sex = self.genders[idx]
        sex_str = 'M' if sex == 0 else 'F'

        return {
            'age': float(age),
            'sex': int(sex),
            'sex_str': sex_str
        }

    def __repr__(self):
        return (
            f"DemographicSampler(\n"
            f"  Age: mean={self.stats['age_mean']:.1f}, "
            f"std={self.stats['age_std']:.1f}, "
            f"range=[{self.stats['age_min']:.0f}, {self.stats['age_max']:.0f}]\n"
            f"  Gender: {self.stats['male_pct']:.1%} Male, "
            f"{self.stats['female_pct']:.1%} Female\n"
            f")"
        )


def build_first_code_prior(
    training_data_path: str,
    age_bins: int = 9
) -> Dict:
    """Build empirical P(first_code | age, gender) from training data.

    Args:
        training_data_path: Path to training data directory with MIMIC-III files
        age_bins: Number of age bins (default: 9 for [0-10), [10-20), ..., [80-90])

    Returns:
        Dictionary mapping (age_bin, gender) -> {code: probability}

    Example:
        >>> prior = build_first_code_prior('/path/to/train_data')
        >>> first_code = sample_first_code(65, 0, prior)
    """
    import pandas as pd

    # Load training data
    admissions = pd.read_csv(f'{training_data_path}/ADMISSIONS.csv')
    patients = pd.read_csv(f'{training_data_path}/PATIENTS.csv')
    diagnoses = pd.read_csv(f'{training_data_path}/DIAGNOSES_ICD.csv')

    # Calculate age at first admission
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    patients['DOB'] = pd.to_datetime(patients['DOB'])

    first_admissions = admissions.loc[
        admissions.groupby('SUBJECT_ID')['ADMITTIME'].idxmin()
    ][['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]

    demo = pd.merge(
        patients[['SUBJECT_ID', 'GENDER', 'DOB']],
        first_admissions,
        on='SUBJECT_ID',
        how='inner'
    )
    demo['AGE'] = (demo['ADMITTIME'].dt.year - demo['DOB'].dt.year)
    demo['AGE'] = demo['AGE'].apply(lambda x: 90 if x > 89 else max(0, x))

    # Get first diagnosis codes
    first_diag = pd.merge(
        demo[['SUBJECT_ID', 'HADM_ID', 'AGE', 'GENDER']],
        diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='inner'
    )

    # Keep only first code per patient (seq_num=1 or first alphabetically)
    first_diag = first_diag.sort_values(['SUBJECT_ID', 'ICD9_CODE'])
    first_diag = first_diag.groupby('SUBJECT_ID').first().reset_index()

    # Bin ages
    first_diag['age_bin'] = pd.cut(
        first_diag['AGE'],
        bins=list(range(0, 91, 10)),
        labels=list(range(age_bins)),
        include_lowest=True
    )

    # Convert gender to int (0=M, 1=F)
    first_diag['gender_int'] = (first_diag['GENDER'] == 'F').astype(int)

    # Calculate empirical distribution
    dist = {}
    for (age_bin, gender), group in first_diag.groupby(['age_bin', 'gender_int']):
        code_counts = group['ICD9_CODE'].value_counts()
        total = code_counts.sum()
        dist[(int(age_bin), int(gender))] = {
            str(code): count / total
            for code, count in code_counts.items()
        }

    return dist


def sample_first_code(
    age: float,
    gender: int,
    first_code_prior: Dict
) -> str:
    """Sample first diagnosis code from empirical distribution.

    Args:
        age: Patient age (0-90)
        gender: Patient gender (0=Male, 1=Female)
        first_code_prior: Prior from build_first_code_prior()

    Returns:
        Diagnosis code string (e.g., 'V3000', '41401')

    Example:
        >>> prior = build_first_code_prior('/path/to/train_data')
        >>> code = sample_first_code(65, 0, prior)
        >>> print(code)  # e.g., 'V3000'
    """
    # Bin age
    age_bin = min(int(age // 10), 8)  # [0-9] -> 0, [10-19] -> 1, ..., [80+] -> 8

    # Get distribution for this demographic
    key = (age_bin, gender)
    if key not in first_code_prior:
        # Fallback to gender-only or overall distribution
        fallback_key = None
        for k in first_code_prior.keys():
            if k[1] == gender:
                fallback_key = k
                break
        if fallback_key:
            key = fallback_key
        else:
            key = list(first_code_prior.keys())[0]

    code_probs = first_code_prior[key]
    codes = list(code_probs.keys())
    probs = list(code_probs.values())

    return np.random.choice(codes, p=probs)


def build_frequency_prior(
    tokenizer,
    frequency_path: Optional[Union[str, Path]] = None,
    epsilon: float = 1e-10,
    vocab_size: Optional[int] = None
) -> torch.Tensor:
    """Build log-frequency prior over vocabulary for frequency-guided generation.

    Args:
        tokenizer: DiagnosisCodeTokenizer with vocab and code_offset attributes.
        frequency_path: Path to training_frequencies.json. If None, uses uniform prior.
        epsilon: Small constant to avoid log(0) (default: 1e-10).
        vocab_size: Model vocabulary size. If None, inferred from tokenizer (not recommended).
            Should match model's lm_head output dimension.

    Returns:
        torch.Tensor of shape [vocab_size] with log-frequencies.
        Special tokens get 0 (neutral prior), diagnosis codes get log(freq + epsilon).

    Example:
        >>> prior = build_frequency_prior(tokenizer, './promptehr_outputs/training_frequencies.json', vocab_size=6963)
        >>> logits_guided = logits + alpha * prior  # Blend with model logits
    """
    # Use provided vocab size or infer from tokenizer
    # WARNING: Inferred size may not match model if there's a mismatch!
    if vocab_size is None:
        vocab_size = len(tokenizer.vocab.idx2code)

    log_freqs = torch.zeros(vocab_size)

    if frequency_path is None:
        # Uniform fallback: all codes equally likely
        uniform_log_freq = math.log(1.0 / len(tokenizer.vocab.idx2code))
        log_freqs[tokenizer.code_offset:] = uniform_log_freq
        return log_freqs

    # Load training frequencies
    with open(frequency_path, 'r') as f:
        freq_data = json.load(f)

    frequencies = freq_data['frequencies']

    # Fill in log-frequencies for each code
    # NOTE: We map code_idx directly to token_id without adding code_offset
    # because the model vocabulary doesn't include code_offset
    for code, freq in frequencies.items():
        if code in tokenizer.vocab.code2idx:
            code_idx = tokenizer.vocab.code2idx[code]
            if code_idx < vocab_size:
                log_freqs[code_idx] = math.log(freq + epsilon)

    # Codes not in training data get very low prior
    min_log_freq = math.log(epsilon)
    log_freqs = torch.where(
        log_freqs == 0,
        torch.tensor(min_log_freq),
        log_freqs
    )

    return log_freqs


def sample_demographics(
    age_mean: float = 60.0,
    age_std: float = 20.0,
    male_prob: float = 0.56
) -> dict:
    """Sample realistic patient demographics.

    Samples demographics from distributions matching MIMIC-III ICU population.

    Args:
        age_mean: Mean age for normal distribution (default: 60).
        age_std: Standard deviation for age (default: 20).
        male_prob: Probability of male gender (default: 0.56).

    Returns:
        Dictionary with:
            - 'age': float in range [0, 90]
            - 'sex': int (0=Male, 1=Female)
            - 'sex_str': str ('M' or 'F')
    """
    # Sample age from normal distribution, clipped to [0, 90]
    age = np.random.normal(age_mean, age_std)
    age = np.clip(age, 0, 90)

    # Sample sex from binomial distribution
    sex = 0 if np.random.rand() < male_prob else 1
    sex_str = 'M' if sex == 0 else 'F'

    return {
        'age': float(age),
        'sex': sex,
        'sex_str': sex_str
    }


def decode_patient_demographics(age: float, gender: int) -> dict:
    """Decode demographics back to readable format.

    Args:
        age: Normalized age value.
        gender: Gender category index.

    Returns:
        Dictionary with decoded demographics.
    """
    # Gender mapping (from data_loader.py)
    gender_map = {0: "M", 1: "F"}  # Fixed: M=0, F=1

    return {
        "age": f"{age:.1f}",
        "gender": gender_map.get(gender, "UNKNOWN")
    }


def parse_sequence_to_visits(
    token_ids: List[int],
    tokenizer
) -> List[List[str]]:
    """Parse generated token sequence into visit structure.

    Extracts visits by splitting at <v> and </v> markers, and decodes
    diagnosis codes within each visit.

    Args:
        token_ids: List of token IDs from model generation.
        tokenizer: PyHealth Tokenizer instance (must have bos_token_id,
            pad_token_id, code_offset, and vocab attributes).

    Returns:
        List of visits, where each visit is a list of ICD-9 code strings.

    Example:
        Input: [BOS, <v>, 401.9, 250.00, </v>, <v>, 428.0, </v>, <END>]
        Output: [['401.9', '250.00'], ['428.0']]
    """
    visits = []
    current_visit_codes = []

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_indices(["<v>"])[0]
    v_end_token_id = tokenizer.convert_tokens_to_indices(["<\\v>"])[0]
    bos_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.convert_tokens_to_indices(["<END>"])[0]

    in_visit = False

    for token_id in token_ids:
        if token_id == v_token_id:
            # Start of visit
            in_visit = True
            current_visit_codes = []
        elif token_id == v_end_token_id:
            # End of visit
            if in_visit:
                visits.append(current_visit_codes)
                in_visit = False
        elif token_id in [bos_token_id, end_token_id, tokenizer.pad_token_id]:
            # Skip special tokens
            continue
        elif in_visit and token_id >= tokenizer.code_offset:
            # Diagnosis code token - token_id is already the correct vocab index
            # FIX: code2idx already includes special tokens, so don't subtract offset
            if token_id < len(tokenizer.vocab.idx2code):
                code = tokenizer.vocab.idx2code[token_id]
                current_visit_codes.append(code)

    # Handle case where sequence ends without closing visit marker
    if in_visit and len(current_visit_codes) > 0:
        visits.append(current_visit_codes)

    return visits


def generate_patient_sequence_conditional(
    model,
    tokenizer,
    target_patient,
    device: torch.device,
    temperature: float = 0.3,
    top_k: int = 0,  # Disabled (test with top_p only)
    top_p: float = 0.95,  # Increased for more diversity
    prompt_prob: float = 0.0,
    max_codes_per_visit: int = 20
) -> dict:
    """Generate synthetic patient via conditional reconstruction (PromptEHR approach).

    Given a real patient from test set, randomly masks codes and reconstructs
    the full visit structure. Default prompt_prob=0.0 means zero-code-prompt
    generation (only demographics provided).

    Args:
        model: Trained PromptBartModel or PromptEHR model.
        tokenizer: DiagnosisCodeTokenizer instance.
        target_patient: Patient record from test set to reconstruct.
            Must have attributes: age, gender (or sex), visits.
        device: Device to run on.
        temperature: Sampling temperature (default: 0.3).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        prompt_prob: Probability of keeping each code as prompt (default: 0.0 = zero prompts).
        max_codes_per_visit: Cap visit codes at this number (default: 20).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of generated code sequences
            - 'target_visits': List[List[str]] of original codes
            - 'prompt_codes': List[List[str]] of codes provided as prompts
            - 'demographics': dict of patient demographics
    """
    model.eval()

    # Extract demographics (handle both 'gender' and 'sex' attributes)
    if hasattr(target_patient, 'age'):
        age = target_patient.age
    else:
        age = target_patient.get('age', 60.0)

    if hasattr(target_patient, 'gender'):
        gender_str = target_patient.gender
    elif hasattr(target_patient, 'sex'):
        gender_str = target_patient.sex
    else:
        gender_str = target_patient.get('gender', 'M')

    gender = 1 if gender_str == 'F' else 0

    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[gender]], dtype=torch.long).to(device)

    # Get visits
    if hasattr(target_patient, 'visits'):
        patient_visits = target_patient.visits
    else:
        patient_visits = target_patient.get('visits', [])

    # Initialize accumulators
    generated_visits = []
    prompt_codes_per_visit = []

    # Create dummy encoder input (prompts are in decoder)
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_indices(["<v>"])[0]
    v_end_token_id = tokenizer.convert_tokens_to_indices(["<\\v>"])[0]

    with torch.no_grad():
        # Process each visit from target patient
        for visit_idx, target_codes in enumerate(patient_visits):
            # Step 1: Cap codes at max_codes_per_visit
            num_codes = len(target_codes)
            if num_codes > max_codes_per_visit:
                target_codes = list(np.random.choice(target_codes, max_codes_per_visit, replace=False))
                num_codes = max_codes_per_visit

            if num_codes == 0:
                # Empty visit - skip
                generated_visits.append([])
                prompt_codes_per_visit.append([])
                continue

            # Step 2: Randomly mask codes (binomial sampling)
            keep_mask = np.random.binomial(1, prompt_prob, num_codes).astype(bool)
            prompt_codes = [code for i, code in enumerate(target_codes) if keep_mask[i]]

            # Step 3: Encode prompt codes as decoder input
            prompt_token_ids = [tokenizer.bos_token_id, v_token_id]
            for code in prompt_codes:
                # FIX: code2idx already returns token ID with offset included
                code_token_id = tokenizer.vocab.code2idx[code]
                prompt_token_ids.append(code_token_id)

            decoder_input_ids = torch.tensor([prompt_token_ids], dtype=torch.long).to(device)

            # Step 4: Generate to reconstruct full visit
            max_new_tokens = num_codes + 2  # Target length

            # Use model.generate() for automatic handling
            generated_ids = model.generate(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                x_num=x_num,
                x_cat=x_cat,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,  # Disable beam search, use sampling only
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=1,  # Prevents duplicate codes
                eos_token_id=v_end_token_id,  # Stop at </v>
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.bos_token_id]]  # Suppress BOS in generation
            )

            # Step 5: Extract generated codes
            visit_token_ids = generated_ids[0].cpu().tolist()

            # Extract code tokens (skip BOS, <v>, </v>)
            generated_code_ids = [
                tid for tid in visit_token_ids
                if tid >= tokenizer.code_offset
            ]

            # Decode codes (convert token IDs back to diagnosis codes)
            # FIX: code2idx already includes special tokens, so don't subtract offset
            generated_codes = []
            for tid in generated_code_ids:
                if tid < len(tokenizer.vocab.idx2code):
                    code = tokenizer.vocab.idx2code[tid]
                    generated_codes.append(code)

            # Step 6: Combine with prompt codes and deduplicate
            all_codes = list(set(generated_codes + prompt_codes))

            # Ensure exactly num_codes by sampling if needed
            if len(all_codes) < num_codes:
                # Not enough unique codes generated - resample with replacement
                needed = num_codes - len(all_codes)
                additional = list(np.random.choice(generated_codes, needed, replace=True)) if len(generated_codes) > 0 else []
                all_codes.extend(additional)
            elif len(all_codes) > num_codes:
                # Too many codes - sample exactly num_codes
                all_codes = list(np.random.choice(all_codes, num_codes, replace=False))

            generated_visits.append(all_codes)
            prompt_codes_per_visit.append(prompt_codes)

    return {
        'generated_visits': generated_visits,
        'target_visits': patient_visits,
        'prompt_codes': prompt_codes_per_visit,
        'demographics': {
            'age': age,
            'gender': gender_str
        }
    }


def generate_patient_with_structure_constraints(
    model,
    tokenizer,
    device: torch.device,
    target_structure: dict,
    age: Optional[float] = None,
    sex: Optional[int] = None,
    first_code: Optional[str] = None,
    temperature: float = 0.7,
    top_k: int = 0,  # Disabled (test with top_p only)
    top_p: float = 0.95,  # Increased for more diversity
    max_codes_per_visit: int = 25
) -> dict:
    """Generate patient with realistic visit structure constraints.

    This function generates patients visit-by-visit with controlled code counts
    sampled from real data distributions, producing more realistic EHR records.

    Args:
        model: Trained PromptBartModel or PromptEHR model.
        tokenizer: DiagnosisCodeTokenizer instance.
        device: Device to run on.
        target_structure: Dict with 'num_visits' and 'codes_per_visit' list.
        age: Patient age (if None, sampled from distribution).
        sex: Patient sex ID (0=M, 1=F; if None, sampled).
        first_code: First diagnosis code to condition on (if None, generated by model).
        temperature: Sampling temperature (default: 0.7).
        top_k: Top-k sampling parameter (default: 40).
        top_p: Nucleus sampling parameter (default: 0.9).
        max_codes_per_visit: Maximum codes per visit safety cap (default: 25).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of diagnosis codes
            - 'demographics': dict with 'age' and 'sex'
            - 'num_visits': int
            - 'num_codes': int
            - 'target_structure': dict (the structure we aimed for)
    """
    model.eval()

    # Sample demographics if not provided
    if age is None or sex is None:
        sampled_demo = sample_demographics()
        age = sampled_demo['age'] if age is None else age
        sex = sampled_demo['sex'] if sex is None else sex

    # Prepare demographic tensors
    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[sex]], dtype=torch.long).to(device)

    # Special token IDs
    bos_token_id = tokenizer.bos_token_id
    v_token_id = tokenizer.convert_tokens_to_indices(["<v>"])[0]
    v_end_token_id = tokenizer.convert_tokens_to_indices(["<\\v>"])[0]
    end_token_id = tokenizer.convert_tokens_to_indices(["<END>"])[0]

    # Extract target structure
    num_visits = target_structure['num_visits']
    codes_per_visit = target_structure['codes_per_visit']

    # Handle case with no visits
    if num_visits == 0 or len(codes_per_visit) == 0:
        return {
            'generated_visits': [],
            'demographics': {'age': age, 'sex': sex},
            'num_visits': 0,
            'num_codes': 0,
            'target_structure': target_structure
        }

    # Initialize generation with empty sequence
    # HuggingFace will prepend decoder_start_token_id (</s>) automatically
    # This matches training pattern: [</s>, <v>, codes...] after first <v> is appended
    decoder_input_ids = torch.tensor([[]], dtype=torch.long).to(device)

    # If first_code provided, prepopulate decoder with <v> + first_code (no </v>)
    # This starts visit 0 with the sampled first code, then continues generating
    first_visit_prepopulated = False
    if first_code is not None and first_code in tokenizer.vocab.code2idx:
        v_token_id_temp = tokenizer.convert_tokens_to_indices(["<v>"])[0]
        first_code_id = tokenizer.vocab.code2idx[first_code]

        # Add <v>, first_code to decoder_input_ids (NO </v> yet - let generation continue)
        prepop_ids = torch.tensor([[v_token_id_temp, first_code_id]],
                                   dtype=torch.long).to(device)
        decoder_input_ids = torch.cat([decoder_input_ids, prepop_ids], dim=1)
        first_visit_prepopulated = True

    # Create dummy encoder input
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    all_visits = []

    with torch.no_grad():
        for visit_idx in range(num_visits):
            target_codes = min(codes_per_visit[visit_idx], max_codes_per_visit)

            # For visit 0 with prepopulated first_code, reduce target by 1 since we already have 1 code
            if visit_idx == 0 and first_visit_prepopulated:
                target_codes = max(1, target_codes - 1)  # At least 1 more code

            # Skip if target is too small
            if target_codes < 1:
                continue

            # Append <v> token to start visit
            v_token_tensor = torch.tensor([[v_token_id]], dtype=torch.long).to(device)
            decoder_input_ids = torch.cat([decoder_input_ids, v_token_tensor], dim=1)

            # Calculate max tokens to generate for this visit
            # Each code is ~1 token, plus 1 for </v>
            # Add 50% buffer for flexibility
            max_new_tokens_this_visit = int(target_codes * 1.5) + 1

            try:
                # Generate codes for this visit
                generated_visit_ids = model.generate(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    x_num=x_num,
                    x_cat=x_cat,
                    max_new_tokens=max_new_tokens_this_visit,
                    do_sample=True,
                    num_beams=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    no_repeat_ngram_size=1,
                    eos_token_id=v_end_token_id,  # Stop at visit end
                    pad_token_id=tokenizer.pad_token_id
                    # Note: NOT passing bos_token_id - let BART use decoder_start_token_id (</s>) automatically
                )

                # Extract only the newly generated tokens (after decoder_input_ids)
                new_tokens = generated_visit_ids[0, decoder_input_ids.shape[1]:]

                # Parse the generated visit codes
                visit_codes = []
                for token_id in new_tokens:
                    token_id_val = token_id.item()
                    if token_id_val == v_end_token_id:
                        break  # End of visit
                    elif token_id_val >= tokenizer.code_offset:
                        # Diagnosis code - token_id_val is already the correct vocab index
                        # FIX: code2idx already includes special tokens, so don't subtract offset
                        if token_id_val < len(tokenizer.vocab.idx2code):
                            code = tokenizer.vocab.idx2code[token_id_val]
                            visit_codes.append(code)

                # If we generated codes, add visit
                if len(visit_codes) > 0:
                    # Truncate to target if we over-generated
                    if len(visit_codes) > target_codes:
                        visit_codes = visit_codes[:target_codes]

                    all_visits.append(visit_codes)

                    # Update decoder_input_ids with the full visit (including </v>)
                    # Reconstruct the visit tokens
                    visit_token_ids = [v_token_id]  # <v>
                    for code in visit_codes:
                        if code in tokenizer.vocab.code2idx:
                            # FIX: code2idx already returns token ID with offset included
                            code_token_id = tokenizer.vocab.code2idx[code]
                            visit_token_ids.append(code_token_id)
                    visit_token_ids.append(v_end_token_id)  # </v>

                    # Convert to tensor and concatenate (skip first <v> since already added)
                    visit_tensor = torch.tensor([visit_token_ids[1:]], dtype=torch.long).to(device)
                    decoder_input_ids = torch.cat([decoder_input_ids, visit_tensor], dim=1)

            except Exception as e:
                # If generation fails for this visit, skip it
                print(f"Warning: Generation failed for visit {visit_idx + 1}: {e}")
                continue

            # Check if we're approaching context limit (512 for BART)
            if decoder_input_ids.shape[1] > 400:
                break  # Stop generating more visits

    # Compute statistics
    total_codes = sum(len(visit) for visit in all_visits)

    return {
        'generated_visits': all_visits,
        'demographics': {'age': age, 'sex': sex},
        'num_visits': len(all_visits),
        'num_codes': total_codes,
        'target_structure': target_structure
    }


def generate_with_frequency_prior(
    model,
    tokenizer,
    device: torch.device,
    target_structure: dict,
    frequency_prior: torch.Tensor,
    alpha: float = 1.0,
    age: Optional[float] = None,
    sex: Optional[int] = None,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.95,
    max_codes_per_visit: int = 25,
    diagnostic_mode: bool = False,
    diagnostic_path: Optional[str] = None
) -> dict:
    """Generate patient with frequency-guided sampling.

    This function is identical to generate_patient_with_structure_constraints,
    but blends model logits with training frequency prior for realistic code distributions.

    Args:
        model: Trained PromptBartModel or PromptEHR model.
        tokenizer: DiagnosisCodeTokenizer instance.
        device: Device to run on.
        target_structure: Dict with 'num_visits' and 'codes_per_visit' list.
        frequency_prior: [vocab_size] log-frequency tensor from build_frequency_prior().
        alpha: Blending weight (0=pure model, higher=more frequency guidance).
            Recommended: 0.5-2.0. Start with 1.0.
        age: Patient age (if None, sampled from distribution).
        sex: Patient sex ID (0=M, 1=F; if None, sampled).
        temperature: Sampling temperature (default: 0.7).
        top_k: Top-k sampling parameter (default: 0 = disabled).
        top_p: Nucleus sampling parameter (default: 0.95).
        max_codes_per_visit: Maximum codes per visit safety cap (default: 25).
        diagnostic_mode: Enable detailed logging of generation process (default: False).
        diagnostic_path: Path to save diagnostic JSON file (required if diagnostic_mode=True).

    Returns:
        Dictionary with:
            - 'generated_visits': List[List[str]] of diagnosis codes
            - 'demographics': dict with 'age' and 'sex'
            - 'num_visits': int
            - 'num_codes': int
            - 'target_structure': dict (the structure we aimed for)
            - 'alpha': float (frequency prior weight used)
            - 'diagnostics': dict (if diagnostic_mode=True) with detailed generation logs

    Example:
        >>> prior = build_frequency_prior(tokenizer, './promptehr_outputs/training_frequencies.json')
        >>> result = generate_with_frequency_prior(
        ...     model, tokenizer, device,
        ...     target_structure={'num_visits': 3, 'codes_per_visit': [5, 8, 6]},
        ...     frequency_prior=prior,
        ...     alpha=1.0
        ... )
    """
    model.eval()

    # Sample demographics if not provided
    if age is None or sex is None:
        sampled_demo = sample_demographics()
        age = sampled_demo['age'] if age is None else age
        sex = sampled_demo['sex'] if sex is None else sex

    # Prepare demographic tensors
    x_num = torch.tensor([[age]], dtype=torch.float32).to(device)
    x_cat = torch.tensor([[sex]], dtype=torch.long).to(device)

    # Move frequency prior to device
    frequency_prior = frequency_prior.to(device)

    # Special token IDs
    bos_token_id = tokenizer.bos_token_id
    v_token_id = tokenizer.convert_tokens_to_indices(["<v>"])[0]
    v_end_token_id = tokenizer.convert_tokens_to_indices(["<\\v>"])[0]

    # Extract target structure
    num_visits = target_structure['num_visits']
    codes_per_visit = target_structure['codes_per_visit']

    # Handle case with no visits
    if num_visits == 0 or len(codes_per_visit) == 0:
        return {
            'generated_visits': [],
            'demographics': {'age': age, 'sex': sex},
            'num_visits': 0,
            'num_codes': 0,
            'target_structure': target_structure,
            'alpha': alpha
        }

    # Initialize generation with empty sequence
    # HuggingFace will prepend decoder_start_token_id (</s>) automatically
    # This matches training pattern: [</s>, <v>, codes...] after first <v> is appended
    decoder_input_ids = torch.tensor([[]], dtype=torch.long).to(device)

    # Create dummy encoder input
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    all_visits = []

    # Initialize diagnostic tracking
    all_diagnostics = {'visits': []} if diagnostic_mode else None

    with torch.no_grad():
        for visit_idx in range(num_visits):
            target_codes = min(codes_per_visit[visit_idx], max_codes_per_visit)

            # Skip if target is too small
            if target_codes < 1:
                continue

            # Append <v> token to start visit
            v_token_tensor = torch.tensor([[v_token_id]], dtype=torch.long).to(device)
            decoder_input_ids = torch.cat([decoder_input_ids, v_token_tensor], dim=1)

            # Generate codes for this visit with frequency guidance
            max_new_tokens_this_visit = int(target_codes * 1.5) + 1
            visit_codes = []

            # Initialize visit diagnostic tracking
            visit_diagnostics = {'visit_idx': visit_idx, 'steps': []} if diagnostic_mode else None

            for step in range(max_new_tokens_this_visit):
                # Forward pass
                outputs = model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    x_num=x_num,
                    x_cat=x_cat,
                    return_dict=True
                )

                # Get logits for next token (handle both dict and object outputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[0, -1, :]  # [vocab_size]
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    logits = outputs['logits'][0, -1, :]  # [vocab_size]
                else:
                    raise TypeError(f"Unexpected output type: {type(outputs)}")

                # Diagnostic logging: raw model logits
                if diagnostic_mode:
                    step_diagnostics = {
                        'step': step,
                        'raw_logits': {
                            'max': float(logits.max()),
                            'min': float(logits.min()),
                            'mean': float(logits.mean()),
                            'std': float(logits.std()),
                            'top_5_indices': [int(i) for i in logits.topk(5).indices],
                            'top_5_codes': [tokenizer.vocab.idx2code.get(int(i), f"<{i}>")
                                           for i in logits.topk(5).indices],
                            'top_5_values': [float(v) for v in logits.topk(5).values]
                        }
                    }

                # BLEND with frequency prior
                logits_guided = logits + alpha * frequency_prior

                # Diagnostic logging: frequency blending
                if diagnostic_mode:
                    step_diagnostics['blending'] = {
                        'alpha': alpha,
                        'prior_contribution': float((alpha * frequency_prior).abs().mean()),
                        'logits_shift': float((logits_guided - logits).abs().mean()),
                        'top_5_after_blend_indices': [int(i) for i in logits_guided.topk(5).indices],
                        'top_5_after_blend_codes': [tokenizer.vocab.idx2code.get(int(i), f"<{i}>")
                                                     for i in logits_guided.topk(5).indices],
                        'top_5_after_blend_values': [float(v) for v in logits_guided.topk(5).values]
                    }

                # Apply temperature
                scaled_logits = logits_guided / temperature

                # Convert to probabilities
                probs = torch.softmax(scaled_logits, dim=0)

                # Diagnostic logging: probabilities after temperature
                if diagnostic_mode:
                    top_probs, top_indices = torch.topk(probs, 20)
                    step_diagnostics['probabilities'] = {
                        'temperature': temperature,
                        'entropy': float(-(probs * torch.log(probs + 1e-10)).sum()),
                        'top_20': [
                            {'code': tokenizer.vocab.idx2code.get(int(idx), f"<{idx}>"),
                             'prob': float(prob),
                             'idx': int(idx)}
                            for idx, prob in zip(top_indices, top_probs)
                        ]
                    }

                # Apply top-k filtering if enabled
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(0, top_k_indices, top_k_vals)
                    probs = probs_filtered / probs_filtered.sum()

                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                    nucleus_mask = cumsum_probs <= top_p
                    nucleus_mask[0] = True  # Always include top token

                    nucleus_indices = sorted_indices[nucleus_mask]
                    nucleus_probs = sorted_probs[nucleus_mask]
                    nucleus_probs = nucleus_probs / nucleus_probs.sum()

                    # Sample from nucleus
                    sampled_idx = torch.multinomial(nucleus_probs, 1)[0]
                    next_token = int(nucleus_indices[sampled_idx])
                else:
                    # Sample directly from filtered probs
                    next_token = int(torch.multinomial(probs, 1)[0])

                # Diagnostic logging: sampling decision
                if diagnostic_mode:
                    selected_code = tokenizer.vocab.idx2code.get(next_token, f"<{next_token}>")
                    step_diagnostics['selected'] = {
                        'token': next_token,
                        'code': selected_code,
                        'probability': float(probs[next_token]) if next_token < len(probs) else 0.0,
                        'was_top_1': (next_token == int(probs.argmax())),
                        'is_special_token': next_token < tokenizer.code_offset
                    }
                    visit_diagnostics['steps'].append(step_diagnostics)

                # Check if we hit end-of-visit
                if next_token == v_end_token_id:
                    break

                # Extract code if it's a diagnosis code
                # FIX: code2idx already includes special tokens, so don't subtract offset
                if next_token >= tokenizer.code_offset:
                    if next_token < len(tokenizer.vocab.idx2code):
                        code = tokenizer.vocab.idx2code[next_token]
                        if code not in visit_codes:  # Prevent duplicates
                            visit_codes.append(code)

                # Append token to decoder input
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_tensor], dim=1)

                # Stop if we have enough codes
                if len(visit_codes) >= target_codes:
                    break

            # Add visit if we generated codes
            if len(visit_codes) > 0:
                # Truncate to target if over-generated
                if len(visit_codes) > target_codes:
                    visit_codes = visit_codes[:target_codes]

                all_visits.append(visit_codes)

                # Add visit diagnostics
                if diagnostic_mode:
                    visit_diagnostics['generated_codes'] = visit_codes
                    visit_diagnostics['target_codes'] = target_codes
                    all_diagnostics['visits'].append(visit_diagnostics)

                # Append </v> to close visit
                v_end_tensor = torch.tensor([[v_end_token_id]], dtype=torch.long).to(device)
                decoder_input_ids = torch.cat([decoder_input_ids, v_end_tensor], dim=1)

            # Check if we're approaching context limit
            if decoder_input_ids.shape[1] > 400:
                break

    # Compute statistics
    total_codes = sum(len(visit) for visit in all_visits)

    # Build result dictionary
    result = {
        'generated_visits': all_visits,
        'demographics': {'age': age, 'sex': sex},
        'num_visits': len(all_visits),
        'num_codes': total_codes,
        'target_structure': target_structure,
        'alpha': alpha
    }

    # Add diagnostics if enabled
    if diagnostic_mode:
        all_diagnostics['demographics'] = {'age': age, 'sex': sex}
        all_diagnostics['params'] = {
            'alpha': alpha,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }
        all_diagnostics['generated_codes'] = all_visits
        result['diagnostics'] = all_diagnostics

        # Save diagnostics to file if path provided
        if diagnostic_path:
            import json
            import os
            os.makedirs(os.path.dirname(diagnostic_path), exist_ok=True)
            with open(diagnostic_path, 'w') as f:
                json.dump(all_diagnostics, f, indent=2)

    return result
