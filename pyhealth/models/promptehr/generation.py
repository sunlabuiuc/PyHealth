"""
Generate synthetic patient sequences using trained PromptEHR model.

This module provides functions for generating realistic synthetic EHR data
using various conditioning strategies (demographics, visit structures, etc.).
"""
import numpy as np
import torch
from typing import Optional, List


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

    Extracts visits by splitting at <v> and <\v> markers, and decodes
    diagnosis codes within each visit.

    Args:
        token_ids: List of token IDs from model generation.
        tokenizer: DiagnosisCodeTokenizer instance (must have convert_tokens_to_ids,
            bos_token_id, pad_token_id, code_offset, and vocab attributes).

    Returns:
        List of visits, where each visit is a list of ICD-9 code strings.

    Example:
        Input: [BOS, <v>, 401.9, 250.00, <\v>, <v>, 428.0, <\v>, <END>]
        Output: [['401.9', '250.00'], ['428.0']]
    """
    visits = []
    current_visit_codes = []

    # Special token IDs
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")
    bos_token_id = tokenizer.bos_token_id
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")

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
            # Diagnosis code token
            code_idx = token_id - tokenizer.code_offset
            if code_idx < len(tokenizer.vocab):
                code = tokenizer.vocab.idx2code[code_idx]
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
    top_k: int = 40,
    top_p: float = 0.9,
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
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")

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
                # Codes are in vocab, need to add code_offset to get token ID
                code_idx = tokenizer.vocab.code2idx[code]
                code_token_id = tokenizer.code_offset + code_idx
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

            # Extract code tokens (skip BOS, <v>, <\v>)
            generated_code_ids = [
                tid for tid in visit_token_ids
                if tid >= tokenizer.code_offset
            ]

            # Decode codes (convert token IDs back to diagnosis codes)
            generated_codes = []
            for tid in generated_code_ids:
                code_idx = tid - tokenizer.code_offset
                if code_idx < len(tokenizer.vocab):
                    code = tokenizer.vocab.idx2code[code_idx]
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
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
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
    v_token_id = tokenizer.convert_tokens_to_ids("<v>")
    v_end_token_id = tokenizer.convert_tokens_to_ids("<\\v>")
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")

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

    # Initialize generation with BOS token
    decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

    # Create dummy encoder input
    encoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids)

    all_visits = []

    with torch.no_grad():
        for visit_idx in range(num_visits):
            target_codes = min(codes_per_visit[visit_idx], max_codes_per_visit)

            # Skip if target is too small
            if target_codes < 1:
                continue

            # Append <v> token to start visit
            v_token_tensor = torch.tensor([[v_token_id]], dtype=torch.long).to(device)
            decoder_input_ids = torch.cat([decoder_input_ids, v_token_tensor], dim=1)

            # Calculate max tokens to generate for this visit
            # Each code is ~1 token, plus 1 for <\v>
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
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=bos_token_id
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
                        # Diagnosis code
                        code_idx = token_id_val - tokenizer.code_offset
                        if code_idx < len(tokenizer.vocab):
                            code = tokenizer.vocab.idx2code[code_idx]
                            visit_codes.append(code)

                # If we generated codes, add visit
                if len(visit_codes) > 0:
                    # Truncate to target if we over-generated
                    if len(visit_codes) > target_codes:
                        visit_codes = visit_codes[:target_codes]

                    all_visits.append(visit_codes)

                    # Update decoder_input_ids with the full visit (including <\v>)
                    # Reconstruct the visit tokens
                    visit_token_ids = [v_token_id]  # <v>
                    for code in visit_codes:
                        if code in tokenizer.vocab.code2idx:
                            code_idx = tokenizer.vocab.code2idx[code]
                            code_token_id = code_idx + tokenizer.code_offset
                            visit_token_ids.append(code_token_id)
                    visit_token_ids.append(v_end_token_id)  # <\v>

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
