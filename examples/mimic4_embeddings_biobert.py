"""Example script for generating embeddings from MIMIC-IV text data using BioBERT.

This script demonstrates how to:
1. Load MIMIC-IV dataset and extract text using TextExtractionMIMIC4 task
2. Load a pretrained biomedical language model (BioClinical-ModernBERT)
3. Generate embeddings from extracted text samples
4. Store embeddings with metadata for downstream analysis

The script uses the TextExtractionMIMIC4 task to extract structured text from
EHR tables (labevents, prescriptions) and then generates embeddings using
a pretrained transformer model. This workflow is useful for:
- Creating patient/visit representations for downstream ML tasks
- Building semantic search over clinical text
- Transfer learning from clinical text to other healthcare tasks

Example usage:
    python mimic4_embeddings_biobert.py

The script will process a subset of patients and generate
embeddings for all extracted text samples.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Any, Optional

# For local development: run below code if pyhealth is not installed
# import sys
# from pathlib import Path

# script_dir = Path(__file__).parent
# parent_dir = script_dir.parent
# sys.path.insert(0, str(parent_dir))

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import TextExtractionMIMIC4


def get_embedding(
    text: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 512,
) -> Optional[np.ndarray]:
    """Extract embedding vector from text using a pretrained transformer model.

    Called for each text sample to generate a fixed-size embedding vector.
    The function tokenizes the input text, runs it through the model, and
    extracts the embedding using mean pooling over the sequence length.

    Args:
        text (str): Input text string to embed. Should be the extracted text
            from EHR events (e.g., "labevents label Glucose labevents value 120").
            If empty or not a string, returns None.
        model (torch.nn.Module): Pretrained transformer model (e.g., AutoModel
            from transformers). Should be in evaluation mode and on the
            specified device.
        tokenizer (Any): Pretrained tokenizer (e.g., AutoTokenizer from
            transformers) corresponding to the model. Used to tokenize input
            text into model-compatible format.
        device (torch.device): Device to run inference on (e.g., "cuda" or
            "cpu"). Model and inputs will be moved to this device.
        max_length (int): Maximum sequence length for tokenization. Longer
            sequences will be truncated. Defaults to 512 tokens.

    Returns:
        Optional[np.ndarray]: Numpy array of shape (embedding_dim,) containing
            the mean-pooled embedding vector. Returns None if text is empty or
            not a string. The embedding dimension depends on the model (e.g.,
            768 for BERT-base models).

    Note:
        The embedding is computed as the mean of the last hidden state across
        the sequence dimension.
    """
    if not text or not isinstance(text, str):
        return None

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embedding (mean pooling of last hidden state)
    embedding = outputs.last_hidden_state.mean(dim=1)

    # Convert to numpy
    embedding = embedding.cpu().detach().numpy()[0]

    return embedding


# Step 1: Load Dataset
# Load MIMIC-IV dataset with required tables for text extraction
dataset = MIMIC4Dataset(
    ehr_root="https://physionet.org/files/mimic-iv-demo/2.2/",
    ehr_tables=["admissions", "labevents", "prescriptions"],
    dev=True,
)

# Step 2: Load pretrained model
# Load BioClinical-ModernBERT model and tokenizer for generating embeddings
MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Model and tokenizer loaded successfully!")

# Step 3: Process dataset and generate embeddings
# Extract text from EHR tables using TextExtractionMIMIC4 task
text_extraction_task = TextExtractionMIMIC4(max_patients=10)

sample_dataset = dataset.set_task(
    text_extraction_task,
    num_workers=1,
    cache_dir=None,
)

print(f"Total samples extracted: {len(sample_dataset)}")
if sample_dataset.samples:
    sample_keys = list(sample_dataset.samples[0].keys())
    print(f"Sample keys: {sample_keys}")

# Generate embeddings for each extracted text sample
all_embeddings = []
all_metadata = []

for sample in tqdm(sample_dataset.samples, desc="Generating embeddings"):
    embedding = get_embedding(sample["text"], model, tokenizer, device)

    if embedding is not None:
        all_embeddings.append(embedding)
        all_metadata.append(
            {
                "patient_id": sample["patient_id"],
                "visit_id": sample.get("visit_id"),
                "event_type": sample["event_type"],
                "text": sample["text"],
            }
        )

print(f"\nGenerated {len(all_embeddings)} embeddings using task approach")
print(f"Embedding dimension: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")

# Step 4: Check Results
# Display sample embeddings and metadata for verification
print("Sample embeddings and metadata:\n")
for i in range(min(5, len(all_metadata))):
    print(f"Example {i+1}:")
    print(f"  Patient ID: {all_metadata[i]['patient_id']}")
    print(f"  Visit ID: {all_metadata[i]['visit_id']}")
    print(f"  Event Type: {all_metadata[i]['event_type']}")
    print(f"  Text: {all_metadata[i]['text'][:100]}...")
    print(f"  Embedding shape: {all_embeddings[i].shape}")
    print(f"  Embedding (first 5 values): {all_embeddings[i][:5]}")
    print()
