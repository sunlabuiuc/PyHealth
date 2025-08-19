import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# Add the parent directory to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pyhealth.models import CADRETransformer
from pyhealth.datasets.sample_dataset import SampleDataset

if __name__ == '__main__':
    # --- Example Data Generation ---
    batch_size = 4
    num_genes = 150
    emb_dim = 200
    num_drugs = 10
    seq_len = 1500

    # Create a example gene embedding matrix
    example_gene_emb_matrix = torch.randn(num_genes, emb_dim)

    # Create example sample data
    example_samples = [
        {'gene_indices': torch.randint(0, num_genes, (seq_len,)), 'drug_ids': torch.randint(0, num_drugs, (1,)).item(), 'response': 1.0}
        for _ in range(batch_size)
    ]

    # Define the input and output schema for the example dataset
    input_schema = {'gene_indices': 'raw', 'drug_ids': 'raw'}
    output_schema = {'response': 'raw'}  # Example output schema

    # Create a example SampleDataset
    example_dataset = SampleDataset(samples=example_samples, input_schema=input_schema, output_schema=output_schema)

    # --- Test Case 1: Model Initialization ---
    print("--- Test Case 1: Model Initialization ---")
    try:
        model = CADRETransformer(dataset=example_dataset, gene_emb_matrix=example_gene_emb_matrix, num_drugs=num_drugs, emb_dim=emb_dim)
        print("CADRETransformer initialized successfully!")
    except Exception as e:
        print(f"Error during model initialization: {e}")

    # --- Test Case 2: Forward Pass with Single Batch ---
    print("\n--- Test Case 2: Forward Pass with Single Batch ---")
    try:
        dataloader = DataLoader(example_dataset, batch_size=batch_size)
        batch = next(iter(dataloader))

        output = model(batch)
        print("Forward pass successful!")
        print("Output shape:", output.shape)
        print("Output:", output)

        # Assertions for output shape and type
        assert output.shape == (batch_size,), f"Output shape should be {(batch_size,)}, but got {output.shape}"
        assert output.dtype == torch.float32, f"Output dtype should be torch.float32, but got {output.dtype}"
        print("Basic output assertions passed!")

    except Exception as e:
        print(f"Error during forward pass: {e}")

    # --- Test Case 3: Forward Pass with Different Pooling ---
    print("\n--- Test Case 3: Forward Pass with Different Pooling ---")
    try:
        model_max_pool = CADRETransformer(dataset=example_dataset, gene_emb_matrix=example_gene_emb_matrix, num_drugs=num_drugs, emb_dim=emb_dim, pooling='max')
        dataloader = DataLoader(example_dataset, batch_size=batch_size)
        batch = next(iter(dataloader))
        output_max_pool = model_max_pool(batch)
        print("Forward pass with max pooling successful!")
        print("Output shape (max pool):", output_max_pool.shape)
        assert output_max_pool.shape == (batch_size,), f"Output shape (max pool) should be {(batch_size,)}, but got {output_max_pool.shape}"
        assert output_max_pool.dtype == torch.float32, f"Output dtype (max pool) should be torch.float32, but got {output_max_pool.dtype}"
        print("Basic output assertions (max pool) passed!")
    except Exception as e:
        print(f"Error during forward pass with max pooling: {e}")

    # --- Test Case 4: Handling Different Batch Sizes ---
    print("\n--- Test Case 4: Handling Different Batch Sizes ---")
    try:
        batch_size_small = 2
        dataloader_small = DataLoader(example_dataset, batch_size=batch_size_small)
        batch_small = next(iter(dataloader_small))
        output_small = model(batch_small)
        print(f"Forward pass with batch size {batch_size_small} successful!")
        print("Output shape (small batch):", output_small.shape)
        assert output_small.shape == (batch_size_small,), f"Output shape (small batch) should be {(batch_size_small,)}, but got {output_small.shape}"
        assert output_small.dtype == torch.float32, f"Output dtype (small batch) should be torch.float32, but got {output_small.dtype}"
        print("Basic output assertions (small batch) passed!")
    except Exception as e:
        print(f"Error during forward pass with small batch size: {e}")