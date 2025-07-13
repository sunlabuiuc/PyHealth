#!/usr/bin/env python3
"""
Generate synthetic patient records from MIMIC-III data using CorGAN.

Usage:
    python examples/synthetic_data_generation_mimic3_corgan.py \
        --data_path ./data_files \
        --output_path ./synthetic_results \
        --n_samples 1000
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from pyhealth.models.generators.corgan import (
    CorGANAutoencoder, CorGANGenerator, CorGANDiscriminator, 
    CorGANDataset, weights_init
)


def load_mimic3_data(data_path):
    """Load MIMIC-III diagnosis data using native ICD-9 codes."""
    print("Loading MIMIC-III data...")
    
    # Diagnosis, in ICD-9
    diagnoses_df = pd.read_csv(os.path.join(data_path, "DIAGNOSES_ICD.csv.gz"))
    diagnoses_df = diagnoses_df.dropna(subset=['ICD9_CODE'])
    diagnoses_df['ICD9_CODE'] = diagnoses_df['ICD9_CODE'].astype(str)
    
    # for each row in diagnosis, get the subject ID and add ICD code to the patient agg
    patient_codes = defaultdict(set)
    for _, row in diagnoses_df.iterrows():
        patient_codes[row['SUBJECT_ID']].add(row['ICD9_CODE'])
    
    # Binary matrix
    all_codes = sorted(set().union(*patient_codes.values()))
    patients = sorted(patient_codes.keys())
    
    binary_matrix = np.zeros((len(patients), len(all_codes)), dtype=np.float32)
    code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
    
    for patient_idx, patient_id in enumerate(patients):
        for code in patient_codes[patient_id]:
            binary_matrix[patient_idx, code_to_idx[code]] = 1.0
    
    print(f"Loaded {len(patients)} patients, {len(all_codes)} ICD-9 codes")
    return binary_matrix, all_codes


def train_corgan(data, n_codes, device, args):
    """Train CorGAN model."""
    # Initialize models
    autoencoder = CorGANAutoencoder(feature_size=n_codes).to(device)
    
    # Get encoder output dimension
    with torch.no_grad():
        test_input = torch.randn(1, 1, n_codes).to(device)
        latent_dim = autoencoder.encoder(test_input).view(1, -1).shape[1]
    
    generator = CorGANGenerator(latent_dim=args.latent_dim, hidden_dim=latent_dim).to(device)
    discriminator = CorGANDiscriminator(input_dim=latent_dim, hidden_dim=256).to(device)
    
    # weights init
    for model in [autoencoder, generator, discriminator]:
        model.apply(weights_init)
    
    # CG dataset and dataloader
    dataset = CorGANDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # A training
    print("Training autoencoder...")
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    
    for epoch in range(args.ae_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"AE Epoch {epoch+1}"):
            batch = batch.to(device)
            
            # Forward pass
            reconstructed = autoencoder(batch.unsqueeze(1)).squeeze(1)
            
            # Handle dimension mismatch
            if reconstructed.shape[1] != n_codes:
                if reconstructed.shape[1] > n_codes:
                    reconstructed = reconstructed[:, :n_codes]
                else:
                    padding = torch.zeros(reconstructed.shape[0], n_codes - reconstructed.shape[1], device=device)
                    reconstructed = torch.cat([reconstructed, padding], dim=1)
            
            loss = F.binary_cross_entropy(reconstructed, batch)
            
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            
            total_loss += loss.item()
        
        print(f"AE Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")
    
    # Train GAN
    print("Training GAN...")
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    
    autoencoder.eval()
    
    for epoch in range(args.gan_epochs):
        g_loss_total = d_loss_total = 0
        
        for batch in tqdm(dataloader, desc=f"GAN Epoch {epoch+1}"):
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # D training
            for _ in range(5):  # WGAN: train D more than G
                d_optimizer.zero_grad()
                
                # Real data
                with torch.no_grad():
                    real_encoded = autoencoder.encoder(batch.unsqueeze(1))
                real_pred = discriminator(real_encoded.view(batch_size, -1))
                d_loss_real = -torch.mean(real_pred)
                
                # Fake data
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake_encoded = generator(z)
                fake_pred = discriminator(fake_encoded.detach())
                d_loss_fake = torch.mean(fake_pred)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # clip weights (WGAN)
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            # G training
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_encoded = generator(z)
            fake_pred = discriminator(fake_encoded)
            g_loss = -torch.mean(fake_pred)
            
            g_loss.backward()
            g_optimizer.step()
            
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
        
        print(f"GAN Epoch {epoch+1}: G={g_loss_total/len(dataloader):.3f}, D={d_loss_total/len(dataloader):.3f}")
    
    return autoencoder, generator


def generate_synthetic_data(autoencoder, generator, n_samples, n_codes, device, args):
    """Generate synthetic patient data"""
    print(f"Generating {n_samples} synthetic patients...")
    
    autoencoder.eval()
    generator.eval()
    
    synthetic_data = []
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            current_batch_size = min(args.batch_size, n_samples - i * args.batch_size)
            
            # Generate latent codes
            z = torch.randn(current_batch_size, args.latent_dim, device=device)
            fake_encoded = generator(z)
            
            # reshape fake encoded codes
            decoder_channels = 128
            latent_width = max(1, fake_encoded.shape[1] // decoder_channels)
            fake_reshaped = fake_encoded[:, :decoder_channels * latent_width].view(
                current_batch_size, decoder_channels, latent_width
            )
            
            # Decode to medical codes
            decoded = autoencoder.decoder(fake_reshaped)
            if decoded.dim() == 3:
                decoded = decoded.squeeze(-1)
            if decoded.dim() == 1:
                decoded = decoded.unsqueeze(0)
            
            if decoded.shape[1] > n_codes:
                decoded = decoded[:, :n_codes]
            
            # sigmoid and threshold
            synthetic_batch = torch.sigmoid(decoded)
            binary_batch = (synthetic_batch >= 0.5).float()
            synthetic_data.append(binary_batch.cpu().numpy())
    
    return np.vstack(synthetic_data)[:n_samples]


def main():
    parser = argparse.ArgumentParser(description="CorGAN synthetic EHR generation")
    parser.add_argument("--data_path", required=True, help="Path to MIMIC-III data")
    parser.add_argument("--output_path", required=True, help="Output directory")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of synthetic samples")
    parser.add_argument("--ae_epochs", type=int, default=10, help="Autoencoder epochs")
    parser.add_argument("--gan_epochs", type=int, default=50, help="GAN epochs")
    parser.add_argument("--latent_dim", type=int, default=128, help="Generator latent dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    binary_matrix, code_vocab = load_mimic3_data(args.data_path)
    _, n_codes = binary_matrix.shape
    autoencoder, generator = train_corgan(binary_matrix, n_codes, device, args)
    
    # Generate data
    synthetic_matrix = generate_synthetic_data(
        autoencoder, generator, args.n_samples, n_codes, device, args
    )
    
    print("Saving results...")
    np.save(os.path.join(args.output_path, "synthetic_data.npy"), synthetic_matrix)
    
    # convert to CSV
    synthetic_df = pd.DataFrame(synthetic_matrix, columns=code_vocab)
    synthetic_df['patient_id'] = [f'synthetic_{i}' for i in range(len(synthetic_matrix))]
    synthetic_df.to_csv(os.path.join(args.output_path, "synthetic_data.csv"), index=False)
    
    # vocabulary
    with open(os.path.join(args.output_path, "icd9_vocabulary.txt"), 'w') as f:
        for code in code_vocab:
            f.write(f"{code}\n")
    
    # model checkpoints
    torch.save({
        'autoencoder': autoencoder.state_dict(),
        'generator': generator.state_dict(),
        'config': vars(args)
    }, os.path.join(args.output_path, "corgan_models.pth"))


if __name__ == "__main__":
    main()

"""
SLURM SCRIPT:

#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=corgan_mimic3
#SBATCH --output=logs/corgan_mimic3_%j.out
#SBATCH --error=logs/corgan_mimic3_%j.err
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd "$SLURM_SUBMIT_DIR"

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

cd /u/jalenj4/PyHealth
source pyhealth/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth/PyHealth:$PYTHONPATH

mkdir -p logs corgan_results

echo "Starting CorGAN synthetic EHR generation..."
python examples/synthetic_data_generation_mimic3_corgan.py \
    --data_path ./data_files \
    --output_path ./corgan_results \
    --ae_epochs 10 \
    --gan_epochs 50 \
    --n_samples 1000 \
    --batch_size 128 \
    --latent_dim 128 \
    --lr 0.0001

echo "CorGAN training completed!"
echo "Results saved to: ./corgan_results/"
"""