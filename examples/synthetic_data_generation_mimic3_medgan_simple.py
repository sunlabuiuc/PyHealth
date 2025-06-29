"""
Synthetic data generation using MedGAN on MIMIC-III data.

This example demonstrates how to train MedGAN to generate synthetic phecode matrices
from MIMIC-III data, following PyHealth conventions.
"""

import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import pickle
import json
from tqdm import tqdm
import pandas as pd

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets.phecode_dataset import PhecodeDataset, PhecodeMatrixDataset
from pyhealth.models.generators.medgan import MedGAN

"""
python examples/synthetic_data_generation_mimic3_medgan.py --autoencoder_epochs 5 --gan_epochs 10 --batch_size 16
"""
def train_medgan(model, dataloader, n_epochs, device, save_dir, gen_freq=1, lr=0.001, weight_decay=0.0001, b1=0.5, b2=0.9):
    """
    Train MedGAN model using the original synthEHRella approach.
    
    Args:
        model: MedGAN model
        dataloader: DataLoader for training data
        n_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
        gen_freq: Frequency of generator training (1 = every batch, like original)
        lr: Learning rate
        weight_decay: Weight decay for regularization
        b1: Beta1 for Adam optimizer
        b2: Beta2 for Adam optimizer
    
    Returns:
        loss_history: Dictionary containing loss history
    """
    # Use original synthEHRella loss functions
    def generator_loss(y_fake, y_true):
        """
        Original synthEHRella generator loss
        """
        epsilon = 1e-12
        return -0.5 * torch.mean(torch.log(y_fake + epsilon))
    
    def discriminator_loss(outputs, labels):
        """
        Original synthEHRella discriminator loss
        """
        loss = -torch.mean(labels * torch.log(outputs + 1e-12)) - torch.mean((1 - labels) * torch.log(1. - outputs + 1e-12))
        return loss
    
    # Optimizers (use original learning rates)
    optimizer_g = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.autoencoder.decoder.parameters(), 'lr': lr * 0.1}
    ], lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), 
                                  lr=lr * 0.1, betas=(b1, b2), weight_decay=weight_decay)
    
    # Loss tracking
    g_losses = []
    d_losses = []
    
    print("="*60)
    print("Epoch | D_loss | G_loss | Progress")
    print("="*60)
    
    for epoch in range(n_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Create labels (no smoothing, use original approach)
            valid = torch.ones(batch_size).to(device)  # 1D tensor like original
            fake = torch.zeros(batch_size).to(device)  # 1D tensor like original
            
            # Sample noise as generator input
            z = torch.randn(batch_size, model.latent_dim).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            
            # disable discriminator gradients for generator training
            for p in model.discriminator.parameters():
                p.requires_grad = False
            
            # generate fake samples
            fake_samples = model.generator(z)
            fake_samples = model.autoencoder.decode(fake_samples)
            
            # generator loss using original medgan loss function
            fake_output = model.discriminator(fake_samples).view(-1)
            g_loss = generator_loss(fake_output, valid)
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Enable discriminator gradients
            for p in model.discriminator.parameters():
                p.requires_grad = True
            
            optimizer_d.zero_grad()
            
            # Real samples
            real_output = model.discriminator(real_data).view(-1)
            real_loss = discriminator_loss(real_output, valid)
            real_loss.backward()
            
            # Fake samples (detached)
            fake_output = model.discriminator(fake_samples.detach()).view(-1)
            fake_loss = discriminator_loss(fake_output, fake)
            fake_loss.backward()
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            optimizer_d.step()
            
            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        # Store losses for tracking
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Print progress
        progress = (epoch + 1) / n_epochs * 100
        print(f"{epoch+1:5d} | {avg_d_loss:.4f} | {avg_g_loss:.4f} | {progress:5.1f}%")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"medgan_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': model.generator.state_dict(),
                'discriminator_state_dict': model.discriminator.state_dict(),
                'autoencoder_state_dict': model.autoencoder.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    print("="*60)
    print("GAN Training Completed!")
    print(f"Final G_loss: {g_losses[-1]:.4f}")
    print(f"Final D_loss: {d_losses[-1]:.4f}")
    
    # Save loss history
    loss_history = {
        'g_losses': g_losses,
        'd_losses': d_losses,
    }
    np.save(os.path.join(save_dir, "loss_history.npy"), loss_history)
    
    return loss_history


def postprocess_synthetic_data(synthetic_matrix, phecode_mapping, output_path):
    """
    Postprocess synthetic data to convert from raw codes to PhecodeXM format.
    This mimics the synthEHRella postprocessing pipeline.
    
    Args:
        synthetic_matrix: Raw synthetic data matrix (n_patients x n_raw_codes)
        phecode_mapping: Phecode mapping from PhecodeDataset
        output_path: Directory to save postprocessed data
    
    Returns:
        phecodexm_matrix: Postprocessed synthetic data in PhecodeXM format
    """
    print("Postprocessing synthetic data to PhecodeXM format...")
    
    # DEBUG: Print the full phecode_mapping structure
    print(f"\nDEBUG: phecode_mapping keys: {list(phecode_mapping.keys())}")
    print(f"DEBUG: phecode_mapping type: {type(phecode_mapping)}")
    
    # Get the mapping information
    icd9_to_icd10 = phecode_mapping.get('icd9_to_icd10', {})
    icd10_to_phecodex = phecode_mapping.get('icd10_to_phecodex', {})
    phecodex_to_phecodexm = phecode_mapping.get('phecodex_to_phecodexm', {})
    
    icd10_types = phecode_mapping.get('icd10_types', {})
    phecodex_types = phecode_mapping.get('phecodex_types', {})
    phecodexm_types = phecode_mapping.get('phecodexm_types', {})
    
    print(f"Raw synthetic data shape: {synthetic_matrix.shape}")
    print(f"ICD-10 types: {len(icd10_types)}")
    print(f"PhecodeX types: {len(phecodex_types)}")
    print(f"PhecodeXM types: {len(phecodexm_types)}")
    
    # DEBUG: Print mapping sizes
    print(f"DEBUG: icd9_to_icd10 mappings: {len(icd9_to_icd10)}")
    print(f"DEBUG: icd10_to_phecodex mappings: {len(icd10_to_phecodex)}")
    print(f"DEBUG: phecodex_to_phecodexm mappings: {len(phecodex_to_phecodexm)}")
    
    # DEBUG: Print sample mappings
    if icd9_to_icd10:
        print(f"DEBUG: Sample icd9_to_icd10: {list(icd9_to_icd10.items())[:3]}")
    if icd10_to_phecodex:
        print(f"DEBUG: Sample icd10_to_phecodex: {list(icd10_to_phecodex.items())[:3]}")
    if phecodex_to_phecodexm:
        print(f"DEBUG: Sample phecodex_to_phecodexm: {list(phecodex_to_phecodexm.items())[:3]}")
    
    # Check if synthetic data is already in PhecodeX format
    n_patients, n_codes = synthetic_matrix.shape
    if n_codes == len(phecodex_types):
        print(f"INFO: Synthetic data appears to be in PhecodeX format ({n_codes} codes)")
        print("Skipping ICD-9 to ICD-10 and ICD-10 to PhecodeX conversion")
        
        # Use synthetic data directly as PhecodeX matrix
        phecodex_matrix = synthetic_matrix
        print(f"DEBUG: Using synthetic data directly as PhecodeX matrix: {phecodex_matrix.shape}")
        
    else:
        print(f"INFO: Synthetic data appears to be in ICD-9 format ({n_codes} codes)")
        
        # DEBUG: Check if we have the expected data structure
        if not phecodexm_types:
            print("ERROR: phecodexm_types is empty! This means the mapping was not created correctly.")
            print("This could be because:")
            print("1. The PhecodeDataset was not created with use_phecode_mapping=True")
            print("2. The mapping files are missing or corrupted")
            print("3. The get_phecode_mapping() method has a bug")
            
            # Try to load the mapping files directly
            print("\nDEBUG: Attempting to load mapping files directly...")
            try:
                from pyhealth.datasets.phecode_dataset import PhecodeTransformer
                transformer = PhecodeTransformer()
                
                # Load PhecodeXM types directly
                phecodexm_types_path = os.path.join(transformer.mapping_dir, "phecodexm_types.json")
                with open(phecodexm_types_path, 'r') as f:
                    phecodexm_types = json.load(f)
                print(f"DEBUG: Loaded {len(phecodexm_types)} PhecodeXM types directly")
                
                # Load PhecodeX to PhecodeXM mapping directly
                phecodex_to_phecodexm_path = os.path.join(transformer.mapping_dir, "phecodex_to_phecodexm_mapping.json")
                with open(phecodex_to_phecodexm_path, 'r') as f:
                    phecodex_to_phecodexm_mapping = json.load(f)
                print(f"DEBUG: Loaded {len(phecodex_to_phecodexm_mapping)} PhecodeX to PhecodeXM mappings directly")
                
                # Create the mapping manually
                phecodex_to_phecodexm = {}
                for phecodex_code, phecodex_idx in transformer.phecodex_types_dict.items():
                    phecodex_idx_str = str(phecodex_idx)
                    if phecodex_idx_str in phecodex_to_phecodexm_mapping:
                        phecodexm_idx = phecodex_to_phecodexm_mapping[phecodex_idx_str]
                        phecodex_to_phecodexm[str(phecodex_idx)] = [phecodexm_idx]
                
                print(f"DEBUG: Created {len(phecodex_to_phecodexm)} PhecodeX to PhecodeXM mappings manually")
                
            except Exception as e:
                print(f"ERROR: Failed to load mapping files directly: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 1: Convert ICD-9 to ICD-10
        print("Step 1: Converting ICD-9 to ICD-10...")
        icd10_matrix = np.zeros((n_patients, len(icd10_types)), dtype=int)
        
        print(f"DEBUG: Step 1 - n_patients={n_patients}, n_icd9_codes={n_codes}, icd10_types={len(icd10_types)}")
        
        for icd9_idx in tqdm(range(n_codes), desc="ICD-9 to ICD-10"):
            if str(icd9_idx) in icd9_to_icd10:
                for icd10_code in icd9_to_icd10[str(icd9_idx)]:
                    if icd10_code in icd10_types:
                        icd10_idx = icd10_types[icd10_code]
                        # Set ICD-10 code to 1 if any patient has the corresponding ICD-9 code
                        icd10_matrix[:, icd10_idx] = np.maximum(icd10_matrix[:, icd10_idx], synthetic_matrix[:, icd9_idx])
        
        print(f"DEBUG: Step 1 completed - icd10_matrix shape: {icd10_matrix.shape}, non-zero: {np.count_nonzero(icd10_matrix)}")
        
        # convert icd10 to phecodex
        print("Step 2: Converting ICD-10 to PhecodeX...")
        phecodex_matrix = np.zeros((n_patients, len(phecodex_types)), dtype=int)
        
        print(f"DEBUG: Step 2 - phecodex_types={len(phecodex_types)}")
        
        for icd10_idx in tqdm(range(len(icd10_types)), desc="ICD-10 to PhecodeX"):
            if str(icd10_idx) in icd10_to_phecodex:
                for phecodex_code in icd10_to_phecodex[str(icd10_idx)]:
                    if phecodex_code in phecodex_types:
                        phecodex_idx = phecodex_types[phecodex_code]
                        # Set PhecodeX code to 1 if any patient has the corresponding ICD-10 code
                        phecodex_matrix[:, phecodex_idx] = np.maximum(phecodex_matrix[:, phecodex_idx], icd10_matrix[:, icd10_idx])
        
        print(f"DEBUG: Step 2 completed - phecodex_matrix shape: {phecodex_matrix.shape}, non-zero: {np.count_nonzero(phecodex_matrix)}")
    
    # convert phecodex to phecodexm
    print("Step 3: Converting PhecodeX to PhecodeXM...")
    phecodexm_matrix = np.zeros((n_patients, len(phecodexm_types)), dtype=int)
    
    print(f"DEBUG: Step 3 - phecodexm_types={len(phecodexm_types)}")
    print(f"DEBUG: phecodex_to_phecodexm mappings available: {len(phecodex_to_phecodexm)}")
    print(f"DEBUG: phecodex_matrix shape: {phecodex_matrix.shape}, non-zero: {np.count_nonzero(phecodex_matrix)}")
    
    for phecodex_idx in tqdm(range(len(phecodex_types)), desc="PhecodeX to PhecodeXM"):
        if str(phecodex_idx) in phecodex_to_phecodexm:
            phecodexm_indices = phecodex_to_phecodexm[str(phecodex_idx)]
            for phecodexm_idx in phecodexm_indices:
                # Set PhecodeXM code to 1 if any patient has the corresponding PhecodeX code
                phecodexm_matrix[:, phecodexm_idx] = np.maximum(phecodexm_matrix[:, phecodexm_idx], phecodex_matrix[:, phecodex_idx])
    
    print(f"Postprocessed PhecodeXM matrix shape: {phecodexm_matrix.shape}")
    print(f"PhecodeXM matrix mean activation: {phecodexm_matrix.mean():.4f}")
    print(f"PhecodeXM matrix sparsity: {(phecodexm_matrix == 0).mean():.4f}")
    print(f"DEBUG: Final phecodexm_matrix non-zero elements: {np.count_nonzero(phecodexm_matrix)}")
    
    # Save intermediate results
    if 'icd10_matrix' in locals():
        np.save(os.path.join(output_path, "synthetic_icd10_matrix.npy"), icd10_matrix)
    np.save(os.path.join(output_path, "synthetic_phecodex_matrix.npy"), phecodex_matrix)
    np.save(os.path.join(output_path, "synthetic_phecodexm_matrix.npy"), phecodexm_matrix)
    
    # Save as CSV for compatibility with synthEHRella
    phecodexm_df = pd.DataFrame(phecodexm_matrix)
    phecodexm_df.to_csv(os.path.join(output_path, "synthetic_phecodexm.csv"), index=False, header=False)
    
    return phecodexm_matrix


def main():
    parser = argparse.ArgumentParser(description="Train MedGAN for synthetic data generation")
    parser.add_argument("--data_path", type=str, default="./data_files", help="path to MIMIC-III data")
    parser.add_argument("--output_path", type=str, default="./medgan_results", help="Output directory")
    parser.add_argument("--autoencoder_epochs", type=int, default=100, help="Autoencoder pretraining epochs")
    parser.add_argument("--gan_epochs", type=int, default=1000, help="GAN training epochs")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--use_phecode_mapping", action="store_true", help="use proper phecode mapping (ICD-9 to ICD-10 to PhecodeX)")
    parser.add_argument("--save_dir", type=str, default="medgan_results", help="directory to save results")
    parser.add_argument("--postprocess", action="store_true", help="postprocess synthetic data to PhecodeXM format (requires --use_phecode_mapping)")
    parser.add_argument("--gen_freq", type=int, default=2, help="train generator every N discriminator updates (default: 2)")
    args = parser.parse_args()
    
    # setup
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load MIMIC-III data
    print("Loading MIMIC-III data")
    dataset = MIMIC3Dataset(root=args.data_path, tables=["DIAGNOSES_ICD"])
    
    # create phecode dataset
    print("Creating phecode dataset")
    phecode_dataset = PhecodeDataset(
        base_dataset=dataset,
        output_path=args.save_dir,
        use_phecode_mapping=args.use_phecode_mapping
    )
    
    if args.use_phecode_mapping:
        print("Using proper phecode mapping (ICD-9 → ICD-10 → PhecodeX)")
    else:
        print("Using raw ICD codes (no phecode mapping)")
    
    phecode_matrix = phecode_dataset.get_phecode_matrix()
    print(f"Phecode matrix shape: {phecode_matrix.shape}")
    
    # save phecode matrix
    np.save(os.path.join(args.output_path, "phecode_matrix.npy"), phecode_matrix)
    
    # get phecode mapping
    phecode_mapping = phecode_dataset.get_phecode_mapping()
    
    # print phecode_mapping info
    print(f"\nDEBUG: phecode_mapping type: {type(phecode_mapping)}")
    print(f"DEBUG: phecode_mapping keys: {list(phecode_mapping.keys())}")
    if 'phecodexm_types' in phecode_mapping:
        print(f"DEBUG: phecodexm_types count: {len(phecode_mapping['phecodexm_types'])}")
    if 'phecodex_to_phecodexm' in phecode_mapping:
        print(f"DEBUG: phecodex_to_phecodexm count: {len(phecode_mapping['phecodex_to_phecodexm'])}")
    
    # save phecode mapping for debugging
    with open(os.path.join(args.output_path, "phecode_mapping_debug.pkl"), 'wb') as f:
        pickle.dump(phecode_mapping, f)
    print(f"DEBUG: Saved phecode_mapping to {os.path.join(args.output_path, 'phecode_mapping_debug.pkl')}")
    
    # initialize MedGAN model
    print("Initializing MedGAN model...")
    model = MedGAN.from_phecode_matrix(
        phecode_matrix=phecode_matrix,
        latent_dim=args.latent_dim,
        autoencoder_hidden_dim=args.hidden_dim,
        discriminator_hidden_dim=args.hidden_dim,
        minibatch_averaging=True
    )
    
    # device stuff
    model = model.to(device)
    model.autoencoder = model.autoencoder.to(device)
    model.generator = model.generator.to(device)
    model.discriminator = model.discriminator.to(device)
    
    # make a dataloader
    print("Creating dataloader...")
    phecode_matrix_dataset = PhecodeMatrixDataset(phecode_matrix)
    dataloader = DataLoader(
        phecode_matrix_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # autoencoder pretraining
    print("Pretraining autoencoder...")
    autoencoder_losses = model.pretrain_autoencoder(
        dataloader=dataloader,
        epochs=args.autoencoder_epochs,
        lr=args.lr,
        device=device
    )
    
    # train GAN
    print("Training GAN...")
    gan_loss_history = train_medgan(
        model=model,
        dataloader=dataloader,
        n_epochs=args.gan_epochs,
        device=device,
        save_dir=args.save_dir,
        gen_freq=args.gen_freq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        b1=args.b1,
        b2=args.b2
    )
    
    # generate synthetic data
    print("Generating synthetic data...")
    with torch.no_grad():
        synthetic_data = model.generate(1000, device)
        binary_data = model.sample_transform(synthetic_data, threshold=0.5)
    
    synthetic_matrix = binary_data.cpu().numpy()
    
    # save
    print("Saving results...")
    torch.save({
        'model_config': {
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'autoencoder_hidden_dim': args.hidden_dim,
            'discriminator_hidden_dim': args.hidden_dim,
            'input_dim': phecode_matrix.shape[1],
        },
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'autoencoder_state_dict': model.autoencoder.state_dict(),
    }, os.path.join(args.output_path, "medgan_final.pth"))
    
    np.save(os.path.join(args.output_path, "synthetic_binary_matrix.npy"), synthetic_matrix)
    
    # save loss histories
    loss_history = {
        'autoencoder_losses': autoencoder_losses,
        'gan_losses': gan_loss_history,
    }
    np.save(os.path.join(args.output_path, "loss_history.npy"), loss_history)
    
    # save phecode mapping
    with open(os.path.join(args.output_path, "phecode_mapping.pkl"), 'wb') as f:
        pickle.dump(phecode_mapping, f)
    
    # print final stats
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Real data shape: {phecode_matrix.shape}")
    print(f"Real data mean activation: {phecode_matrix.mean():.4f}")
    print(f"Real data sparsity: {(phecode_matrix == 0).mean():.4f}")
    print(f"Synthetic data shape: {synthetic_matrix.shape}")
    print(f"Synthetic data mean activation: {synthetic_matrix.mean():.4f}")
    print(f"Synthetic data sparsity: {(synthetic_matrix == 0).mean():.4f}")
    print(f"Results saved to: {args.output_path}")
    print("="*50)

    # postprocessing: synthetic data to phecodexm format (only if using proper phecode mapping)
    if args.postprocess:
        if not args.use_phecode_mapping:
            print("\nError: --postprocess requires --use_phecode_mapping to be specified.")
            print("Please run with both flags: --use_phecode_mapping --postprocess")
        else:
            print("\n" + "="*50)
            print("POSTPROCESSING SYNTHETIC DATA")
            print("="*50)
            postprocess_synthetic_data(synthetic_matrix, phecode_mapping, args.output_path)
            print("Postprocessing completed! Check the following files:")
            print("- synthetic_phecodexm_matrix.npy: Final postprocessed synthetic data (594 codes)")
            print("- synthetic_phecodexm.csv: CSV format for compatibility with synthEHRella")
            print("- synthetic_icd10_matrix.npy: Intermediate ICD-10 conversion")
            print("- synthetic_phecodex_matrix.npy: Intermediate PhecodeX conversion")
    elif args.use_phecode_mapping:
        print("\nNote: Synthetic data generated with phecode mapping but not postprocessed.")
        print("To get postprocessed data with 594 PhecodeXM codes, run with --postprocess flag.")
    else:
        print("\nNote: Using raw ICD codes. For postprocessed data, run with --use_phecode_mapping --postprocess flags.")


if __name__ == "__main__":
    main() 


"""
Slurm script example:

#!/bin/bash
#SBATCH --account=jalenj4-ic
#SBATCH --job-name=medgan_pyhealth
#SBATCH --output=logs/medgan_pyhealth_%j.out
#SBATCH --error=logs/medgan_pyhealth_%j.err
#SBATCH --partition=IllinoisComputes-GPU              # Change to appropriate partition
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Change to the directory where you submitted the job
cd "$SLURM_SUBMIT_DIR"

# Print useful Slurm environment variables for debugging
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Optional: check what GPU(s) is/are actually visible
echo "Running nvidia-smi to confirm GPU availability:"
nvidia-smi

# Load modules or activate environment
# module load python/3.10
# module load cuda/11.7
# conda activate pyhealth

# Create output directories
mkdir -p logs
mkdir -p medgan_results

# Set parameters (matching original synthEHRella defaults)
export AUTOENCODER_EPOCHS=100
export GAN_EPOCHS=1000
export BATCH_SIZE=128
export LATENT_DIM=128
export HIDDEN_DIM=128
export NUM_SAMPLES=1000
export LEARNING_RATE=0.001
export WEIGHT_DECAY=0.0001
export BETA1=0.5
export BETA2=0.9

echo "Starting PyHealth MedGAN training with parameters:"
echo "  Autoencoder epochs: $AUTOENCODER_EPOCHS"
echo "  GAN epochs: $GAN_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Latent dimension: $LATENT_DIM"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Number of synthetic samples: $NUM_SAMPLES"
echo "  Learning rate: $LEARNING_RATE"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Beta1: $BETA1"
echo "  Beta2: $BETA2"

# Run the comprehensive PyHealth MedGAN script
python examples/synthetic_data_generation_mimic3_medgan.py \
    --data_path ./data_files \
    --output_path ./medgan_results \
    --autoencoder_epochs $AUTOENCODER_EPOCHS \
    --gan_epochs $GAN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --b1 $BETA1 \
    --b2 $BETA2 \
    --use_phecode_mapping \
    --postprocess

echo "PyHealth MedGAN training completed!"
echo "Results saved to: ./medgan_results/"
echo "Check the following files:"
echo "  - synthetic_binary_matrix.npy: Raw synthetic data"
echo "  - synthetic_phecodexm_matrix.npy: Postprocessed PhecodeXM data (594 codes)"
echo "  - synthetic_phecodexm.csv: CSV format for compatibility"
echo "  - medgan_final.pth: Trained model"
echo "  - loss_history.npy: Training loss history" 
"""