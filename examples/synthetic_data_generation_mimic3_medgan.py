"""
Synthetic data generation using MedGAN on MIMIC-III data.

This example demonstrates how to train MedGAN to generate synthetic ICD-9 matrices
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
from pyhealth.datasets.icd9_matrix import create_icd9_matrix, ICD9MatrixDataset
from pyhealth.models.generators.medgan import MedGAN

"""
python examples/synthetic_data_generation_mimic3_medgan.py --autoencoder_epochs 5 --gan_epochs 10 --batch_size 16
"""
def train_medgan(model, dataloader, n_epochs, device, save_dir, lr=0.001, weight_decay=0.0001, b1=0.5, b2=0.9):
    """
    Train MedGAN model using the original synthEHRella approach.
    
    Args:
        model: MedGAN model
        dataloader: DataLoader for training data
        n_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
        lr: Learning rate
        weight_decay: Weight decay for regularization
        b1: Beta1 for Adam optimizer
        b2: Beta2 for Adam optimizer
    
    Returns:
        loss_history: Dictionary containing loss history
    """

    def generator_loss(y_fake):
        """
        Original synthEHRella generator loss
        """
        # standard GAN generator loss - want fake samples to be classified as real
        return -torch.mean(torch.log(y_fake + 1e-12))
    
    def discriminator_loss(outputs, labels):
        """
        Original synthEHRella discriminator loss
        """
        loss = -torch.mean(labels * torch.log(outputs + 1e-12)) - torch.mean((1 - labels) * torch.log(1. - outputs + 1e-12))
        return loss

    optimizer_g = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.autoencoder.decoder.parameters(), 'lr': lr * 0.1}
    ], lr=lr, betas=(b1, b2), weight_decay=weight_decay)
    
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), 
                                  lr=lr * 0.1, betas=(b1, b2), weight_decay=weight_decay)
    
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
            
            valid = torch.ones(batch_size).to(device)  # 1D tensor
            fake = torch.zeros(batch_size).to(device)  # 1D tensor
            
            z = torch.randn(batch_size, model.latent_dim).to(device)
            
            # Disable discriminator gradients for generator training to prevent discriminator from being updated
            for p in model.discriminator.parameters():
                p.requires_grad = False
            
            # generate fake samples
            fake_samples = model.generator(z)
            fake_samples = model.autoencoder.decode(fake_samples)
            
            # generator loss using original medgan loss function
            fake_output = model.discriminator(fake_samples).view(-1)
            g_loss = generator_loss(fake_output)
            
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
        
        # calculate average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        # store losses for trackin
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        progress = (epoch + 1) / n_epochs * 100
        print(f"{epoch+1:5d} | {avg_d_loss:.4f} | {avg_g_loss:.4f} | {progress:5.1f}%")
        
        # save every 50 epochs
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
    
    # save loss history
    loss_history = {
        'g_losses': g_losses,
        'd_losses': d_losses,
    }
    np.save(os.path.join(save_dir, "loss_history.npy"), loss_history)
    
    return loss_history




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
    parser.add_argument("--save_dir", type=str, default="medgan_results", help="directory to save results")
    args = parser.parse_args()
    
    # setup
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load MIMIC-III data
    print("Loading MIMIC-III data")
    dataset = MIMIC3Dataset(root=args.data_path, tables=["DIAGNOSES_ICD"])
    
    # create ICD-9 matrix using utility function
    print("Creating ICD-9 matrix")
    icd9_matrix, icd9_types = create_icd9_matrix(dataset, args.output_path)
    print(f"ICD-9 matrix shape: {icd9_matrix.shape}")
    
    
    # initialize MedGAN model
    print("Initializing MedGAN model...")
    model = MedGAN.from_binary_matrix(
        binary_matrix=icd9_matrix,
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
    icd9_matrix_dataset = ICD9MatrixDataset(icd9_matrix)
    dataloader = DataLoader(
        icd9_matrix_dataset, 
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
            'input_dim': icd9_matrix.shape[1],
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
    
    # print final stats
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Real data shape: {icd9_matrix.shape}")
    print(f"Real data mean activation: {icd9_matrix.mean():.4f}")
    print(f"Real data sparsity: {(icd9_matrix == 0).mean():.4f}")
    print(f"Synthetic data shape: {synthetic_matrix.shape}")
    print(f"Synthetic data mean activation: {synthetic_matrix.mean():.4f}")
    print(f"Synthetic data sparsity: {(synthetic_matrix == 0).mean():.4f}")
    print(f"Results saved to: {args.output_path}")
    print("="*50)

    print("\nGenerated synthetic data in original MIMIC3 ICD-9 format.")


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
source pyhealth/bin/activate
export PYTHONPATH=/u/jalenj4/PyHealth/PyHealth:$PYTHONPATH

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

echo "PyHealth MedGAN training completed!"
echo "Results saved to: ./medgan_results/"
echo "Check the following files:"
echo "  - synthetic_binary_matrix.npy: Synthetic data in original MIMIC3 ICD-9 format"
echo "  - medgan_final.pth: Trained model"
echo "  - loss_history.npy: Training loss history" 
"""