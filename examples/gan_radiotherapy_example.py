# %% [markdown]
# # Automated Treatment Planning using GANs (Replication Attempt)
# 
# This notebook attempts to replicate the logic from Mahmood et al. (2018), "Automated Treatment Planning in Radiation Therapy using Generative Adversarial Networks," using the CORT "LIVER" dataset.
# 
# **Paper Reference:** https://github.com/rafidrm/gancer \
# **CORT Dataset:** http://gigadb.org/dataset/100110

# %%
import os
import glob
import random

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from gan_radiotherapy import load_ct_series, parse_ctvoxel_info, get_structure_masks_from_voilists, CORTGANData, GeneratorUNet, Discriminator, weights_init_normal

# %% [markdown]
# ## Config & Hyperparams

# %%
# User provided Kaggle path
kaggle_path = '/kaggle/input/imrt-optimization-research-the-cort-dataset/'

CONFIG = {
    # Paths
    "kaggle_base_path": kaggle_path,
    "case_name": "LIVER",  # For this example, we're only using the LIVER data
    "data_dir_base": os.path.join(kaggle_path, "dataset"),  # Base for CORT data
    
    # Model Params (from research paper)
    "image_size": 128,
    "input_ct_channels": 1,
    "output_dose_channels": 1,
    "batch_size": 4,
    "lr_g": 0.0002,
    "lr_d": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "lambda_l1": 90,
    "num_epochs": 25,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "display_interval": 20,  # (num_slices / batch_size, currently every 5)
    "random_seed": 42,
    
    # Data Processing params
    # Since the paper used its own dataset, we need to adapt the data processing
    # One process is normalizing our data to [-1, 1] range
    "ct_window_width": 400, # Hounsfield unit (HU) for normalization, used for CT scans
    "ct_window_level": 40,  # HU
    "structure_pixel_values": { # Values for input CT
        "PTV": 2000,
        "OAR": 1500,
    },
    "synthetic_dose_values": { # Values for target dose map [0,1] (prior to [-1,1] map)
        "PTV": 1.0,
        "OAR": 0.1,
        "OTHER": 0.0,
    },
    
    # (Liver) File path
    "ct_dicom_dir": os.path.join(kaggle_path, "dataset/Liver_dicom"),
}

# .mat files dir
CONFIG['mat_files_dir'] = os.path.join(CONFIG['data_dir_base'], CONFIG['case_name']) # kaggle_path/dataset/LIVER/
# CTVOXEL_INFO.txt path
CONFIG['ctvoxel_info_file'] = os.path.join(kaggle_path, f"dataset/{CONFIG['case_name']}/CTVOXEL_INFO_{CONFIG['case_name']}.txt")

print(f"Using CT DICOM directory: {CONFIG['ct_dicom_dir']}")
print(f"Using .mat files directory: {CONFIG['mat_files_dir']}")
print(f"Using CTVOXEL_INFO file: {CONFIG['ctvoxel_info_file']}")
print(f"Using device: {CONFIG['device']}")

# %%
# Set seed
random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["random_seed"])

# %% [markdown]
# ### Execute on Loading Data

# %%
# Load LIVER data
print(f"Loading data for case: {CONFIG['case_name']}")
if not os.path.isdir(CONFIG['ct_dicom_dir']):
    raise FileNotFoundError(f"CT DICOM directory not found: {CONFIG['ct_dicom_dir']}. Please check Kaggle path and dataset structure.")

ct_volume_hu, ct_spacing_info = load_ct_series(CONFIG['ct_dicom_dir'])
ctvoxel_info = parse_ctvoxel_info(CONFIG['ctvoxel_info_file'])

if ct_volume_hu is None:
    raise ValueError("Failed to load CT data. Check DICOM paths and files.")

print(f"CT volume shape: {ct_volume_hu.shape if ct_volume_hu is not None else 'N/A'}")
if ctvoxel_info:
    print(f"CTVOXEL_INFO sample: voxels x={ctvoxel_info.get('number of voxels [cerr x] right -> left', 'N/A')}")
else:
    print("CTVOXEL_INFO was not loaded (or failed to parse). This may impact mask generation if dimensions differ from CT.")

# %%
# Define structure names for LIVER data
ptv_structure_names = ["PTV"]
oar_structure_names = [
    "Liver",
    "SpinalCord",
    "KidneyR",
    "KidneyL",
    "Stomach"
]

print(f"\n--- Checking for .mat files in: {CONFIG['mat_files_dir']} ---")
found_mat_files = (glob.glob(os.path.join(CONFIG['mat_files_dir'], '*_VOILIST.mat')) + 
                   glob.glob(os.path.join(CONFIG['mat_files_dir'], '*_VOILISTS.mat'))
                   )
if not found_mat_files:
    print(f"WARNING: No '*_VOILIST(S).mat' files found in {CONFIG['mat_files_dir']}. Masks will be empty.")
else:
    print("Found the following .mat structure files:")
    for f_name in found_mat_files:
        print(f"  {os.path.basename(f_name)}")

# Load structure masks
all_s_names = list(set(ptv_structure_names + oar_structure_names))
structure_masks_3d = get_structure_masks_from_voilists(CONFIG['mat_files_dir'], all_s_names, ct_volume_hu.shape, ctvoxel_info)

# %%
# Create PTV mask by combining all PTV structures
combined_ptv_mask = np.zeros_like(ct_volume_hu, dtype=bool)
for name in ptv_structure_names:
    if name in structure_masks_3d and structure_masks_3d[name] is not None:
        if structure_masks_3d[name].shape == combined_ptv_mask.shape:
            combined_ptv_mask = np.logical_or(combined_ptv_mask, structure_masks_3d[name])
        else:
            print(f"Shape mismatch for PTV mask {name}: {structure_masks_3d[name].shape} vs CT {combined_ptv_mask.shape}. Skipping this PTV component.")

# Create OAR mask by combining all OAR structures
combined_oar_mask = np.zeros_like(ct_volume_hu, dtype=bool)
for name in oar_structure_names:
    if name in structure_masks_3d and structure_masks_3d[name] is not None:
        if structure_masks_3d[name].shape == combined_oar_mask.shape:
            non_overlapping_oar_mask = np.logical_and(structure_masks_3d[name], np.logical_not(combined_ptv_mask))
            combined_oar_mask = np.logical_or(combined_oar_mask, non_overlapping_oar_mask)

# %%
# Create Dataset and DataLoader
dataset = CORTGANData(ct_volume_hu, combined_ptv_mask, combined_oar_mask, CONFIG)
dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

# %%
# Visualize a sample from the dataloader
if len(dataloader) > 0:
    print("\n--- Fetching first sample from DataLoader (will trigger __getitem__ diagnostics) ---")
    sample_input_ct, sample_target_dose = next(iter(dataloader))
    sample_input_ct_item = sample_input_ct[0]
    sample_target_dose_item = sample_target_dose[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample_input_ct_item.squeeze().cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title("Sample Composite Input CT (from DataLoader)")
    axes[0].axis('off')
    axes[1].imshow(sample_target_dose_item.squeeze().cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title("Sample Synthetic Target Dose (from DataLoader)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Init Models, Losses, Optimizers

# %%
# Init models
generator = GeneratorUNet(
    in_channels=CONFIG["input_ct_channels"],
    out_channels=CONFIG["output_dose_channels"]
).to(CONFIG["device"])
discriminator = Discriminator(
    in_channels=CONFIG["output_dose_channels"]
).to(CONFIG["device"])

# Init weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss funcs + optimizers
criterion_GAN = nn.BCEWithLogitsLoss().to(CONFIG["device"]) # For PatchGAN output
criterion_L1 = nn.L1Loss().to(CONFIG["device"])
optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG["lr_g"], betas=(CONFIG["beta1"], CONFIG["beta2"]))
optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG["lr_d"], betas=(CONFIG["beta1"], CONFIG["beta2"]))

# %%
# Sanity check model outputs + calc D_patch_shape
if len(dataset) > 0:
    # Get next dataload sample
    sample_batch_input_ct, _ = next(iter(dataloader))
    test_input_ct = sample_batch_input_ct[0].unsqueeze(0).to(CONFIG["device"]) # Take first from batch, add batch dim

    # Verify input shape
    with torch.no_grad():
        test_fake_dose = generator(test_input_ct)
        test_disc_out = discriminator(test_fake_dose)
    print(f"Generator output shape: {test_fake_dose.shape}")
    print(f"Discriminator output shape: {test_disc_out.shape}")
    
    CONFIG['D_patch_shape'] = test_disc_out.shape[2:] # Store patch shape for labels
else:
    raise Exception("Dataset is empty, cannot perform model sanity check with data.")

# %% [markdown]
# ### Execution

# %%
# Execute model train loop

from gan_radiotherapy import train_gan

fixed_input_batch, fixed_target_batch = train_gan(generator, discriminator, dataloader, CONFIG, criterion_GAN, criterion_L1, optimizer_G, optimizer_D)


# %% [markdown]
# ## Evaluation & Visualization

# %%
# Load fixed batch for eval
generator.eval()
eval_input_ct = fixed_input_batch.to(CONFIG["device"])
eval_target_dose = fixed_target_batch.to(CONFIG["device"])

with torch.no_grad():
    eval_generated_dose = generator(eval_input_ct)

# Convert tensors to numpy arrays for visualization
eval_input_ct_np = eval_input_ct.cpu().numpy()
eval_target_dose_np = eval_target_dose.cpu().numpy()
eval_generated_dose_np = eval_generated_dose.cpu().numpy()

# Rescale to [0, 1] for viz
num_samples_to_show = min(4, eval_input_ct_np.shape[0])
fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(15, num_samples_to_show * 5))
if num_samples_to_show == 1: axes = [axes]

for i in range(num_samples_to_show):
    # For each sample, plot the input CT, target dose, and generated dose
    axes[i][0].imshow(eval_input_ct_np[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
    axes[i][0].set_title("Input Composite CT"); axes[i][0].axis('off')
    axes[i][1].imshow(eval_target_dose_np[i].squeeze(), cmap='viridis', vmin=-1, vmax=1)
    axes[i][1].set_title("Synthetic Target Dose"); axes[i][1].axis('off')
    axes[i][2].imshow(eval_generated_dose_np[i].squeeze(), cmap='viridis', vmin=-1, vmax=1)
    axes[i][2].set_title("Generated Dose"); axes[i][2].axis('off')

    target_01 = (eval_target_dose_np[i].squeeze() + 1) / 2
    generated_01 = (eval_generated_dose_np[i].squeeze() + 1) / 2
    mae = np.mean(np.abs(target_01 - generated_01)) * 2 # MAE on [-1,1] scale by re-scaling difference
    mse = np.mean((target_01 - generated_01) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf') # Assumes max_pixel_val is 1 for [0,1] range
    axes[i][2].text(5, 15, f"MAE: {mae:.3f}\nPSNR: {psnr:.2f}dB", color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

plt.suptitle("Evaluation Results (Post-Training) - LIVER case", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("evaluation_results_liver.png"); plt.show()

# %% [markdown]
# ## Saving Model

# %%
os.makedirs("saved_models", exist_ok=True)
model_save_name_suffix = f"{CONFIG['case_name']}_epoch{CONFIG['num_epochs']}.pth"

torch.save(generator.state_dict(), f"saved_models/generator_{model_save_name_suffix}")
torch.save(discriminator.state_dict(), f"saved_models/discriminator_{model_save_name_suffix}")

print(f"Models saved with suffix: {model_save_name_suffix}")


