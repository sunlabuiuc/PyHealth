# %% [markdown]
# # Automated Treatment Planning using GANs (Replication Attempt)
# 
# This notebook attempts to replicate the logic from Mahmood et al. (2018), "Automated Treatment Planning in Radiation Therapy using Generative Adversarial Networks," using the CORT "LIVER" dataset.
# 
# **Paper Reference:** https://github.com/rafidrm/gancer \
# **CORT Dataset:** http://gigadb.org/dataset/100110

# %%
import os
from tqdm.notebook import tqdm

import numpy as np
import scipy.io
import pydicom 

import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# %% [markdown]
# ## Data Ingestion and Pre-Processing Utils

# %%
def load_ct_series(dicom_dir: str):
    """
    Loads a CT series from a directory of DICOM files, ensuring they are of type CT + sorted.
    """
    if not os.path.isdir(dicom_dir):
        raise Exception(f"CT DICOM directory not found: {dicom_dir}")
    
    # Get all files that are DICOMs
    potential_files = []
    for root, _, files in os.walk(dicom_dir):
        for fn in files:
            # .dcm or no extension == potential DICOM
            if fn.lower().endswith(".dcm") or '.' not in fn:
                potential_files.append(os.path.join(root, fn))

    # Edge Case: no data found!
    if not potential_files:
        raise Exception(f"No potential DICOM files (ending in .dcm or no extension) found in {dicom_dir} or its subdirectories.")

    ct_slices_data = [] # Store tuples of (sort_key, pixel_array, dicom_header)
    print(f"Found {len(potential_files)} potential DICOM files. Reading and filtering for CT modality.")
    for f_path in tqdm(potential_files, desc="Reading DICOM headers"):
        try:
            ds = pydicom.dcmread(f_path)
            # Check if a CT image
            if hasattr(ds, 'Modality') and ds.Modality == 'CT':
                # Try to get a sort key -> fallback to InstanceNumber
                sort_key = None
                if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
                    sort_key = float(ds.ImagePositionPatient[2])
                elif hasattr(ds, 'InstanceNumber'):
                    sort_key = int(ds.InstanceNumber)

                if sort_key is not None:
                    ct_slices_data.append({'sort_key': sort_key, 'dataset': ds})
                else:
                    print(f"Warning: CT slice {f_path} is missing common sorting attributes (ImagePositionPatient[2], InstanceNumber). Skipping.")

        except Exception as e:
            pass  # Not DICOM

    # Edge Case: no valid CT slices found
    if not ct_slices_data:
        raise Exception("No valid CT slices with sort keys could be read from the directory.")
        
    # Use sort_key to organize collected CT slices
    try:
        ct_slices_data.sort(key=lambda x: x['sort_key'])
    except Exception as e:
        raise Exception(f"Error during final sorting of collected CT slices: {e}. Order may be incorrect or data unusable.")

    # Normalize data to have same slope/intercept among all slices (using HU conversion - RescaleSlope and RescaleIntercept)
    sorted_dicom_objects = [item['dataset'] for item in ct_slices_data]
    first_slice_ds = sorted_dicom_objects[0]
    slope = getattr(first_slice_ds, 'RescaleSlope', 1.0) # Default to 1.0
    intercept = getattr(first_slice_ds, 'RescaleIntercept', 0.0) # Default to 0.0
    try:
        # Stack, convert to float32, apply slope/intercept, and convert back to int16
        image_stack = np.stack([s.pixel_array for s in sorted_dicom_objects])
        image_stack = image_stack.astype(np.float32) * float(slope) + float(intercept)
        image_stack = image_stack.astype(np.int16)
    except Exception as e:
        raise Exception(f"Error stacking pixel arrays or applying HU conversion: {e}")
    
    # Check pixel spacing and slice thickness
    pixel_spacing = getattr(first_slice_ds, 'PixelSpacing', [1.0, 1.0])
    try: # SliceThickness can sometimes be missing or not a number
        slice_thickness = float(getattr(first_slice_ds, 'SliceThickness', 1.0))
    except ValueError:
        slice_thickness = 1.0
        print(f"Warning: Could not parse SliceThickness ('{first_slice_ds.SliceThickness}'). Defaulting to 1.0.")

    print(f"Successfully loaded and sorted {len(sorted_dicom_objects)} CT slices.")
    return image_stack, (pixel_spacing, slice_thickness)

# %%
def parse_ctvoxel_info(filepath: str):
    """
    Parses the CTVOXEL_INFO.txt file.
    
    Converts values to left of "=" to key, value pair of str:float or str:list.
    """
    # Edge Case: file not found
    if not os.path.exists(filepath):
        raise Exception(f"CTVOXEL_INFO.txt not found at {filepath}")

    info = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                try:
                    if ',' in value:
                        info[key] = [float(v.strip()) for v in value.strip().replace('[','').replace(']','').split(',')]
                    else:
                        info[key] = float(value.strip())
                except ValueError:
                    info[key] = value.strip()
    return info

# %%
def get_structure_masks_from_voilists(mat_files_dir, structure_names, ct_shape_zyx, ctvoxel_info):
    """
    Creates 3D masks for specified structures from _VOILIST.mat files.
    Param ct_shape_zyx -> (num_slices_z, height_y, width_x)
    """
    masks = {}
    
    if ctvoxel_info is None:
        # Default vals w/o ctvoxel_info come from ct_shape_zyx
        print("Error: CTVOXEL_INFO is None. Cannot determine mask dimensions from it.")
        num_voxels_z, num_voxels_y, num_voxels_x = ct_shape_zyx
        print(f"Warning: Using CT shape {ct_shape_zyx} for mask dimensions due to missing CTVOXEL_INFO.")
    else:
        # Extract dimensions from CTVOXEL_INFO
        num_voxels_x = int(ctvoxel_info.get('number of voxels [cerr x] right -> left', ct_shape_zyx[2]))
        num_voxels_y = int(ctvoxel_info.get('number of voxels [cerr gui y] patient post[back side] -> ant[front side]', ct_shape_zyx[1]))
        num_voxels_z = int(ctvoxel_info.get('number of voxels in [cerr gui z] patient head -> feet', ct_shape_zyx[0]))
    
    expected_ct_shape_from_info_zyx = (num_voxels_z, num_voxels_y, num_voxels_x)
    
    if ct_shape_zyx != expected_ct_shape_from_info_zyx and ctvoxel_info is not None:
        print(f"Warning: Loaded CT shape {ct_shape_zyx} (z,y,x) differs from CTVOXEL_INFO-derived shape {expected_ct_shape_from_info_zyx} (z,y,x).")
        mask_dims_zyx = expected_ct_shape_from_info_zyx
    else:
        mask_dims_zyx = ct_shape_zyx

    for name in structure_names:
        mat_filepath = os.path.join(mat_files_dir, f"{name}_VOILIST.mat")
        
        if mat_filepath:
            try:
                # Extract voxel indices from .mat file
                data = scipy.io.loadmat(mat_filepath)
                voilist_key = [k for k in data.keys() if not k.startswith('__')][0]
                voxel_indices_flat = data[voilist_key].flatten().astype(int) - 1
                current_mask_3d = np.zeros(mask_dims_zyx, dtype=bool)
                
                # Convert flat Fortran-style (x,y,z) indices to 3D Python (z,y,x) indices
                # Rules:
                # - flat_idx = x + y*Nx + z*Nx*Ny (0-based CERR/Fortran-style flattening)
                # - x varies fastest, then y, then z.
                # - Nz_m, Ny_m, Nx_m are dimensions of the mask grid
                Nz_m, Ny_m, Nx_m = mask_dims_zyx 
                
                # Convert flat indices to 3D coordinates
                x_coords = voxel_indices_flat % Nx_m
                temp = voxel_indices_flat // Nx_m
                y_coords = temp % Ny_m
                z_coords = temp // Ny_m
                # Sanity Check: bound check against mask_dims_zyx
                valid_indices = (x_coords < Nx_m) & (y_coords < Ny_m) & (z_coords < Nz_m) & \
                                (x_coords >= 0) & (y_coords >= 0) & (z_coords >= 0)
                # Filter valid indices
                filtered_x = x_coords[valid_indices]
                filtered_y = y_coords[valid_indices]
                filtered_z = z_coords[valid_indices]
                # Create mask
                current_mask_3d[filtered_z, filtered_y, filtered_x] = True
                
                # If mask_dims_zyc (from CTVOXEL_INFO) is different from ct_shape_zyx, we need to volumetrically resample to match CT geometry for overlay.
                # Here, we use nearest-neighbor style resampling since it's very simple (order = 0 is the nearest neighbor).
                if current_mask_3d.shape != ct_shape_zyx:
                    print(f"Resampling mask for {name} from {current_mask_3d.shape} to CT shape {ct_shape_zyx}")
                    resampled_mask_numeric = np.zeros(ct_shape_zyx, dtype=np.float32)  # Create shape
                    for z_orig in range(current_mask_3d.shape[0]):
                        slice_resampled = sk_resize(current_mask_3d[z_orig,:,:].astype(np.float32),
                                                    (ct_shape_zyx[1], ct_shape_zyx[2]),
                                                    order=0, 
                                                    preserve_range=True, 
                                                    anti_aliasing=False
                                                    )
                        # Apply z-scaling given adjustments
                        z_new = int(z_orig * ct_shape_zyx[0] / current_mask_3d.shape[0])
                        if z_new < ct_shape_zyx[0]:
                            resampled_mask_numeric[z_new,:,:] = np.maximum(resampled_mask_numeric[z_new,:,:], slice_resampled)
                    current_mask_3d = resampled_mask_numeric.astype(bool)


                masks[name] = current_mask_3d
            except Exception as e:
                print(f"Error loading or processing {mat_filepath} for structure '{name}': {e}")
                masks[name] = np.zeros(ct_shape_zyx, dtype=bool) # If error, empty mask of same shape
        else:
            print(f"Structure .mat file not found for: {name} (Searched for {name}_VOILIST.mat)")
            masks[name] = np.zeros(ct_shape_zyx, dtype=bool)  # Same if file not found
    return masks

# %%
def normalize_ct_window(ct_slice_hu, width, level):
    """
    Window CT slice and normalize to [-1, 1].
    """
    lower = level - width / 2
    upper = level + width / 2
    ct_slice = np.clip(ct_slice_hu.astype(np.float32), lower, upper)
    ct_slice = (ct_slice - lower) / (upper - lower) # Normalize to [0, 1]
    ct_slice = (ct_slice * 2) - 1 # Then, to [-1, 1]
    return ct_slice.astype(np.float32)

# %%
def create_composite_input(raw_hu_ct_slice, ptv_mask_slice, oar_mask_slice, config):
    """
    Creates composite input image normalized to [-1, 1].
    The strat is to window the CT slice first, then embed structure information using distinct values within the normalized [-1,1] range.
    """
    # Normalize the base CT slice first
    norm_ct_slice = normalize_ct_window(raw_hu_ct_slice, config["ct_window_width"], config["ct_window_level"])
    composite = np.copy(norm_ct_slice)
    
    # For visualization:
    # We select normalized values for OARs and PTVs that are distinct within the [-1, 1] range.
    # For example, OARs can be set to 0.0 (mid-gray) and PTVs to 1.0 (white).
    # This allows us to visualize the structures clearly against the CT slice.
    
    oar_display_val = 0.0
    ptv_display_val = 1.0
    
    if oar_mask_slice is not None and oar_mask_slice.any():
        composite[oar_mask_slice] = oar_display_val 
    if ptv_mask_slice is not None and ptv_mask_slice.any():
        composite[ptv_mask_slice] = ptv_display_val # NOTE: PTV overrides OAR in composite if overlap

    return composite.astype(np.float32)

# %%
def create_synthetic_dose_map(ptv_mask_slice, oar_mask_slice, shape, config):
    """
    Creates a synthetic dose map
    PTV=high, OAR=low, Other=lowest
    """
    dose_map_01 = np.full(shape, config["synthetic_dose_values"]["OTHER"], dtype=np.float32)
    
    # Set PTV and OAR values
    if oar_mask_slice is not None and oar_mask_slice.any():
        dose_map_01[oar_mask_slice] = config["synthetic_dose_values"]["OAR"]
    if ptv_mask_slice is not None and ptv_mask_slice.any():
        dose_map_01[ptv_mask_slice] = config["synthetic_dose_values"]["PTV"]
    
    # Convert from [0,1] (based on config vals) to [-1,1] for proper domain output
    dose_map_neg1_1 = (dose_map_01 * 2) - 1
    return dose_map_neg1_1.astype(np.float32)

# %% [markdown]
# ## PyTorch Dataset and DataLoader

# %% [markdown]
# ### Dataset Class

# %%
class CORTGANData(Dataset):
    def __init__(self, ct_volume_raw_hu, ptv_mask_3d, oar_mask_3d, config):
        """
        Dataset ingestion class for CORTGAN data.

        Args:
            ct_volume_raw_hu: Raw CT volume in Hounsfield units (HU)
            ptv_mask_3d: 3D mask for the PTV (Planning Target Volume)
            oar_mask_3d: 3D mask for the OAR (Organs at Risk)
            config: Config dict with params for data processing
        """
        self.ct_volume_raw_hu = ct_volume_raw_hu
        self.ptv_mask_3d = ptv_mask_3d
        self.oar_mask_3d = oar_mask_3d
        self.config = config
        self.num_slices = ct_volume_raw_hu.shape[0]

        # Sanity check: num slices for ptv_mask_3d == num slices for ct_volume_raw_hu
        self.slices_with_ptv = []
        if self.ptv_mask_3d is not None and self.ptv_mask_3d.shape[0] == self.num_slices :
            self.slices_with_ptv = [i for i in range(self.num_slices) if self.ptv_mask_3d[i].any()]
        
        # Make sure we got (PTV slice) data to work with
        if not self.slices_with_ptv:
            raise Exception('''
                            IMPORTANT WARNING: No slices with PTV content found in `ptv_mask_3d`. 
                            Thus, `ptv_mask_3d` is likely all zeros or not correctly aligned/loaded. 
                            The "Synthetic Target Dose" will also be blank (and the DataLoader will effectively be non-functional). :D
                            ''')
        else:
            self.relevant_slices_indices = self.slices_with_ptv
        
        print(f"Dataset init - Total CT slices: {self.num_slices}, using {len(self.relevant_slices_indices)} slices.")
        print(f"   Total PTV voxels in 3D mask: {self.ptv_mask_3d.sum()}")
        print(f"   Total OAR voxels in 3D mask: {self.oar_mask_3d.sum()}")

    def __len__(self):
        return len(self.relevant_slices_indices)

    def __getitem__(self, idx):
        # Get the slice index from the relevant slices
        slice_idx = self.relevant_slices_indices[idx]
        raw_hu_ct_slice = self.ct_volume_raw_hu[slice_idx, :, :]
        
        # Verify masks are not None type before slice
        ptv_slice_mask = np.zeros_like(raw_hu_ct_slice, dtype=bool)
        if self.ptv_mask_3d is not None and slice_idx < self.ptv_mask_3d.shape[0]:
            ptv_slice_mask = self.ptv_mask_3d[slice_idx, :, :]
        oar_slice_mask = np.zeros_like(raw_hu_ct_slice, dtype=bool)
        if self.oar_mask_3d is not None and slice_idx < self.oar_mask_3d.shape[0]:
            oar_slice_mask = self.oar_mask_3d[slice_idx, :, :]

        # Create composite input + target dose map
        composite_input_np = create_composite_input(raw_hu_ct_slice, ptv_slice_mask, oar_slice_mask, self.config)
        target_dose_np = create_synthetic_dose_map(ptv_slice_mask, oar_slice_mask, raw_hu_ct_slice.shape, self.config)

        # Resize to target shape (if necessary)
        target_shape = (self.config["image_size"], self.config["image_size"])
        if composite_input_np.shape != target_shape:
            composite_input_np = sk_resize(composite_input_np, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
        if target_dose_np.shape != target_shape:
            target_dose_np = sk_resize(target_dose_np, target_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)

        # Expand dims to match expected input shape
        composite_input_np = np.expand_dims(composite_input_np, axis=0)
        target_dose_np = np.expand_dims(target_dose_np, axis=0)

        return torch.from_numpy(composite_input_np), torch.from_numpy(target_dose_np)

# %% [markdown]
# ## Model Architecture (Pix2Pix Style)

# %% [markdown]
# ### Generator (U-Net)

# %%
# Model architecture comes from pix2pix paper
# https://arxiv.org/pdf/1611.07004.pdf
# The U-Net architecture is an extremely popular choice for "image-to-image"-type translation tasks.

# Breakdown:
# It consists of an encoder-decoder structure with skip connections.
# The encoder downsamples the input image, while the decoder upsamples it back to the original size.
# The skip connections allow the model to retain spatial information lost during downsampling.

class UNetDown(nn.Module):
    """
    UNetDown block for the U-Net architecture.
    The encoder part of the U-Net, which downsamples the input image.
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True)) # affine=True is default, learns scale/shift
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """
    UNetUp block for the U-Net architecture.
    The decoder part of the U-Net, which upsamples the input image.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        # For pix2pix U-Net with kernel=4, stride=2, padding=1, dimensions should halve/double correctly for powers of 2.
        if x.shape[2:] != skip_input.shape[2:]:
            # ConvTranspose output size calculations may break this rule. -> resize x to match skip_input
            print(f"Warning: Skip connection size mismatch. x: {x.shape}, skip: {skip_input.shape}. Resizing x.")
            x = nn.functional.interpolate(x, size=skip_input.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    """
    Generator U-Net for the pix2pix architecture.
    This is the generator network that transforms input CT images into synthetic dose maps (image-to-image use-case).
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        # Encoder (Downsampling path for 128x128 input -> 1x1 bottleneck)
        self.down1 = UNetDown(in_channels, 64, normalize=False)        # Output: 64x64x64
        self.down2 = UNetDown(64, 128)                                 # Output: 32x32x128
        self.down3 = UNetDown(128, 256)                                # Output: 16x16x256
        self.down4 = UNetDown(256, 512, dropout=0.5)                   # Output: 8x8x512
        self.down5 = UNetDown(512, 512, dropout=0.5)                   # Output: 4x4x512
        self.down6 = UNetDown(512, 512, dropout=0.5)                   # Output: 2x2x512
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)  # Output: 1x1x512 (Bottleneck)

        # Decoder (Upsampling path)
        self.up1 = UNetUp(512, 512, dropout=0.5)   # Input: 1x1x512.   Output: 2x2x512.   After cat: 2x2x1024.  Skip from down6 (2x2x512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # Input: 2x2x1024.  Output: 4x4x512.   After cat: 4x4x1024.  Skip from down5 (4x4x512)
        self.up3 = UNetUp(1024, 512, dropout=0.5)  # Input: 4x4x1024.  Output: 8x8x512.   After cat: 8x8x1024.  Skip from down4 (8x8x512)
        self.up4 = UNetUp(1024, 256)               # Input: 8x8x1024.  Output: 16x16x256. After cat: 16x16x512. Skip from down3 (16x16x256)
        self.up5 = UNetUp(512, 128)                # Input: 16x16x512. Output: 32x32x128. After cat: 32z32x256. Skip from down2 (32x32x128)
        self.up6 = UNetUp(256, 64)                 # Input: 32x32x256. Output: 64x64x64.  After cat: 64x64x128. Skip from down1 (64x64x64)
        
        # Final layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1), # Output: 128 x 128 x out_channels
            nn.Tanh() # Tanh Output = [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6) # Bottleneck
        # Decpder
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final_up(u6)


# %% [markdown]
# ### Discriminator (PatchGAN)

# %%
class Discriminator(nn.Module):
    """
    GAN - Discriminator for the pix2pix architecture.
    
    Designed to distinguish between real and fake dose maps.
    - Trained to minimize difference b/t real and fake patches.
    - PatchGAN architecture that classifies 70x70 ovcrlapping patches.
    -> Output: single val claiming real or fake.
    """
    def __init__(self, in_channels=1): # Takes only dose image (real or fake)
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True, stride=2):
            """
            Discriminator block for the GAN architecture.
            
            Each block consists of a convolutional layer followed by an optional normalization and activation function.
            The stride parameter controls the downsampling factor. (ex. 2 on 4x4 kernel -> halving the input size, 1 = no change)
            """
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=stride, padding=1, bias=False)]
            if normalize: layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        # For 128x128 input to  dose map
        # Model Arch: C64 -> C128 -> C256 -> C512 -> Output Conv
        # Image size progression: 128 -> 64 -> 32 -> 16 -> 8
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),  # 128x128 -> 64x64
            *discriminator_block(64, 128),                           # 64x64   -> 32x32
            *discriminator_block(128, 256),                          # 32x32   -> 16x16
            *discriminator_block(256, 512),                          # 16x16   -> 8x8
            # Final convlution layer produces 1-channel patch output (real/fake)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)

# %% [markdown]
# ### Weights Initialization

# %%
def weights_init_normal(m):
    """
    This function inits weights of the model using a normal distribution.
    
    The initialization is based on the type of layer:
    - Conv layers are init with norm dist with mean 0 and std 0.02.
    - BatchNorm2d and InstanceNorm2d layers are init w/ a norm dist with mean 1 and std 0.02 for weights, and constant 0 for biases.
    - Other layers are init with a norm dist with mean 0 and std 0.02.
    """
    classname = m.__class__.__name__
    
    if classname.find("Conv") != -1:
        # For Conv layers, we use a normal distribution with mean 0 and std 0.02.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
            
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        # For Norm layers, bias is often initialized to 0 and weight to 1.
        if hasattr(m, "weight") and m.weight is not None:
             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

# %% [markdown]
# ## Training Loop

# %% [markdown]
# ### Helper Functions

# %%
def plot_sample_results(epoch, fixed_input_ct_batch, fixed_target_dose_batch, generator_model, device, save_dir="training_progress", num_samples_to_show=1):
    """
    Plots sample results from the generator model.
    """
    generator_model.eval()
    with torch.no_grad():
        # Get subset of samples to visualize
        input_ct_samples = fixed_input_ct_batch[:num_samples_to_show].to(device)
        target_dose_samples = fixed_target_dose_batch[:num_samples_to_show].to(device)
        generated_dose_samples = generator_model(input_ct_samples)

    # Convert tensors to numpy arrays for viz
    input_ct_samples = input_ct_samples.cpu().numpy()
    target_dose_samples = target_dose_samples.cpu().numpy()
    generated_dose_samples = generated_dose_samples.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(12, num_samples_to_show * 4))
    if num_samples_to_show == 1: axes = [axes]

    for i in range(num_samples_to_show):
        # For each sample, plot the input CT, target dose, and generated dose
        axes[i][0].imshow(input_ct_samples[i].squeeze(), cmap='gray', vmin=-1, vmax=1)
        axes[i][0].set_title(f"Input CT (Epoch {epoch+1})")
        axes[i][0].axis('off')

        axes[i][1].imshow(target_dose_samples[i].squeeze(), cmap='viridis', vmin=-1, vmax=1)
        axes[i][1].set_title("Synthetic Target Dose")
        axes[i][1].axis('off')

        axes[i][2].imshow(generated_dose_samples[i].squeeze(), cmap='viridis', vmin=-1, vmax=1)
        axes[i][2].set_title("Generated Dose")
        axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1:03d}_sample.png"))
    plt.show()
    plt.close(fig)
    generator_model.train()

# %%
def train_gan(generator, discriminator, dataloader, CONFIG, criterion_GAN, criterion_L1, optimizer_G, optimizer_D):
    """
    Training loop for the GAN model.
    """
    # Use first batch for visualizing model progress
    fixed_input_batch, fixed_target_batch = None, None
    if len(dataloader) > 0:
        try:
            fixed_input_batch, fixed_target_batch = next(iter(dataloader))
        except StopIteration:
            print("Error: Dataloader is empty, cannot get fixed batch for visualization.")
    else:
        print("Dataloader is empty. Cannot get fixed batch for visualization.")

    print(f"Starting training for {CONFIG['num_epochs']} epochs...")
    G_losses, D_losses, L1_losses = [], [], []

    if len(dataloader) == 0:
        print("Dataloader is empty. Skipping training loop.")
    else:
        patch_H, patch_W = CONFIG['D_patch_shape']
        for epoch in range(CONFIG['num_epochs']):
            epoch_g_loss, epoch_d_loss, epoch_l1_loss = 0, 0, 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
            for i, (real_input_ct, real_target_dose) in enumerate(progress_bar):
                # Move data to device
                real_input_ct = real_input_ct.to(CONFIG["device"])
                real_target_dose = real_target_dose.to(CONFIG["device"])

                # Get current batch size
                current_batch_size = real_input_ct.size(0)
                valid = torch.ones((current_batch_size, 1, patch_H, patch_W), device=CONFIG["device"], requires_grad=False)
                fake = torch.zeros((current_batch_size, 1, patch_H, patch_W), device=CONFIG["device"], requires_grad=False)

                # Train Generator
                optimizer_G.zero_grad()
                fake_dose = generator(real_input_ct)
                pred_fake_g = discriminator(fake_dose)
                loss_GAN_g = criterion_GAN(pred_fake_g, valid)
                loss_L1_g = criterion_L1(fake_dose, real_target_dose)
                loss_G = loss_GAN_g + CONFIG["lambda_l1"] * loss_L1_g
                loss_G.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                pred_real_d = discriminator(real_target_dose)
                loss_real_d = criterion_GAN(pred_real_d, valid)
                pred_fake_d = discriminator(fake_dose.detach()) # Detach to avoid backprop to G
                loss_fake_d = criterion_GAN(pred_fake_d, fake)
                loss_D = 0.5 * (loss_real_d + loss_fake_d)
                loss_D.backward()
                optimizer_D.step()

                # Calc loss
                epoch_g_loss += loss_G.item()
                epoch_d_loss += loss_D.item()
                epoch_l1_loss += loss_L1_g.item()

                if (i + 1) % CONFIG["display_interval"] == 0:
                     progress_bar.set_postfix({
                        'D_loss': f'{loss_D.item():.4f}',
                        'G_loss': f'{loss_G.item():.4f}',
                        'G_GAN': f'{loss_GAN_g.item():.4f}',
                        'G_L1':  f'{loss_L1_g.item():.4f}'
                     })
            
            # Average losses
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_l1_loss = epoch_l1_loss / len(dataloader)
            G_losses.append(avg_g_loss)
            D_losses.append(avg_d_loss)
            L1_losses.append(avg_l1_loss)

            print(f"Epoch {epoch+1} Summary: Avg D loss: {avg_d_loss:.4f}, Avg G loss: {avg_g_loss:.4f} (L1: {avg_l1_loss:.4f})")

            if fixed_input_batch is not None and ((epoch + 1) % 5 == 0 or epoch == CONFIG['num_epochs'] - 1):
                plot_sample_results(epoch, fixed_input_batch, fixed_target_batch, generator, CONFIG["device"], num_samples_to_show=min(3, fixed_input_batch.size(0)))

        print("Training finished.")

        plt.figure(figsize=(10, 5))
        plt.plot(G_losses, label="Generator Total Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.plot(L1_losses, label="Generator L1 Loss (raw)")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.title("Training Losses")
        plt.savefig("training_losses_liver.png"); plt.show()
        
        return fixed_input_batch, fixed_target_batch
