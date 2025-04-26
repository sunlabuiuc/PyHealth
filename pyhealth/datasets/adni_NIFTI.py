from pyhealth.datasets import BaseDataset
import nibabel as nib
import pandas as pd
import numpy as np

class ADNICognitiveDataset(BaseDataset):
    def __init__(self, dataset_name: str, root: str):
        super().__init__(dataset_name=dataset_name, root=root)
        self.load_data()

    def load_cohort(adni_csv_path):
        # Load the ADNI dataset containing subject information
        adni_df = pd.read_csv(adni_csv_path)

        # Filter for AD and CN subjects
        selected_df = adni_df[adni_df['Group'].isin(['AD', 'CN'])]

        # For matched ages, we could apply a simple age range filtering (e.g., mean age ± 7 years)
        mean_age = 77  #Selects AD and CN subjects from ADNI with matched age (mean 77 ± 7)
        age_range = (mean_age - 7, mean_age + 7)
        selected_df = selected_df[selected_df['Age'].between(age_range[0], age_range[1])]

        # Keep only 358 subjects
        selected_df = selected_df.sample(n=38, random_state=42)  # Randomly select subset if needed

        # Map labels: 'CN' -> 0, 'AD' -> 1
        selected_df['label'] = selected_df['Group'].map({'CN': 0, 'AD': 1})
        selected_df['subjectID'] = selected_df['Subject']

        return selected_df[['subjectID', 'label']]

    def normalize_intensity(image):
        #Normalize intensity to [0,1] range.
        img = image.astype(np.float32)
        img -= img.min()
        img /= img.max()
        return img

    def crop_brain_region(image, crop_size=(120, 144, 120)):
        """Crop the image to the inner brain region in MNI space."""
        if (len(image.shape) == 4):
          x, y, z, zz = image.shape
        else:
          x, y, z = image.shape

        cx, cy, cz = crop_size
        start_x, start_y, start_z = (x - cx) // 2, (y - cy) // 2, (z - cz) // 2
        return image[start_x:start_x+cx, start_y:start_y+cy, start_z:start_z+cz]

    def preprocess_fmriprep_mni(filepath):
        """Preprocess an MNI-aligned fMRIPrep NIfTI image: normalization & cropping."""
        img = nib.load(filepath).get_fdata()
        img = normalize_intensity(img)
        img = crop_brain_region(img)
        return img

    def extract_patches(image, patch_size=(30, 36, 30)):
        """Extract non-overlapping 3D patches from the cropped MNI-aligned image."""
        if (len(image.shape) == 4):
          cx, cy, cz, czz = image.shape
        else:
          cx, cy, cz = image.shape
        px, py, pz = patch_size
        patches = []

        for i in range(0, cx, px):
            for j in range(0, cy, py):
                for k in range(0, cz, pz):
                    patch = image[i:i+px, j:j+py, k:k+pz]
                    if patch.shape == patch_size:  # Ensure exact patch size
                        patches.append(patch)

        return np.array(patches)


	def load_data(self):
		image_dir = os.path.join(self.root, "images")
		label_file = os.path.join(self.root, "labels.csv")
        
        labels_df = load_cohort(label_file)

        for subject_id, label in labels_df[['subjectID', 'label']].values:
            # Assuming images are in the format 'sub-<subject_id>_T1w.nii.gz'
            image_path = f"{image_dir}/{subject_id}.nii.gz"
            
            img = preprocess_fmriprep_mni(image_path) #Output shape (120,144,120)

            # Extract patches from image  64 non-overlapping patches of 30x36x30 px. CONFIRMED CORRECT
            patches = extract_patches(img) # Output shape (64, 30, 360, 30)

            # Compute persistent homology for TDA
            persistent_features = compute_persistent_homology_with_GUDHI(img)
            
            self.samples.append(subject_id, patches, persistent_features, label)
