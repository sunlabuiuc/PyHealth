# Contributor: [Your Name]
# NetID: [Your NetID]

import os
import pandas as pd
from pathlib import Path
from pyhealth.datasets import BaseDataset

class DermoscopyDataset(BaseDataset):
    """Dataset loader for Dermoscopy image collections (ISIC 2018, HAM10000, PH2).
    
    Supports combining arbitrary combinations of datasets to replicate the transfer-learning methodologies detailed in Jin (2025).

    Paper Reference:
        - "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)

    Dataset Citations:
        - ISIC 2018: Codella et al., "Skin Lesion Analysis Toward Melanoma Detection" (2018)
        - HAM10000: Tschandl et al., "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions" (2018)
        - PH2: Mendonça et al., "PH2 - A dermoscopic image database for research and benchmarking" (2013)

    EXPECTED LOCAL DIRECTORY STRUCTURE:
    ===========================================================================
    data/
    ├── ham10000/
    │   ├── images/
    │   │   ├── metadata.csv                     <-- MUST contain 'isic_id' & 'diagnosis_1'
    │   │   ├── ISIC_0034320.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── ISIC_0034320_segmentation.png
    │       └── ...
    ├── isic2018/
    │   ├── images/
    │   │   ├── metadata.csv                     <-- MUST contain 'isic_id' & 'diagnosis_1'
    │   │   ├── ISIC_0016072.jpg
    │   │   └── ...
    │   └── masks/
    │       ├── ISIC_0016072_segmentation.png
    │       └── ...
    └── ph2/
        ├── PH2_dataset.txt                      <-- Original PH2 annotation file
        └── PH2 Dataset images/
            ├── IMD002/
            │   ├── IMD002_Dermoscopic_Image/IMD002.bmp
            │   └── IMD002_lesion/IMD002_lesion.bmp
            └── ...
    ===========================================================================

    Args:
        root (str): Root directory containing the dataset folders.
        datasets (list of str, optional): List of dataset folders to include 
            (e.g., ["isic2018", "ham10000"]). 
        dataset_name (str, optional): Backwards compatibility for PyHealth loader.
        dev (bool, optional): If True, runs in development mode.
    """
    def __init__(self, root, datasets=None, dataset_name=None, **kwargs):
        # 1. Allow flexible dataset combinations for transfer learning
        if datasets is not None:
            self.datasets = datasets
            name = "_".join(datasets)
        elif dataset_name is not None:
            self.datasets = [dataset_name]
            name = dataset_name
        else:
            self.datasets = ["isic2018"]
            name = "isic2018"
            
        self._index_data(root)
        config_path = str(Path(__file__).parent / "configs" / "dermoscopy.yaml")

        super().__init__(
            root=root,
            tables=["dermoscopy"],
            dataset_name=name, 
            config_path=config_path, 
            **kwargs
        )

    def _index_data(self, root):
        """Internal method to aggregate metadata from targeted sources."""
        all_dfs = []
        for folder in self.datasets:
            # Handle native PH2 format and generated PH2 Trap Sets
            if "ph2" in folder:
                txt_path = os.path.join(root, folder, "PH2_dataset.txt")
                img_root_dir = os.path.join(root, folder, "PH2 Dataset images")
                
                # If it's a generated trap set with a standard CSV, it skips this and uses the CSV parser below
                if os.path.exists(txt_path): 
                    records = []
                    with open(txt_path, 'r') as f:
                        for line in f:
                            if line.startswith('|| IMD'):
                                parts = line.split('||')
                                if len(parts) >= 4:
                                    imd_id = parts[1].strip()
                                    label = 1 if parts[3].strip() == '2' else 0
                                    patient_folder = os.path.join(img_root_dir, imd_id)
                                    img_path = os.path.join(patient_folder, f"{imd_id}_Dermoscopic_Image", f"{imd_id}.bmp")
                                    mask_path = os.path.join(patient_folder, f"{imd_id}_lesion", f"{imd_id}_lesion.bmp")
                                    records.append({"image_path": img_path, "mask_path": mask_path, "label": label, "source_dataset": folder, "patient_id": imd_id, "visit_id": imd_id})
                    all_dfs.append(pd.DataFrame(records))
                    continue
                
            # Standard ISIC/HAM10000 CSV metadata parsing
            csv_path = os.path.join(root, folder, "images", "metadata.csv")
            img_dir = os.path.join(root, folder, "images")
            if not os.path.exists(csv_path): continue

            df = pd.read_csv(csv_path)
            id_col = 'isic_id' if 'isic_id' in df.columns else 'image_id'
            df['image_path'] = df[id_col].apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
            if 'diagnosis_1' in df.columns:
                df['label'] = df['diagnosis_1'].apply(lambda x: 1 if str(x).strip().lower() == 'malignant' else 0)
            
            df['mask_path'] = "" 
            df['source_dataset'] = folder
            df['patient_id'] = df[id_col]
            df['visit_id'] = df[id_col]
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(root, f"dermoscopy-metadata-{'_'.join(self.datasets)}.csv")
        combined_df.to_csv(out_path, index=False)