
# Dermoscopy Artifact Robustness under Diffusion-Based Perturbations

This repository contains the complete experimental pipeline and reproducibility codebase for analyzing the robustness of foundation models in dermatology against diffusion-generated artifacts. This work builds upon and extends the methodology presented in _A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations (CHIL 2025)_.

## 📚 Core Citations & Repositories

**Core Methodology & Annotations:**

- **Primary Methodology:** Jin, Q., et al. (2025). _A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations._ (CHIL 2025). [Authors' Original Repository](https://github.com/QixuanJin99/dermoscopic_artifacts/tree/main)

- **Artifact Annotations:** Bissoto, A., Valle, E., & Avila, S. (2020). _Debiasing skin lesion datasets and models? Not so fast._ In Proceedings of the IEEE/CVF CVPR Workshops (pp. 740-741). [Annotation Repository](https://github.com/alceubissoto/debiasing-skin/tree/master/artefacts-annotation)

**Datasets:**

- **ISIC 2018:** Codella, N., et al. (2019). _Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)._ arXiv. [Dataset Link](https://challenge.isic-archive.com/data/#2018)

- **HAM10000:** Tschandl, P., Rosendahl, C., & Kittler, H. (2018). _The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions._ Scientific Data, 5, 180161. [Dataset Link](https://challenge.isic-archive.com/data/#2018)

- **PH2:** Mendonça, T., et al. (2013). _PH2 - A dermoscopic image database for research and benchmarking._ 35th Annual International Conference of the IEEE EMBS. [Dataset Link](https://www.fc.up.pt/addi/ph2%20database.html)

**Generative Modeling (Phase 2):**

- **DreamBooth:** Ruiz, N., et al. (2023). _DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation._ CVPR.

- **LoRA:** Hu, E. J., et al. (2022). _LoRA: Low-Rank Adaptation of Large Language Models._ ICLR.

- **Diffusers:** von Platen, P., et al. (2022). _Diffusers: State-of-the-art diffusion models._ Hugging Face.

**Vision Architectures:**

- **ResNet50:** He, K., et al. (2016). _Deep Residual Learning for Image Recognition._ CVPR.

- **Swin Tiny:** Liu, Z., et al. (2021). _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows._ ICCV.

- **DINOv2:** Oquab, M., et al. (2024). _DINOv2: Learning Robust Visual Features without Supervision._ TMLR.

- **ConvNeXT Tiny:** Liu, Z., et al. (2022). _A ConvNet for the 2020s._ CVPR.

## 🖥 Hardware & Compute Environment

To provide a benchmark for execution time and memory constraints, all experiments, data curation, and model training in this repository were conducted on the **Google Cloud Platform (GCP)** using the following instance configuration:

- **Machine Type:** `g2-standard-8`

- **CPU:** 8 vCPUs (Intel Cascade Lake Platform, Intel(R) Xeon(R) CPU @ 2.20GHz)

- **Memory:** 32 GB RAM

- **GPU:** 1 x NVIDIA L4 GPU (24GB VRAM)

- **Storage:** 300 GB SSD Persistent Disk

- **Operating System:** Ubuntu 22.04.5 LTS (Deep Learning VM Image)

- **Shell Environment:** Bash (v5.1+)

- **Compute Framework:** CUDA 13.2 / NVIDIA Driver 595+

## 🛠 Project Structure & PyHealth Setup

Our experimental setup operates inside and alongside the PyHealth repository.

- **Data Directory:** `../data`

- **Outputs/Weights:** `~/dermoscopy_outputs`

- **Plot Exports:** `~/heatmaps`, `~/ablation_plots`, `~/methodology_plots`

- **Log Directory:** `~/dermoscopy_logs`

  - _Note on Logging:_ In addition to the standard text logs parsed by our custom aggregator scripts, `train_dermoscopy.py` is configured to automatically generate **TensorBoard** event files for real-time tracking of validation metrics and training loss curves of each fold. These files are saved to `~/dermoscopy_outputs` in the specific model folds.

### The PyHealth Cache

By default, the first time `DermoscopyDataset` is initialized, it processes and caches all supported datasets found in the data directory. To keep file systems organized, we strictly route the cache into `../data/.cache` rather than the user's home directory.

**Trap Set Caching Behavior:** You will notice that `DermoscopyDataset` does _not_ automatically load the synthetic artifact datasets (e.g., `ph2_with_ruler`) upon first initialization. This is intentional. The dataset loader only auto-searches for hardcoded standard medical datasets (`isic2018`, `ham10000`, `ph2`). The custom Trap Sets are only processed and **appended** to the existing `.cache` file—without overlapping the original baseline data—when explicitly requested by the evaluation scripts (e.g., passing `--eval_dataset ph2_with_ruler`).

**Troubleshooting Metadata:** If the master `dermoscopy-metadata-pyhealth.csv` is ever overwritten or corrupted during training, it can be instantly regenerated via:

```bash
python -c "import os; from pyhealth.datasets import DermoscopyDataset; DermoscopyDataset(root='../data', cache_dir=os.path.join('../data', '.cache'))"

```

### Core PyHealth Extensions

To support this highly specialized dermoscopy pipeline, we extended the base PyHealth repository by authoring several core architecture files and modifying one core architecture file. These files define the custom datasets, preprocessing logic, tasks, and vision architectures utilized throughout the experiments:

- `pyhealth/datasets/dermoscopy.py`: The custom dataset loader built to seamlessly ingest ISIC2018, HAM10000, PH2, and dynamically route our generated Trap Sets.

- `pyhealth/processors/dermoscopy_image_processor.py`: The image processing engine responsible for handling the 12 distinct spatial and frequency ablation modes (e.g., generating high-pass filters, applying bounding box masks).

- `pyhealth/tasks/dermoscopy_melanoma_classification.py`: Defines the formal PyHealth task, standardizing the problem into a binary classification structure (benign vs. malignant).

- `pyhealth/models/dinov2.py`: A custom model integration bringing Meta's DINOv2 Vision Transformer natively into the PyHealth ecosystem.

- `pyhealth/models/torchvision_model.py`: Modified to officially support, register, and route the ConvNeXT Tiny architecture. This script also already supports the Resnet50 and Swin Tiny architectures.

## ✅ Unit Tests

Before running the full pipelines, you can verify the integrity of the data loaders and model architectures using the provided unit tests. These are located in `/tests/core` and utilize three sample images from the HAM10000 dataset stored locally in `/test-resources/core/dermoscopy`.

Run the following tests to ensure your PyHealth environment is configured correctly:

- `pytest tests/core/test_dermoscopy.py`

- `pytest tests/core/test_convnext.py`

- `pytest tests/core/test_dinov2.py`

## 🚀 Pipeline 1: Basic Melanoma Classification

This pipeline demonstrates a standard, lightweight workflow for melanoma classification using PyHealth. It is ideal for testing your local environment, exploring the dataset structure, or running a quick baseline training loop before advancing to the complex artifact studies.

**Relevant Files (located in `/examples/dermoscopy`):**

- `dermoscopy_melanoma_classification.py`

- `dermoscopy_melanoma_classification.ipynb`

**Example Run:**

```bash
python examples/dermoscopy/dermoscopy_melanoma_classification.py \
    --model resnet50 \
    --epochs 5 \
    --data_dir ../data
```

You can also walk through the Jupyter Notebook for an interactive, cell-by-cell execution of this pipeline.

## 🔬 Pipeline 2: Artifact Robustness Experiments

This is the experimental pipeline of the repository. It scales up the core PyHealth classification logic into a massive framework designed to generate synthetic artifacts, evaluate Out-Of-Distribution (OOD) robustness, map epoch learning dynamics, and compare aggregation methodologies.

**Core Pipeline Files (located in `/examples/dermoscopy`):**

- `train_dermoscopy.py`

- `generate_artifact_data.py`

- `evaluate_artifact_robustness.py`

- `run_epoch_ablation.py`

## 🧪 Phase 1: Baseline Model Training (Tables 1 & 2)

**Intent:** The goal of this phase is to establish baseline performance metrics for multiple architectures (ResNet50, Swin Tiny, DINOv2, and ConvNeXT) across standard source (in-domain) and target (out-of-domain) datasets, utilizing 5-fold cross-validation.

**Training Scale:** We originally train a massive matrix of **96 distinct 5-fold cross-validation models** (2 training datasets × 12 modes × 4 architectures).

**Ablation Modes:** To investigate whether models rely on specific physical regions or frequency spectrums, we train and evaluate across 12 distinct data modes:

- **Spatial Ablations:** `whole` (standard image), `lesion` (segmented lesion only), `background` (skin only, lesion masked out), and bounding box crops (`bbox`, `bbox70`, `bbox90`).

- **Frequency Ablations:** High-pass and low-pass filtered versions (`high_whole`, `low_lesion`, etc.) to isolate high-frequency textures from low-frequency shapes.

**Weights:** Weights are saved to `~/dermoscopy_outputs` in the specific model folders.

**Logging:** Logs are saved to `~/dermoscopy_logs/train`

### `run_all_original.sh`

(executed inside root directory of PyHealth repo)
This script handles the training pipeline, storing temporary weights and producing detailed logs.

```bash
#!/bin/bash
# =====================================================================
# Training the Baseline Models (Tables 1 & 2)
# Reproduces the original paper baselines across 4 model architectures, 
# 2 training datasets, and all 12 dataset perturbation modes.
# =====================================================================
DATA_DIR="../data"
MODELS=("resnet50" "swin_t" "dinov2" "convnext_tiny")
MODES=(
    "whole" "lesion" "background" 
    "bbox" "bbox70" "bbox90" 
    "high_whole" "high_lesion" "high_background" 
    "low_whole" "low_lesion" "low_background"
)

echo "====================================================================="
echo "[*] STARTING BASELINE MODEL TRAINING (TABLES 1 & 2)"
echo "====================================================================="
# ---------------------------------------------------------------------
# Part A: Train Models on ISIC2018 (Evaluated on PH2 and HAM10000)
# ---------------------------------------------------------------------
echo ""
echo "---------------------------------------------------------------------"
echo "[*] PART A: TRAINING ON ISIC2018"
echo "---------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        
        echo "[*] Executing: Train(ISIC2018) | Model(${MODEL}) | Mode(${MODE})"
        
        python examples/dermoscopy/train_dermoscopy.py \
            --model "$MODEL" \
            --mode "$MODE" \
            --data_dir "$DATA_DIR" \
            --epochs 10 \
            --train_datasets isic2018 \
            --test_datasets ph2 ham10000
            
    done
done
# ---------------------------------------------------------------------
# Part B: Train Models on PH2 (Evaluated on ISIC2018 and HAM10000)
# ---------------------------------------------------------------------
echo ""
echo "---------------------------------------------------------------------"
echo "[*] PART B: TRAINING ON PH2"
echo "---------------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "[*] Executing: Train(PH2) | Model(${MODEL}) | Mode(${MODE})"

        python examples/dermoscopy/train_dermoscopy.py \
            --model "$MODEL" \
            --mode "$MODE" \
            --data_dir "$DATA_DIR" \
            --epochs 10 \
            --train_datasets ph2 \
            --test_datasets isic2018 ham10000
    done
done

echo "====================================================================="
echo "[*] BASELINE MODEL TRAINING FULLY COMPLETE!"
echo "====================================================================="
```

### Table Generation

Once Phase 1 completes, running this aggregator parses the logs in `~/dermoscopy_logs/train` and generates formatted summary tables to visualize the baseline results.

**Logging:** Logs are saved to `~/dermoscopy_logs/aggregator`

### `build_tables1and2.py`

(executed outside PyHealth repo in home folder)

```python
"""
Tables 1 & 2 Aggregator.

Run this script AFTER your Phase 1 bash training loops finish. 
It scans the generated `train_...txt` log files, extracts the 5-fold CV 
ROC-AUC scores (Mean ± Std) for each Mode/Dataset combination, and prints 
the formatted Tables 1 and 2.
"""

import os
from collections import defaultdict

from PyHealth.examples.dermoscopy.train_dermoscopy import setup_dynamic_logging

# The exact order the modes appear in the CHIL 2025 paper
MODE_ORDER = [
    "whole", "lesion", "background", 
    "bbox", "bbox70", "bbox90",
    "high_whole", "high_lesion", "high_background",
    "low_whole", "low_lesion", "low_background"
]

def main(log_dir=None, model='resnet50'):
    if log_dir is None:
        print('[!] No log directory provided. Please specify the parent directory of the logs generated by your evaluation runs.')
        return

    # STRICT FILTER: Only grab files that start with ph2_ or isic2018_ (and explicitly ignore isic2018_ham10000_)
    train_files = [
        f for f in os.listdir(log_dir) 
        if f.lower().endswith(".txt") and 
        (f.lower().startswith("ph2_") or (f.lower().startswith("isic2018_") and not f.lower().startswith("isic2018_ham10000_")))
    ]
    
    # Filter by model
    train_files = [f for f in train_files if model.lower() in f.lower()]
    
    if not train_files:
        print(f"[!] No training logs found in {log_dir} for model {model.upper()}.")
        return

    # Data structure: results[train_dataset][mode][eval_dataset] = "0.0000 ± 0.0000"
    results = defaultdict(lambda: defaultdict(dict))

    print(f"[*] Parsing {len(train_files)} Phase 1 training logs for {model.upper()}...")
    for filename in train_files:
        filepath = os.path.join(log_dir, filename)
        
        # Format expected: {train_dataset}_{model}_{mode}_[timestamp].txt
        # e.g., isic2018_resnet50_whole_20250414_120000.txt
        parts = filename.split('_')
        try:
            train_dataset = parts[0]
            parsed_model = parts[1]
            mode = parts[2]
            
            # Handle modes that have underscores in their name (e.g., high_whole)
            if parts[3] in ["whole", "lesion", "background"]:
                mode = f"{parts[2]}_{parts[3]}"

            if parts[1] == 'swin':
                parsed_model =  'swin_t'
                mode = parts[3]
                if parts[4] in ["whole", "lesion", "background"]:
                    mode = f"{parts[3]}_{parts[4]}"
            elif parts[1] == 'convnext':
                parsed_model =  'convnext_tiny'
                mode = parts[3]
                if parts[4] in ["whole", "lesion", "background"]:
                    mode = f"{parts[3]}_{parts[4]}"

        except IndexError:
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.readlines()
            
            # Parse the text lines looking for the summary table
            for line in content:
                # Regex looks for lines formatted like: "SOURCE (isic2018) | roc_auc    | 0.7920 ± 0.0240"
                # or "HAM10000        | roc_auc    | 0.7210 ± 0.0610"
                if "roc_auc" in line and "±" in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        eval_ds_raw = parts[0].strip()
                        score = parts[2].strip()
                        
                        # Clean up the evaluation dataset name
                        if "SOURCE" in eval_ds_raw:
                            eval_ds = "Source"
                        else:
                            eval_ds = eval_ds_raw.upper()
                            
                        results[train_dataset][mode][eval_ds] = score

    # ==========================================
    # PRINT TABLE 1: ISIC 2018 TRAINED
    # ==========================================
    if "isic2018" in results:
        print("\n" + "="*80)
        print(f" TABLE 1: ROC-AUC (%) OF MODELS TRAINED ON ISIC 2018 | MODEL: {model.upper()} ")
        print("="*80)
        
        # Columns based on the paper
        headers = ["Mode", "ISIC 2018 (Source)", "HAM10000 (Target)", "PH2 (Target)"]
        header_str = f"{headers[0]:<16} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<20}"
        print(header_str)
        print("-" * len(header_str))
        
        isic_data = results["isic2018"]
        for mode in MODE_ORDER:
            if mode in isic_data:
                source = isic_data[mode].get("Source", "N/A")
                ham = isic_data[mode].get("HAM10000", "N/A")
                ph2 = isic_data[mode].get("PH2", "N/A")
                print(f"{mode:<16} | {source:<20} | {ham:<20} | {ph2:<20}")
        print("="*80)

    # ==========================================
    # PRINT TABLE 2: PH2 TRAINED
    # ==========================================
    if "ph2" in results:
        print("\n" + "="*80)
        print(f" TABLE 2: ROC-AUC (%) OF MODELS TRAINED ON PH2 | MODEL: {model.upper()} ")
        print("="*80)
        
        # Columns based on the paper
        headers = ["Mode", "PH2 (Source)", "ISIC 2018 (Target)", "HAM10000 (Target)"]
        header_str = f"{headers[0]:<16} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<20}"
        print(header_str)
        print("-" * len(header_str))
        
        ph2_data = results["ph2"]
        for mode in MODE_ORDER:
            if mode in ph2_data:
                source = ph2_data[mode].get("Source", "N/A")
                isic = ph2_data[mode].get("ISIC2018", "N/A")
                ham = ph2_data[mode].get("HAM10000", "N/A")
                print(f"{mode:<16} | {source:<20} | {isic:<20} | {ham:<20}")
        print("="*80)

if __name__ == "__main__":
    # Initialize logging ONCE for all models
    parent_log_dir = os.path.expanduser("~/dermoscopy_logs")
    run_details = "tables1and2_summary_all_models"
    logfilename = setup_dynamic_logging(parent_log_dir, "aggregator", run_details)

    # Resolve the log directory once
    base_log_dir = os.path.join(os.path.dirname(os.path.dirname(logfilename)), 'train')

    # Loop through models using the single log session
    model_list = ["dinov2", "convnext", "resnet50", "swin_t"]
    for target_model in model_list:
        main(log_dir=base_log_dir, model=target_model)
```

## 🧬 Phase 2: Generating Synthetic Artifacts (LoRA)

**Intent:** To evaluate artifact robustness, we need highly controlled "Trap Sets." Because the original authors did not release their specific DreamBooth weights, we curate data and train our own custom LoRAs for five artifacts: `dark-corner`, `gel-bubble`, `ink`, `patches`, and `ruler`. _(Note: Artifacts are named with hyphens to prevent file-parsing conflicts)._

### 1. Data Curation & Resizing

Utilizing the labeled ISIC2018 dataset from Bissoto et al., we manually cropped 16-20 clear, square images for each artifact (where the artifact constitutes 30-60% of the image). We ensured roughly 50:50 splits for `malignant` vs `benign` lesions. For `patches` (where only 1 malignant label existed), we rotated the image to 4 different orientations to achieve a 4:16 ratio.

**Extra image directories:** `~/curated_artifacts` is where the cropped images were manually stored.

### `resize_crops.py`

(executed outside PyHealth repo in home folder)

```python
import os
from PIL import Image

def main():
    """
    Batch resizes all manually cropped 1:1 square images to exactly 512x512 
    pixels using high-quality LANCZOS downsampling for Stable Diffusion.
    Supports nested 'benign' and 'malignant' subfolders for purist training.
    """
    base_dir = "./curated_artifacts"
    target_size = (512, 512)

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Please create it and add your crops.")
        return

    for artifact_folder in os.listdir(base_dir):
        artifact_path = os.path.join(base_dir, artifact_folder)
        
        if os.path.isdir(artifact_path):
            processed_count = 0
            
            # Loop through both diagnostic subfolders
            for diagnosis in ["benign", "malignant"]:
                diag_path = os.path.join(artifact_path, diagnosis)
                
                if os.path.isdir(diag_path):
                    for filename in os.listdir(diag_path):
                        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                            filepath = os.path.join(diag_path, filename)
                            
                            with Image.open(filepath) as img:
                                # Force RGB to prevent channel errors during training
                                img = img.convert("RGB")
                                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                                resized_img.save(filepath)
                            processed_count += 1
                            
            print(f"[+] Resized {processed_count} images in {artifact_folder} to 512x512.")

if __name__ == "__main__":
    main()
```

### 2. DreamBooth LoRA Training

Outside the PyHealth repo, we performed DreamBooth LoRA training by following these requirements and automatically downloading `train_dreambooth_lora.py`:

(executed outside PyHealth repo in home folder)

```bash
pip install diffusers accelerate transformers peft
wget -O train_dreambooth_lora.py https://raw.githubusercontent.com/huggingface/diffusers/v0.37.1/examples/dreambooth/train_dreambooth_lora.py
accelerate config
```

**Fixing the HuggingFace Script:**
To prevent hidden `.ipynb_checkpoints` from crashing the data loader, we updated line ~600 in `train_dreambooth_lora.py`:

```python
# Modified in train_dreambooth_lora.py
# Replace this: Original Instance Images Path
self.instance_images_path = list(Path(instance_data_root).iterdir())

# With this: New Instance Images Path
self.instance_images_path = [
    p for p in Path(instance_data_root).iterdir() 
    if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
]
```

**Fixing the LoRA Alpha Default:**
The paper utilizes an Alpha of 32 to scale the LoRA weight updates by 0.5 (`alpha / rank`), preventing the model from overfitting to the artifacts and forgetting core lesion morphology. However, `train_dreambooth_lora.py` does not expose an `--alpha` flag; it hardcodes `lora_alpha=args.rank`. To replicate the paper, we manually edited the source code at line ~939 to force the correct scaling:

```python
# Modified in train_dreambooth_lora.py
unet_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=32, # Changed from args.rank
    # ...
)
```

We utilize the paper's specific token and strength mapping parameters (e.g., `dark-corner` -> `lun`, `ink` -> `httr`). We instruct the model to generate 200 prior preservation images (`class_images/benign` and `class_images/malignant`) to ensure the foundation model retains the core morphological concepts of a skin lesion while injecting the artifact token.

#### Hyperparameter Alignment (CHIL 2025)

Our DreamBooth LoRA training pipeline strictly adheres to the prompt conditioning, prior preservation mechanics, and base architectures (`runwayml/stable-diffusion-v1-5`) outlined in Jin (2025).

**Steps vs. Epochs:**
Jin (2025) states that models were fine-tuned for "3 to 5 epochs." Because the Hugging Face training script measures duration in discrete optimization steps rather than epochs, we utilize `--max_train_steps=500`. This aligns with the target:

- We curate roughly **20** artifact images and generate **200** prior-preservation class images (Total Dataset = ~220 images).
- Utilizing a batch size of 2, it takes **110 steps** to complete one full epoch (220 images / 2).
- Therefore, 500 steps results in **~4.5 epochs**, hitting the authors' optimal training window.

**Strict Replication Flags:**
By default, standard Hugging Face scripts utilize lower LoRA ranks and do not optimize the text encoder. To attempt a 1:1 replication of the CHIL 2025 paper, we appended the following specific flags to our `accelerate launch` command:

- `--train_text_encoder`: The authors explicitly fine-tuned both the UNet and the Text Encoder to ensure the model tightly binds the rare token to the visual artifact.
- `--rank=64`: Defines the dimension of the LoRA update matrices. A rank of 64 provides enough capacity to learn complex artifact textures.
- `--num_class_images=200`: Generates 200 standard skin lesion images without the artifact token. This is part of the prior preservation technique that prevents catastrophic forgetting.
- `--prior_loss_weight=0.3`: Scales the importance of the prior preservation images during training.
- `--mixed_precision="fp16"`: Utilizes 16-bit floating-point math rather than 32-bit. This reduces GPU VRAM consumption and accelerates training time.
- `--train_batch_size=2`: Matches the paper's batching memory constraints and is required to calculate the correct gradient updates per epoch.
- `--seed="0"`: Ensures deterministic noise generation identical to the authors' original compute environment.

**Weights:** Weights are saved to `~/artifact_lora_weights` in the specific artifact folders.

**Extra image directories:** Class images are saved to `~/class_images`

### `train_artifact_loras.sh`

(executed outside PyHealth repo in home folder)

```bash
#!/bin/bash
# ==============================================================================
# Script: train_artifact_loras.sh
# Purpose: Replication of Jin (2025) text conditioning methodology.
# Uses Sequential Training to map exact diagnosis prompts to the DreamBooth LoRA.
#
# Artifact Ground Truth labels sourced from:
# Bissoto, A., Valle, E., & Avila, S. (2020). Debiasing skin lesion datasets 
# and models? Not so fast. In Proceedings of the IEEE/CVF Conference on 
# Computer Vision and Pattern Recognition Workshops (pp. 740-741).
# Repository: https://github.com/alceubissoto/debiasing-skin/tree/master/artefacts-annotation
# ==============================================================================

MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_DIR="./curated_artifacts"
OUTPUT_DIR="./artifact_lora_weights"
CLASS_DIR="./class_images"

ARTIFACTS=("dark-corner" "gel-bubble" "ink" "patches" "ruler")
TOKENS=("lun" "sown" "httr" "olis" "dits")

echo "[*] Starting Artifact DreamBooth LoRA Training Pipeline..."

for i in "${!ARTIFACTS[@]}"; do
    ARTIFACT="${ARTIFACTS[$i]}"
    TOKEN="${TOKENS[$i]}"
    SAVE_DIR="${OUTPUT_DIR}/${ARTIFACT}"
    
    echo "[*] ========================================================"
    echo "[*] Processing Artifact: $ARTIFACT (Token: $TOKEN)"
    echo "[*] ========================================================"
    # ---------------------------------------------------------
    # PHASE 1: Train on Benign Images (Steps 0 to 500)
    # ---------------------------------------------------------
    if [ -d "${DATA_DIR}/${ARTIFACT}/benign" ] && [ "$(ls -A ${DATA_DIR}/${ARTIFACT}/benign)" ]; then
        echo "[+] Found benign images. Training Phase 1..."
        
        accelerate launch train_dreambooth_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --instance_data_dir="${DATA_DIR}/${ARTIFACT}/benign" \
          --class_data_dir="${CLASS_DIR}/benign" \
          --output_dir=$SAVE_DIR \
          --instance_prompt="a dermoscopic image of ${TOKEN} benign" \
          --class_prompt="a dermoscopic image of benign" \
          --with_prior_preservation \
          --prior_loss_weight=0.3 \
          --num_class_images=200 \
          --resolution=512 \
          --train_batch_size=2 \
          --gradient_accumulation_steps=1 \
          --mixed_precision="fp16" \
          --learning_rate=1e-4 \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --checkpointing_steps=500 \
          --train_text_encoder \
          --rank=64 \
          --seed="0"
    fi

    # ---------------------------------------------------------
    # PHASE 2: Train on Malignant Images (Steps 500 to 1000)
    # ---------------------------------------------------------
    if [ -d "${DATA_DIR}/${ARTIFACT}/malignant" ] && [ "$(ls -A ${DATA_DIR}/${ARTIFACT}/malignant)" ]; then
        echo "[+] Found malignant images. Training Phase 2..."
        # Check if we need to resume from the benign run
        RESUME_FLAG=""
        if [ -d "$SAVE_DIR/checkpoint-500" ]; then
            echo "[*] Resuming from Benign checkpoint..."
            RESUME_FLAG="--resume_from_checkpoint=latest"
        fi

        accelerate launch train_dreambooth_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --instance_data_dir="${DATA_DIR}/${ARTIFACT}/malignant" \
          --class_data_dir="${CLASS_DIR}/malignant" \
          --output_dir=$SAVE_DIR \
          --instance_prompt="a dermoscopic image of ${TOKEN} malignant" \
          --class_prompt="a dermoscopic image of malignant" \
          --with_prior_preservation \
          --prior_loss_weight=0.3 \
          --num_class_images=200 \
          --resolution=512 \
          --train_batch_size=2 \
          --gradient_accumulation_steps=1 \
          --mixed_precision="fp16" \
          --learning_rate=1e-4 \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --checkpointing_steps=500 \
          $RESUME_FLAG \
          --train_text_encoder \
          --rank=64 \
          --seed="0"
    fi

    echo "[+] Completed artifact training for $ARTIFACT!"
done

echo "[+] All artifact LoRAs successfully generated!"
```

### 3. Generating the Trap Sets

The resulting `.safetensors` weights are used to perturb the PH2 dataset, generating isolated Trap Sets (e.g., `ph2_with_ruler`) that are saved into the `../data` directory.

**Logging:** Logs are saved to `~/dermoscopy_logs/generate_artifacts`

### `run_all_generate_artifacts.sh`

(executed inside root directory of PyHealth repo)

```bash
python examples/dermoscopy/generate_artifact_data.py --data_dir ../data --source_dataset ph2 --artifact_type dark-corner
python examples/dermoscopy/generate_artifact_data.py --data_dir ../data --source_dataset ph2 --artifact_type gel-bubble
python examples/dermoscopy/generate_artifact_data.py --data_dir ../data --source_dataset ph2 --artifact_type ink
python examples/dermoscopy/generate_artifact_data.py --data_dir ../data --source_dataset ph2 --artifact_type patches
python examples/dermoscopy/generate_artifact_data.py --data_dir ../data --source_dataset ph2 --artifact_type ruler
```

## 📊 Phase 3: Artifact Robustness (Table 4 & Heatmaps)

**Intent:** This phase measures the performance gap between clean evaluation data and artifact-injected evaluation data. By testing the previously trained Phase 1 models against the synthetic Phase 2 Trap Sets, we aim to quantify any potential drops in AUROC.

**Logging:** Logs are saved to `~/dermoscopy_logs/eval_artifacts`

### `run_all_eval_artifacts.sh`

(executed inside root directory of PyHealth repo)

```bash
#!/bin/bash
# =====================================================================
# Artifact Robustness Evaluation Pipeline
# 
# Loops through all trained models and evaluates them against 
# the synthetic out-of-distribution artifact datasets (Trap Sets).
# Logs are automatically saved by the Python script.
# =====================================================================

# Set your dataset directory
DATA_DIR="../data"

# Define the grids to loop over
TRAIN_SETS=("isic2018" "ph2")
MODELS=("convnext_tiny" "dinov2" "resnet50" "swin_t") 
MODES=("whole" "background" "bbox" "bbox70" "bbox90" "high_whole" "low_whole" "high_background" "low_background")
ARTIFACTS=("clean" "dark-corner" "gel-bubble" "ink" "patches" "ruler")

echo "====================================================================="
echo "[*] STARTING EVALUATIONS..."
echo "====================================================================="

for TRAIN_SET in "${TRAIN_SETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for MODE in "${MODES[@]}"; do
            # The path to trained model folds
            EXP_DIR=../dermoscopy_outputs/${TRAIN_SET}_${MODEL}_${MODE}
            
            # Safety check: Only evaluate if the training directory actually exists
            if [ ! -d "$EXP_DIR" ]; then
                echo "[!] Skipping ${TRAIN_SET} + ${MODEL} + ${MODE} (Directory not found)"
                continue
            fi

            for ARTIFACT in "${ARTIFACTS[@]}"; do
                python examples/dermoscopy/evaluate_artifact_robustness.py \
                    --exp_dir "$EXP_DIR" \
                    --strategy fold_average \
                    --data_dir "$DATA_DIR" \
                    --eval_dataset ph2 \
                    --artifact "$ARTIFACT" \
                    --mode "$MODE" \
                    --model "$MODEL"
                    
            done
        done
    done
done

echo "====================================================================="
echo "[*] ALL EVALUATIONS COMPLETE!"
echo "====================================================================="
```

Running `build_table4.py` outputs the Table 4 metrics found in the base paper, alongside advanced Seaborn Delta Heatmaps (saved to `~/heatmaps`) designed to visualize the exact robustness drop (`Δ AUROC = Artifact - Clean`).

**Logging:** Logs are saved to `~/dermoscopy_logs/aggregator`

### `build_table4.py`

(executed outside PyHealth repo in home directory)

```python
"""
Table 4 & Heatmap Aggregator.

Run this script AFTER your bash evaluation loops finish. It scans the generated 
log files, extracts the AUROC scores for each Artifact/Mode combination, 
prints the formatted Table 4 grid side-by-side, and generates Delta Seaborn heatmaps
showing the exact change in AUROC compared to the clean baseline.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from PyHealth.examples.dermoscopy.train_dermoscopy import setup_dynamic_logging

# The exact order the modes appear in the CHIL 2025 paper
MODE_ORDER = [
    "whole", "background", "bbox", "bbox70", "bbox90",
    "high_whole", "low_whole","high_background", "low_background"
]

def main(log_dir=None, model=None):
    if log_dir is None:
        print('[!] No log directory provided. Please specify the parent directory of the logs generated by your evaluation runs.')
        return

    # STRICT FILTER: Only grab Table 4 specific evaluation logs (ignore Experiment D)
    eval_files = [
        f for f in os.listdir(log_dir) 
        if f.lower().endswith(".txt") and 
        (f.lower().startswith("train_ph2_eval_ph2") or f.lower().startswith("train_isic2018_eval_ph2"))
    ]
    
    # Filter by model if requested
    if model:
        eval_files = [f for f in eval_files if model.lower() in f.lower()]

    if not eval_files:
        print(f"[!] No valid Table 4 evaluation logs found in {log_dir} for model {model.upper()}.")
        return

    # Structure to hold: results[model][train_set][mode][artifact] = roc_auc
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    print(f"[*] Parsing {len(eval_files)} evaluation logs for {model.upper()}...")
    for filename in eval_files:
        filepath = os.path.join(log_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract train set from the evaluation banner
            banner_match = re.search(r"EVALUATING:\s+Train\((.*?)\).*?Eval\((.*?)\)", content)
            
            # Extract variables directly from the Table 4 output format in the logs
            artifact_match = re.search(r"Artifact:\s+([^\n]+)", content)
            model_match = re.search(r"Model:\s+([^\n]+)", content)
            mode_match = re.search(r"Mode:\s+([^\n]+)", content)
            auc_match = re.search(r"AUROC:\s+([\d\.]+)", content)
            
            if banner_match and artifact_match and model_match and mode_match and auc_match:
                train_val = banner_match.group(1).strip().lower()
                artifact_val = artifact_match.group(1).strip().lower()
                model_val = model_match.group(1).strip().lower()
                mode_val = mode_match.group(1).strip().lower()
                auc_val = float(auc_match.group(1).strip())
                
                # Double-check model filtering since we extracted it from content
                if model and model.lower() not in model_val:
                    continue
                    
                results[model_val][train_val][mode_val][artifact_val] = auc_val

    if not results:
        print(f"[!] Could not extract any valid data from logs for {model}. Check format.")
        return

    os.makedirs("heatmaps", exist_ok=True)

    # Build the Tables and Heatmaps for each Model evaluated
    for model_name, model_data in results.items():
        
        # Gather and sort Train Sets (PH2 first, then ISIC to match paper)
        train_sets = sorted(list(model_data.keys()), key=lambda x: (x != 'ph2', x))
        display_train_sets = [ts.upper() for ts in train_sets]

        # Gather all modes across all train sets
        all_modes = set()
        for ts_data in model_data.values():
            all_modes.update(ts_data.keys())
            
        # Order modes based on CHIL paper if present, otherwise append
        modes = [m for m in MODE_ORDER if m in all_modes]
        for m in all_modes:
            if m not in modes:
                modes.append(m)

        # ---------------------------------------------------------
        # PRINT TEXT TABLE 4 (Side-by-Side View - Absolute AUROC)
        # ---------------------------------------------------------
        print("\n" + "="*80)
        print(f" TABLE 4: ARTIFACT ROBUSTNESS (AUROC) - MODEL: {model_name.upper()} ")
        print("="*80)
        
        header = f"{'Dataset Mode':<22} | {'Artifact':<15} | " + " | ".join([f"{ts:<10}" for ts in display_train_sets])
        print(header)
        print("-" * len(header))

        for mode in modes:
            # Collect all artifacts for this mode across all train sets
            mode_artifacts = set()
            for ts in train_sets:
                if mode in model_data[ts]:
                    mode_artifacts.update(model_data[ts][mode].keys())
            
            # Sort artifacts: 'clean' comes first, then alphabetically
            artifacts = sorted(list(mode_artifacts), key=lambda x: (x != 'clean', x))
            
            for i, artifact in enumerate(artifacts):
                display_mode = mode if i == 0 else ""
                # Map 'clean' to 'original' in the text output to exactly match the paper
                display_art = "original" if artifact == 'clean' else artifact
                
                row_str = f"{display_mode:<22} | {display_art:<15} | "
                row_vals = []
                for ts in train_sets:
                    val = model_data[ts].get(mode, {}).get(artifact, np.nan)
                    if not np.isnan(val):
                        row_vals.append(f"{val:<10.4f}")
                    else:
                        row_vals.append(f"{'N/A':<10}")
                row_str += " | ".join(row_vals)
                print(row_str)

            if mode != modes[-1]:
                print("-" * len(header))

        print("="*80)

        # ---------------------------------------------------------
        # GENERATE DELTA SEABORN HEATMAPS (Separated by Train Set)
        # ---------------------------------------------------------
        for train_set in train_sets:
            ts_data = model_data[train_set]
            
            plot_data = []
            
            # Get the set of artifacts specific to this training set
            ts_artifacts_found = set()
            for m_data in ts_data.values():
                ts_artifacts_found.update(m_data.keys())
                
            # For the Delta Heatmap, we exclude the 'clean' baseline itself
            heatmap_artifacts = sorted([a for a in ts_artifacts_found if a != 'clean'])
            
            for artifact in heatmap_artifacts:
                row_data = []
                for mode in modes:
                    # Get both the artifact value and the clean baseline value
                    val = ts_data.get(mode, {}).get(artifact, np.nan)
                    clean_val = ts_data.get(mode, {}).get('clean', np.nan)
                    
                    # Calculate Delta (Artifact - Clean)
                    if not np.isnan(val) and not np.isnan(clean_val):
                        delta = val - clean_val
                    else:
                        delta = np.nan
                        
                    row_data.append(delta)
                plot_data.append(row_data)
                
            plot_data_np = np.array(plot_data)
            
            # Only plot if we have valid numeric data
            if not np.isnan(plot_data_np).all() and len(plot_data_np) > 0:
                # Add extra padding to the bottom and left so the angled text fits
                plt.figure(figsize=(max(10, len(modes)*1.2), max(6, len(heatmap_artifacts)*0.8)))
                
                # 'RdBu' diverging colormap (Red = Negative/Drop, White = 0, Blue = Positive/Gain)
                # Force yticklabels to be completely lowercase
                sns.heatmap(plot_data_np, annot=True, fmt=".3f", cmap="RdBu", center=0,
                            xticklabels=modes, yticklabels=[a.lower() for a in heatmap_artifacts], 
                            cbar_kws={'label': 'Δ AUROC (Artifact - Clean)'})
                
                # Rotate both Y and X axis labels by 45 degrees so they don't squish
                plt.yticks(rotation=45)
                plt.xticks(rotation=45, ha='right') # ha='right' anchors the text cleanly
                
                plt.title(f"Robustness Drop (Δ AUROC) - Train: {train_set.upper()} | Model: {model_name.upper()}")
                plt.xlabel("Ablation Mode")
                plt.ylabel("Artifact Type")
                plt.tight_layout()
                
                # Updated filename to reflect that it's a delta map
                map_path = os.path.join("heatmaps", f"Table4_Delta_Heatmap_{train_set}_{model_name}.png")
                plt.savefig(map_path, dpi=300)
                plt.close()
                print(f"[*] Saved Delta Heatmap to {map_path}")

if __name__ == "__main__":
    # Initialize logging ONCE for all models
    parent_log_dir = os.path.expanduser("~/dermoscopy_logs")
    run_details = "table4_summary_all_models"
    logfilename = setup_dynamic_logging(parent_log_dir, "aggregator", run_details)
    
    # Resolve the log directory once
    base_log_dir = os.path.join(os.path.dirname(os.path.dirname(logfilename)), 'eval_artifacts')
    
    # Loop through models using the single log session
    model_list = ["dinov2", "convnext", "resnet50", "swin_t"]
    for target_model in model_list:
        main(log_dir=base_log_dir, model=target_model)
```

## ⏱ Phase 4: Epoch Ablation Study (Learning Dynamics)

**Intent:** To explore the learning dynamics of foundation models versus standard CNNs, we conduct an Epoch Ablation Study. Machine learning models may latch onto visual shortcuts before learning complex clinical features. By evaluating the Trap Sets at specific epoch milestones (3, 5, 10, 15, 20), we can observe the training progression and analyze at which stage different model architectures might diverge from baseline reasoning and become susceptible to artifact bias.

_(Note: The `run_epoch_ablation.py` script holds temporary weights directly in GPU memory to prevent saving redundant checkpoints. Temporary metrics log to `~/dermoscopy_logs/ablation_test`)_.

**Logging:** Logs are saved to `~/dermoscopy_logs/ablation_test`

### `run_epochs.sh`

(executed inside root directory of PyHealth repo)

```bash
#!/bin/bash
# =====================================================================
# Comprehensive Epoch Ablation Study
# Tests the hypothesis: Do Foundation Models resist learning 
# artifacts longer than standard CNNs across different artifact types?
# =====================================================================

DATA_DIR="../data"
MODELS=("resnet50" "swin_t" "dinov2" "convnext_tiny")

# All 5 diffusion-generated Trap Sets + Clean Baseline Control
ARTIFACTS=("clean" "ruler" "ink" "patches" "dark-corner" "gel-bubble")

echo "====================================================================="
echo "[*] STARTING COMPREHENSIVE EPOCH ABLATION EXPERIMENT"
echo "====================================================================="

for MODEL in "${MODELS[@]}"; do
    for ARTIFACT in "${ARTIFACTS[@]}"; do
        python examples/dermoscopy/run_epoch_ablation.py \
            --model "$MODEL" \
            --mode whole \
            --data_dir "$DATA_DIR" \
            --train_datasets isic2018 \
            --eval_dataset ph2 \
            --artifact "$ARTIFACT" \
            --epochs 20
            
    done
done

echo "====================================================================="
echo "[*] COMPREHENSIVE ABLATION EXPERIMENT COMPLETE!"
echo "====================================================================="
```

Running `plot_ablation_dynamics.py` generates chronological line graphs (saved to `~/ablation_plots`), allowing for a visual assessment of Trap Set AUROC over time.

**Logging:** Logs are saved to `~/dermoscopy_logs/aggregator`

### `plot_ablation_dynamics.py`

(executed outside PyHealth repo in home directory)

```python
"""
Learning Dynamics Line Plotter.

Scans the ablation_test log folder, parses the epoch-by-epoch AUROC scores,
and generates colored line graphs showing how different models
learn (or resist) diffusion-based artifacts over time.
"""

import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PyHealth.examples.dermoscopy.train_dermoscopy import setup_dynamic_logging

def parse_ablation_logs(log_dir):
    """Scans log files and extracts epoch ablation data into a Pandas DataFrame."""
    log_files = glob.glob(os.path.join(log_dir, "*.txt"))
    if not log_files:
        print(f"[!] No log files found in {log_dir}")
        return pd.DataFrame()

    records = []

    print(f"[*] Parsing {len(log_files)} ablation log files...")
    for filepath in log_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            # Find the summary block
            if "EPOCH ABLATION SUMMARY" not in content:
                continue

            model_match = re.search(r"Model:\s+([A-Za-z0-9_]+)", content)
            mode_match = re.search(r"Mode:\s+([A-Za-z0-9_]+)", content)
            # FIX: Use \S+ to match any non-whitespace characters (including hyphens!)
            eval_match = re.search(r"Eval:\s+(\S+)", content)

            if model_match and eval_match:
                model = model_match.group(1).lower()
                mode = mode_match.group(1).lower() if mode_match else "whole"
                eval_target = eval_match.group(1)

                # Extract the artifact name from the eval string (e.g., "ph2_with_ruler" -> "ruler")
                if "with" in eval_target:
                    artifact = eval_target.split("_with_")[1]
                else:
                    artifact = "clean"

                # Extract all Epoch/AUC pairs
                epoch_matches = re.findall(r"Epoch\s+(\d+):\s+([0-9\.]+)", content)
                for ep_str, auc_str in epoch_matches:
                    records.append({
                        "Model": model.upper(),
                        "Mode": mode,
                        "Artifact": artifact, # FIX: Removed .capitalize() to preserve exact names
                        "Epoch": int(ep_str),
                        "AUROC": float(auc_str)
                    })

    return pd.DataFrame(records)

def main():
    log_dir = os.path.expanduser("~/dermoscopy_logs/ablation_test")
    out_dir = "ablation_plots"
    os.makedirs(out_dir, exist_ok=True)

    df = parse_ablation_logs(log_dir)

    if df.empty:
        print("[!] No valid data extracted. Exiting.")
        return

    # Set Seaborn styling for publication-ready plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Hardcode a consistent color palette so lines match perfectly across all generated plots
    MODEL_COLORS = {
        "RESNET50": "#1f77b4",       # Blue
        "SWIN_T": "#ff7f0e",         # Orange
        "DINOV2": "#2ca02c",         # Green
        "CONVNEXT_TINY": "#d62728"   # Red
    }

    # Get the unique artifacts and modes
    artifacts = df['Artifact'].unique()
    modes = df['Mode'].unique()

    print("[*] Generating learning dynamic curves...")

    for artifact in artifacts:
        for mode in modes:
            subset = df[(df['Artifact'] == artifact) & (df['Mode'] == mode)]

            if subset.empty:
                continue

            plt.figure(figsize=(8, 5))

            # Draw the line plot
            sns.lineplot(
                data=subset, 
                x="Epoch", 
                y="AUROC", 
                hue="Model", 
                palette=MODEL_COLORS, # <-- Force the consistent colors here
                hue_order=["RESNET50", "SWIN_T", "DINOV2", "CONVNEXT_TINY"], # <-- Lock the legend order
                marker="o", 
                linewidth=2.5,
                markersize=8
            )

            # Dynamic titles and Y-axis labels for the Clean baseline
            if artifact == "clean":
                plt.title(f"Learning Dynamics: PH2 (Clean Baseline) | Mode: {mode}", fontsize=14, weight='bold')
                plt.ylabel("Clean Set AUROC", fontsize=12)
            else:
                plt.title(f"Learning Dynamics: PH2 + {artifact} | Mode: {mode}", fontsize=14, weight='bold')
                plt.ylabel("Trap Set AUROC", fontsize=12)

            plt.xlabel("Training Epoch", fontsize=12)

            # Force the X-axis to only show the exact milestone epochs we tested
            plt.xticks(sorted(subset['Epoch'].unique()))

            # Place the legend outside the plot so it doesn't cover the data lines
            plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

            # Save the plot
            safe_artifact_name = artifact.replace(" ", "_").replace("(", "").replace(")", "").lower()
            filename = os.path.join(out_dir, f"learning_dynamics_{safe_artifact_name}_mode_{mode}.png")
            plt.savefig(filename, dpi=300)
            plt.close()

            print(f"  -> Saved {filename}")

    # =========================================================
    # Print the Raw Table
    # =========================================================
    print("\n" + "="*80)
    print(" EPOCH ABLATION: TABULAR RESULTS ".center(80))
    print("="*80)

    # Create a clean MultiIndex pivot table with Epochs as columns
    pivot_df = df.pivot_table(
        index=['Model', 'Mode', 'Artifact'], 
        columns='Epoch', 
        values='AUROC'
    )

    # Convert to string and inject separator lines between models
    table_str = pivot_df.to_string(float_format="%.4f", na_rep="N/A")
    first_model_passed = False

    for line in table_str.split('\n'):
        # A new model row always starts at index 0 (no whitespace padding)
        # We ignore the literal strings "Model" and "Epoch" so we don't put lines through headers
        if len(line) > 0 and line[0] != ' ' and not line.startswith("Model") and not line.startswith("Epoch"):
            if first_model_passed:
                print("-" * 80)
            first_model_passed = True
        print(line)

    print("="*80 + "\n")
    print(f"[*] All plots generated successfully in the '{out_dir}' folder!")

if __name__ == "__main__":
    parent_log_dir = os.path.expanduser("~/dermoscopy_logs")
    run_details = "epoch_ablation_experiment_summary_all_artifacts"
    setup_dynamic_logging(parent_log_dir, "aggregator", run_details)

    main()
```

## 🧩 Phase 5: Methodological Analysis (Aggregation Strategies)

**Intent:** Finally, we investigate whether variations in metric aggregation strategies might influence the perceived robustness of a model. Moving past the 96 models trained in Phase 1, we focus on the 4 base architectures (ResNet50, Swin Tiny, DINOv2, ConvNeXT) and train them on a combined `isic2018 + ham10000` dataset. We specifically train each of the 4 models in two distinct configurations: as a **Global Master** (`--cv_folds 1`) and as a **5-Fold** split.

We then compare three distinct aggregation methodologies across the Trap Sets to analyze their impact on the final AUROC:

1. **Fold Averaging:** Taking the mean AUROC of 5 distinct CV folds.

2. **Probability Ensemble:** Soft voting (averaging the raw probabilities across all 5 folds before computing a single AUROC).

3. **Global Master:** Evaluating the single model trained on 100% of the combined data with no validation holdout.

**Weights:** Weights are saved to `~/dermoscopy_outputs` in the specific model folders.

**Logging:** Logs are saved to `~/dermoscopy_logs/train` and `~/dermoscopy_logs/eval_artifacts`

### `run_strategies.sh`

(executed inside root directory of PyHealth repo)

```bash
#!/bin/bash
# =====================================================================
# Experiment: Methodological Analysis
# Tests probability ensemble vs. fold averaging vs. global master
# Training Data: ISIC2018 + HAM10000
# Target Data: PH2 (Clean + Artifacts)
# =====================================================================

DATA_DIR="../data"
MODELS=("resnet50" "swin_t" "dinov2" "convnext_tiny")
ARTIFACTS=("clean" "ruler" "ink" "patches" "dark-corner" "gel-bubble")
STRATEGIES=("master" "ensemble" "fold_average")
MODE="whole"

echo "====================================================================="
echo "[*] STARTING METHODOLOGICAL ANALYSIS EXPERIMENT"
echo "====================================================================="
# ---------------------------------------------------------------------
# PHASE 1: TRAIN THE MODELS (Master & 5-Fold)
# (Safely skips if weights already exist in dermoscopy_outputs)
# ---------------------------------------------------------------------
echo ""
echo "[*] PHASE 1: TRAINING MASTER AND 5-FOLD MODELS"

for MODEL in "${MODELS[@]}"; do
    echo "---------------------------------------------------------------------"
    echo "[*] Ensuring ${MODEL} is trained (Master - 1 Fold)..."
    python examples/dermoscopy/train_dermoscopy.py \
        --model "$MODEL" \
        --mode "$MODE" \
        --data_dir "$DATA_DIR" \
        --epochs 10 \
        --train_datasets isic2018 ham10000 \
        --cv_folds 1

    echo "[*] Ensuring ${MODEL} is trained (5-Fold CV)..."
    python examples/dermoscopy/train_dermoscopy.py \
        --model "$MODEL" \
        --mode "$MODE" \
        --data_dir "$DATA_DIR" \
        --epochs 10 \
        --train_datasets isic2018 ham10000 \
        --cv_folds 5
done

# ---------------------------------------------------------------------
# PHASE 2: EVALUATE STRATEGIES ACROSS TRAP SETS
# ---------------------------------------------------------------------
echo ""
echo "[*] PHASE 2: EVALUATING STRATEGIES ACROSS TRAP SETS"

for MODEL in "${MODELS[@]}"; do
    # Resolves to: ~/dermoscopy_outputs/isic2018_ham10000_resnet50_whole
    EXP_DIR=../dermoscopy_outputs/isic2018_ham10000_${MODEL}_${MODE}

    for ARTIFACT in "${ARTIFACTS[@]}"; do
        for STRATEGY in "${STRATEGIES[@]}"; do
            # evaluate_artifact_robustness.py now prints its own banner, 
            # so we just let Python handle the logging!
            python examples/dermoscopy/evaluate_artifact_robustness.py \
                --exp_dir "$EXP_DIR" \
                --strategy "$STRATEGY" \
                --data_dir "$DATA_DIR" \
                --eval_dataset ph2 \
                --artifact "$ARTIFACT" \
                --mode "$MODE" \
                --model "$MODEL"

        done
    done
done

echo "====================================================================="
echo "[*] EXPERIMENT FULLY COMPLETE!"
echo "====================================================================="
```

Running `plot_methodologies.py` parses these logs and outputs grouped bar charts to `~/methodology_plots`, providing a comparative visualization to analyze how different metric aggregation strategies perform under domain shifts.

**Logging:** Logs are saved to `~/dermoscopy_logs/aggregator`

### `plot_methodologies.py`

(executed outside PyHealth repo in home directory)

```python
"""
Methodologies Experiment Results Visualizer.

Scans the eval_artifacts log folder, filters for models trained on 
the combined ISIC2018+HAM10000 dataset, and generates grouped bar charts
comparing Master, Fold Average, and Ensemble strategies across trap sets.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PyHealth.examples.dermoscopy.train_dermoscopy import setup_dynamic_logging

def parse_exp_logs(log_dir):
    """Scans evaluation logs and extracts methodology comparisons."""
    # STRICT FILTER: Only grab logs trained on the combined experiment dataset
    log_files = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir) 
        if f.lower().endswith(".txt") and "isic2018_ham10000" in f.lower()
    ]
    
    if not log_files:
        print(f"[!] No valid Methodologies Experiment log files found in {log_dir}")
        return pd.DataFrame()

    records = []
    
    print(f"[*] Parsing {len(log_files)} Methodologies Experiment evaluation log files...")
    for filepath in log_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Ensure this is an evaluation log
            if "EVALUATING:" not in content or "RESULTS" not in content:
                continue
                
            # Filter specifically for this Methodologies Experiment training (ISIC2018 + HAM10000)
            train_match = re.search(r"Train\((.*?)\)", content)
            if not train_match or train_match.group(1).lower() != "isic2018_ham10000":
                continue 

            model_match = re.search(r"Model:\s+([A-Za-z0-9_]+)", content)
            mode_match = re.search(r"Mode:\s+([A-Za-z0-9_]+)", content)
            artifact_match = re.search(r"Artifact:\s+([A-Za-z0-9_\-]+)", content)
            auroc_match = re.search(r"AUROC:\s+([0-9\.]+)", content)
            
            # evaluate_artifact_robustness.py only prints 'Strategy' for ensemble/master
            strategy_match = re.search(r"Strategy:\s+([A-Za-z0-9_]+)", content)

            if model_match and artifact_match and auroc_match:
                raw_strategy = strategy_match.group(1) if strategy_match else "fold_average"
                raw_mode = mode_match.group(1).lower() if mode_match else "whole"
                
                records.append({
                    "Model": model_match.group(1).upper(),
                    "Mode": raw_mode,
                    "Artifact": artifact_match.group(1),
                    "Strategy": raw_strategy.replace("_", " ").title(),
                    "AUROC": float(auroc_match.group(1))
                })

    return pd.DataFrame(records)

def main():
    log_dir = os.path.expanduser("~/dermoscopy_logs/eval_artifacts")
    out_dir = "methodology_plots"
    os.makedirs(out_dir, exist_ok=True)
    
    df = parse_exp_logs(log_dir)
    
    if df.empty:
        print("[!] No valid Methodologies Experiment data found in logs. Check if the bash script finished!")
        return

    # Set Seaborn styling for publication-ready plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Enforce logical ordering of the bars
    strategy_order = ["Master", "Fold Average", "Ensemble"]
    models = df['Model'].unique()
    modes = df['Mode'].unique()
    
    print(f"[*] Generating comparative bar charts for {len(models)} models across {len(modes)} modes...")
    
    for model in models:
        for mode in modes:
            subset = df[(df['Model'] == model) & (df['Mode'] == mode)]
            
            if subset.empty:
                continue
            
            plt.figure(figsize=(10, 6))
            
            # Draw the Grouped Bar Chart
            ax = sns.barplot(
                data=subset, 
                x="Artifact", 
                y="AUROC", 
                hue="Strategy",
                hue_order=[s for s in strategy_order if s in subset['Strategy'].values],
                palette="viridis"
            )
            
            plt.title(f"Methodology: {model} | Mode: {mode}", fontsize=14, weight='bold')
            plt.xlabel("Evaluation Target (PH2 + Artifact)", fontsize=12)
            plt.ylabel("AUROC", fontsize=12)
            
            # Crop the Y-axis so the differences between strategies are actually visible
            plt.ylim(0.4, 1.0)
            
            # Place the legend safely outside
            plt.legend(title="Aggregation Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            filename = os.path.join(out_dir, f"methodology_{model.lower()}_mode_{mode}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            
            print(f"  -> Saved {filename}")

    # =========================================================
    # Print the Raw Table for the Paper
    # =========================================================
    print("\n" + "="*80)
    print(" METHODOLOGIES EXPERIMENT TABULAR RESULTS ".center(80))
    print("="*80)
    
    # Create a clean MultiIndex pivot table 
    pivot_df = df.pivot_table(
        index=['Model', 'Mode', 'Artifact'], 
        columns='Strategy', 
        values='AUROC'
    )
    
    # Reorder columns to match the chart
    valid_cols = [c for c in strategy_order if c in pivot_df.columns]
    pivot_df = pivot_df[valid_cols]
    
    # Convert to string and inject separator lines between models
    table_str = pivot_df.to_string(float_format="%.4f", na_rep="N/A")
    first_model_passed = False
    
    for line in table_str.split('\n'):
        # A new model row always starts at index 0 (no whitespace padding)
        # We ignore the literal string "Model" so we don't put a line through the headers
        if len(line) > 0 and line[0] != ' ' and not line.startswith("Model"):
            if first_model_passed:
                print("-" * 80)
            first_model_passed = True
        print(line)

    print("="*80 + "\n")
    print(f"[*] All comparative plots saved to '{out_dir}'!")

if __name__ == "__main__":
    parent_log_dir = os.path.expanduser("~/dermoscopy_logs")
    run_details = "methodologies_experiment_summary_all_models"
    setup_dynamic_logging(parent_log_dir, "aggregator", run_details)

    main()
```
