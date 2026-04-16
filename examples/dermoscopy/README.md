# Dermoscopy Artifact Robustness & Interpretability
**Official PyHealth 2.0 Implementation for CHIL 2025**

> **Paper:** *A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations*

This repository contains the full execution pipeline for training, evaluating, and interpreting melanoma classification models under synthetic artifact shifts. The architecture is fully integrated into the [PyHealth 2.0](https://github.com/sunlabuiuc/PyHealth) ecosystem, utilizing native `DermoscopyDataset`, `Task`, and `Processor` schemas.

---

## 📁 Repository Manifest

### Core Pipeline Scripts
* **`train_dermoscopy.py`**: The master training engine. Executes 5-fold cross-validation or single master model training. Handles out-of-domain transfer learning and dynamically saves model checkpoints.
* **`evaluate_artifact_robustness.py`**: The "Smart Evaluator." Bypasses standard evaluators to allow raw probability ensembling and strict metric computation (ROC-AUC, PR-AUC, F1, etc.) on out-of-distribution "Trap Sets."
* **`generate_artifact_data.py`**: The artifact generator. Uses Stable Diffusion Inpainting (LoRA) to synthetically inject clinical artifacts (rulers, ink, gel bubbles) into existing datasets.
* **`run_tcav.py`**: Interpretability engine. Uses Testing with Concept Activation Vectors (TCAV) to calculate how much human-understandable concepts influence latent space representations.
* **`run_epoch_ablation.py`**: Learning dynamics study. Halts training at periodic epochs to evaluate robustness against Trap Sets, preventing data leakage.

---

## 🚀 Execution Pipeline

### Phase 0: Trap Set Generation
Generate the synthetic artifact datasets using Stable Diffusion before running evaluations.
```bash
python generate_artifact_data.py \
    --data_dir /path/to/data \
	--log_dir ~/dermoscopy_logs \
    --dataset ph2 \
    --lora_path /path/to/lora \
    --artifact ruler
```
*Supported artifacts: `ruler`, `ink`, `dark-corner`, `gel-bubble`, `patches`.*

### Phase 1: Model Training & Baselines
Train models across multiple architectures (ResNet50, Swin, DINOv2) and ablation modes. To ensure a fair comparison of architectural robustness, all models were fine-tuned using a standardized learning rate of 1e-4.
```bash
python train_dermoscopy.py \
    --data_dir /path/to/data \
    --out_dir ~/dermoscopy_outputs \
	--log_dir ~/dermoscopy_logs \
    --train_datasets isic2018 \
    --model resnet50 \
    --mode whole \
    --epochs 10 \
    --cv_folds 5 \
    --lr 1e-4
```

### Phase 2: Artifact Robustness Evaluation
Evaluate the trained models against the synthetic Trap Sets generated in Phase 0.
```bash
python evaluate_artifact_robustness.py \
    --exp_dir ~/dermoscopy_outputs/isic2018_resnet50_whole \
	--log_dir ~/dermoscopy_logs \
    --strategy fold_average \
    --data_dir /path/to/data \
    --eval_dataset ph2 \
    --artifact ruler \
    --mode whole \
    --model resnet50
```

### Phase 3: Interpretability (TCAV)
Extract latent representations and train concept activation vectors to understand model bias.
```bash
python run_tcav.py \
    --exp_dir ~/dermoscopy_outputs/isic2018_resnet50_whole \
	--log_dir ~/dermoscopy_logs \
    --data_dir /path/to/data \
    --train_datasets isic2018 \
    --eval_dataset ph2 \
    --artifact ruler \
    --model resnet50
```

---

## 🗂 Logging & File Management

To maintain repository hygiene, all heavy experimental data and logs are explicitly routed **outside** of the PyHealth repository.

* **`~/dermoscopy_outputs/`**: Centralized, persistent storage for heavy `.pth` model weights, learning curves, and PyHealth's internal epoch tracking data. Inside each specific experiment folder (e.g., `isic2018_resnet50_whole`), you will find:
  * **`fold_X/`**: Contains the specific weights and evaluation logs for individual folds when running cross-validation (e.g., `--cv_folds 5`).
  * **`master/`**: Contains the final model weights trained on the entire dataset when cross-validation is bypassed (e.g., by setting `--cv_folds 1`). The Phase 2 and Phase 3 evaluation scripts will automatically default to this folder if a `fold_0` is not present.
* **`~/dermoscopy_logs/`**: Contains time-stamped, permanent text records of every terminal output across all phases.

---

## 📚 Acknowledgements & Citations
* **PyHealth:** Sun, J., et al. (2022). *PyHealth: A Deep Learning Toolkit for Healthcare Predictive Modeling.*
* **TCAV:** Kim, B., et al. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* ICML.
* **Stable Diffusion:** Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.