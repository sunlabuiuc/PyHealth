# PTB-XL ResNet-18 learning-rate comparison

## Setup

- Dataset: PTB-XL 1.0.3, 500 Hz.
- Scope: PyHealth dev mode; 1,000 patients and 1,341 samples.
- Official-fold split: 1,074 train, 127 validation, 140 test.
- Normalization: per-lead mean and standard deviation computed on training data.
- Model: 1D ResNet-18 with a multilabel prediction head.
- Windowing: random 2.5-second training segment; overlapping 2.5-second evaluation windows with 50% overlap; maximum probability across windows for each label.
- Training: 5 epochs, batch size 16, Adam, learning rate 0.001, seed 0.
- Checkpoint selection: best validation macro-AUROC.

## Results

| Schedule | Best validation macro-AUROC | Test loss | Test macro-AUROC | Training time |
|---|---:|---:|---:|---:|
| Constant learning rate | 0.8524 (epoch 5) | 0.4468 | 0.8419 | 4.5 min |
| Cosine decay | 0.8774 (epoch 5) | 0.4400 | 0.8483 | 5.0 min |

## Interpretation and limitations

Cosine decay was higher by 0.0064 test macro-AUROC in this single dev-mode run. This is preliminary and does not establish a reliable performance difference.

Nonaka and Seita report 0.933 ± 0.006 for ResNet-18 on PTB-XL super-diagnosis, averaged across five trials. This run used a 1,000-patient dev subset, five epochs, and one seed, so its score is not a direct reproduction of that result.

## Run

Real-data comparison:

    python examples/ecg/ptbxl/ptbxl_superclass_classification_resnet18.py --dev --epochs 5 --seed 0

Synthetic smoke test:

    python examples/ecg/ptbxl/ptbxl_superclass_classification_resnet18.py --synthetic --epochs 1 --batch-size 5

Synthetic scores are only pipeline checks, not ECG results.

Paper: [Nonaka and Seita (2021)](https://proceedings.mlr.press/v149/nonaka21a.html)
