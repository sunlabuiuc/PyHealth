# MedMod in-hospital mortality reproduction results

This page reports our results for the two models in the MedMod example
scripts in this folder. The models were run on real MIMIC-IV v1.0 and
MIMIC-CXR-JPG 2.0.0 data, and the results are compared with the paper below.

> Elsharief, S., et al. (2025). MedMod: Multimodal benchmark for medical
> prediction tasks with electronic health records and chest X-ray scans.
> CHIL 2025. https://proceedings.mlr.press/v287/elsharief25a.html

The two scripts are:

- `mimic4_medmod_ihm_early_fusion.py` trains Early Fusion. It uses MedFuse's
  released pretrained encoders, the EHR LSTM `uni_ehr_all` and the ResNet-34
  `cxr_rad_full`. Both are frozen, so they stay exactly as released. Only the
  projection layer and the classifier are trained.
- `mimic4_medmod_convirt.py` pretrains ConVIRT with a contrastive loss and then
  trains a linear probe on the frozen encoders.

Our extension replaces the ResNet-34 image encoder with a frozen MedSigLIP
image encoder in both models (option `--embeddings`).

All numbers are measured on MedMod's fixed test split, which has 1,241 stays
and 187 deaths. This page reports only aggregate scores and no patient-level
data.

The Google Colab notebook `mimic4_medmod_ihm_colab_runs.ipynb` goes with this
page. It holds our real-data runs, with the output each run printed. Both
scripts also run end to end on synthetic data, which needs no MIMIC access
(see Reproduce).

The five-seed results come from the runner scripts in `medmod_runners/`.
These runners are part of this pull request. They train the same models with
the same settings as the example scripts. They also compute bootstrap
confidence intervals and write a report for every run. Each runner saves
checkpoints and continues from the last one after a disconnect. The Early
Fusion and probe runners save every 10 epochs, and the ConVIRT pretraining
runner saves after every epoch. Each example script was also run once on real
data, to check that it gives the same scores as its runner.

The Colab notebook `mimic4_medmod_ihm_colab_runs.ipynb` shows all runs. Its
cell "Summary of the runner reports" prints the score of every seed, the means
and the differences. The cell "Result line of every job" prints the last line
of every run, including the example scripts. The sections below name the cell
that shows each number.

## Our hypotheses

Our project proposal stated two hypotheses. The sections below give the
details.

**H1. Our PyHealth models reach a test AUROC close to the paper's.**

Early Fusion confirms H1. Its mean AUROC over five seeds is 0.8407 ± 0.0004,
and the paper reports 0.842 ± 0.004.

The ConVIRT linear probe does not confirm H1 for the target in our proposal,
Table 4's 0.847 ± 0.002 for linear probing. Its mean AUROC over five seeds is
0.8206 ± 0.0049, below 0.847 with every seed. It is close to Table 8's
0.813 ± 0.010 for linear evaluation. Our proposal chose Table 4 because Table
8 seems to swap the linear-probing and fine-tuning values. MedMod's released
linear-evaluation script also trains the encoders, which may explain the
higher 0.847 (see ConVIRT linear probe below).

**H2. Replacing the ResNet-34 image encoder with frozen MedSigLIP changes the
test scores of both models.**

For Early Fusion, MedSigLIP gives a small but consistent gain. Its AUROC is
higher with each of the five seeds, by 0.0054 on average. The 95% intervals of
the two encoders still overlap with every seed.

For the ConVIRT linear probe, MedSigLIP does not change the AUROC. The mean
difference is -0.0006 ± 0.0049, and MedSigLIP is higher with 3 of the 5 seeds.
The AUPRC is higher by 0.0098 ± 0.0139 on average, with 4 of the 5 seeds.

## Early Fusion

Each score is the mean ± sample standard deviation over five runs. The runs
use MedMod's five seeds 1002, 2918, 5793, 7261 and 84305. The paper also
averages five runs, and the seed list comes from MedMod's code.

The five runs used the runner `medmod_runners/early_fusion_run.py`. With
seed 0, the example script and the runner give the same scores, AUROC 0.8408
and AUPRC 0.5137 (Colab cell "Result line of every job").

| Image encoder | AUROC | AUPRC |
|---|---|---|
| ResNet-34 (MedFuse `cxr_rad_full`), this reproduction | **0.8407 ± 0.0004** | **0.5140 ± 0.0007** |
| Paper | 0.842 ± 0.004 | 0.515 ± 0.020 |
| MedSigLIP (extension) | **0.8461 ± 0.0010** | **0.5367 ± 0.0035** |

Across the five seeds, AUROC ranges from 0.8401 to 0.8410 with the ResNet-34
and from 0.8447 to 0.8474 with MedSigLIP. MedSigLIP scores higher than the
ResNet-34 with every seed. The average difference per seed is +0.0054 ± 0.0009
in AUROC and +0.0227 ± 0.0032 in AUPRC.

This gain is small compared with the uncertainty that comes from the size of
the test set. For example, with seed 1002 the 95% bootstrap interval of the
AUROC is 0.8105 to 0.8678 for the ResNet-34 and 0.8180 to 0.8701 for
MedSigLIP. The two intervals overlap with every seed. So the gain is
consistent across seeds but small.

Our scores vary much less between seeds than the paper's. A likely reason is
that our frozen encoders stay in evaluation mode, so they stay exactly as
released. MedMod's code leaves them in training mode (see Deviations and
caveats).

## ConVIRT linear probe

Each score is the mean ± sample standard deviation over MedMod's five seeds.
Each seed has its own pretraining run and its own probe. As the paper
describes, each probe uses the pretraining epoch with the highest validation
AUROC (see Pretraining below). The test score is taken at the last probe
epoch, as in MedMod's code. The runs used the runners
`medmod_runners/convirt_pretrain.py` and `medmod_runners/convirt_probe.py`.

| Image encoder | AUROC | AUPRC |
|---|---|---|
| ResNet-34, this reproduction | **0.8206 ± 0.0049** | **0.4588 ± 0.0114** |
| Paper, Table 4 (linear probing) | 0.847 ± 0.002 | 0.482 ± 0.016 |
| Paper, Table 8 (linear evaluation) | 0.813 ± 0.010 | |
| MedSigLIP (extension) | **0.8200 ± 0.0037** | **0.4686 ± 0.0077** |

Across the five seeds, AUROC ranges from 0.8134 to 0.8255 with the ResNet-34
and from 0.8148 to 0.8252 with MedSigLIP. The paper's Table 4 value (0.847) is
above every seed's score.

The 95% bootstrap intervals use 1,000 resamples of the test set. With seed
84305, they are AUROC 0.7874 to 0.8512 for the ResNet-34 and 0.7831 to 0.8468
for MedSigLIP. The intervals of the two encoders overlap with every seed.

The example script `mimic4_medmod_convirt.py` ran once with seed 84305, on the
same pretraining epoch as the runner. With the ResNet-34, it gives the same
scores as the runner, AUROC 0.8207 and AUPRC 0.4685. With MedSigLIP, it gives
AUROC 0.8159 and AUPRC 0.4640, and the runner gives 0.8148 and 0.4589. The
runner shuffles the MedSigLIP training batches in a different order.

The paper's tables disagree about which value belongs to the linear probe.
Table 4 gives 0.847 for linear probing and 0.813 ± 0.010 (AUPRC 0.449 ± 0.018)
for fine-tuning. Table 8 gives 0.847 for fine-tuning and 0.813 for linear
evaluation, with 641 trainable parameters. Our ResNet-34 probe also has 641
trainable parameters.

MedMod's released linear-evaluation script for mortality
(`src/scripts/lineareval/mortality/multimodal.sh`) passes the option
`--finetune`. With this option the encoders process the inputs
(`forward_fused` in `src/models/fusion_models.py`), and all parameters except
the projection heads are trained (`freeze_weights` and `configure_optimizers`
in `src/models/selfsupervised/models/finetune.py`). So the 0.847 may come from
fine-tuning and not from a frozen linear probe. We report the frozen probe,
which is what Table 4 describes.

With MedSigLIP, the probe's 1,152 image features are the fixed MedSigLIP
vectors, and pretraining changes only the 128 EHR features. So the MedSigLIP
probe result does not show the effect of contrastive pretraining by itself.

### Pretraining (300 epochs, MedMod's settings)

Every pretraining run keeps the weights of every 10th epoch (9, 19, ..., 299).
For each of these 30 epochs, a quick probe runs with the image features
computed once and no augmentation. The epoch whose quick probe reaches the
highest validation AUROC is chosen, and the normal probe then runs on it. The
test set is not used for this choice.

The table comes from the Colab cells "Pretraining summary, both encoders and
all seeds" and "Summary of the runner reports".

| Image encoder | Seed | Epoch with the lowest validation loss (loss) | Validation loss at the last epoch, 299 | Epoch chosen for the probe |
|---|---|---|---|---|
| ResNet-34 | 1002 | 135 (8.89) | 13.48 | 279 |
| ResNet-34 | 2918 | 110 (9.03) | 15.20 | 209 |
| ResNet-34 | 5793 | 125 (8.98) | 13.42 | 269 |
| ResNet-34 | 7261 | 127 (8.95) | 14.59 | 259 |
| ResNet-34 | 84305 | 138 (8.84) | 12.80 | 189 |
| MedSigLIP | 1002 | 17 (8.68) | 47.20 | 139 |
| MedSigLIP | 2918 | 27 (8.64) | 43.55 | 249 |
| MedSigLIP | 5793 | 21 (8.79) | 46.32 | 219 |
| MedSigLIP | 7261 | 21 (8.73) | 47.86 | 179 |
| MedSigLIP | 84305 | 15 (8.74) | 47.98 | 199 |

Both encoders overfit the contrastive loss. The validation loss is lowest
early and then rises. A model that guesses at random would get about 10.9 on
these validation batches. MedMod's early stopping (patience 30) would stop 30
epochs after the lowest validation loss. Still, the chosen epochs come much
later than the lowest validation loss. So for the probe, a low contrastive
validation loss does not mean the best features.

MedMod's loss leaves the positive pair out of the denominator. For this reason
the training loss can go below zero.

## Settings

- Cohort. Early Fusion uses MedMod's fixed split with 4,485 training, 488
  validation and 1,241 test stays. The paper and our proposal have 1,242 test
  stays. One test image is missing from PhysioNet, so we drop that stay.
  ConVIRT pretraining uses MedMod's phenotyping stays, 7,755 for training and
  882 for validation, each with the EHR of the whole ICU stay, as in MedMod's
  code. One training image is missing there too.
- EHR. We use 1-hour bins and MedMod's phenotyping normalizer, as in the
  released code. The paper text says 2-hour bins and a normalizer fitted on the
  dataset. Our proposal also said 2-hour bins. We follow the code because
  MedFuse's released EHR encoder, which Early Fusion uses, matches the test
  AUROC MedFuse reports for it (0.825, MedFuse Table 1) only with the code's
  settings. In our check on our test split, the frozen encoder alone gave
  0.833 with the code's settings and 0.719 with the paper's.
- Early Fusion. The encoders are MedFuse's released `uni_ehr_all` LSTM and
  `cxr_rad_full` ResNet-34, both frozen. Training uses Adam with learning rate
  9.515e-05 and betas (0.9, 0.9), batch size 16, up to 50 epochs, and
  ReduceLROnPlateau with factor 0.5 and patience 10. The best epoch is chosen
  by validation AUROC. Training stops after 15 epochs without improvement,
  which is MedFuse's default. The paper text says 10 epochs.
- ConVIRT pretraining. LARS with learning rate 0.6044, batch size 256,
  temperature 0.01, a 10-epoch warmup followed by cosine restarts, and 300
  epochs, as in MedMod's `simclr.sh`.
- ConVIRT linear probe. The probe uses the pretraining epoch with the highest
  validation AUROC, as the paper describes. MedMod's linear-evaluation script
  uses pretraining epoch 65 instead. The probe is trained with AdamW, learning
  rate 0.001, batch size 256, for up to 200 epochs, and stops after 30 epochs
  without improvement in validation AUROC. These settings come from MedMod's
  linear-evaluation script. The paper text instead describes Adam, up to 300
  epochs, stopping after 10 epochs and a learning-rate search.
- MedSigLIP (Sellergren et al., 2026, see References). We use the vision
  tower of `google/medsiglip-448` (revision `9cea28a1`), frozen, with its
  pooled output and its own 448 × 448 image processor. The embeddings are
  computed once for all 12,625 resized images (the Colab cell "Check the
  MedSigLIP embeddings file"). There is no image augmentation.
- Confidence intervals. We use 1,000 bootstrap resamples of the test set and
  the percentile method. MedFuse's code uses the basic method. On the same
  resamples, the basic method would move the interval ends of our runs by
  0.016 or less.

## Deviations and caveats

- Our project proposal described Early Fusion with both encoders trained from
  scratch. The paper's Early Fusion, which is the model we proposed to
  reproduce (0.842 ± 0.004), uses pretrained encoders instead. Section 3.3.1
  says it pretrains the two encoders separately, and Table 3 lists it under
  "Multimodal (Pretrained)". MedMod's and MedFuse's `fusion_trainer.py` also
  freeze both encoders for Early Fusion. We follow the paper. One sentence in
  the caption of the paper's Figure 1 says Early Fusion uses "randomly
  initialized encoders". This conflicts with the rest of the paper.
- Early Fusion uses MedFuse's released encoder checkpoints. MedMod pretrains
  its own encoders, the EHR LSTM on mortality and the ResNet-34 on the
  radiology labels, but does not release them. MedFuse is from the same lab,
  and its encoders were trained with the same patient split. So no test
  patient is in the encoders' training data.
- Our frozen encoders stay exactly as released during training. This is the
  usual practice for frozen layers in PyTorch (evaluation mode). MedMod's and
  MedFuse's code leave the frozen encoders in training mode. There, the
  ResNet's BatchNorm statistics still change during training, and dropout
  still acts on the EHR features. This may explain part of the larger spread
  between seeds in the paper.
- Our proposal planned to train with PyHealth's `Trainer`. The example
  scripts use their own short training loops instead, because the `Trainer`
  has no learning-rate scheduler. MedFuse's settings need ReduceLROnPlateau,
  and MedMod's need LARS with a warmup and cosine restarts.
- We use MedFuse's released learning rate. We did not repeat the paper's
  learning-rate search, which used 100 runs for mortality.
- ConVIRT pretraining runs all 300 epochs. MedMod's code stops it when the
  validation loss has not improved for 30 epochs. That rule would have stopped
  every run before its chosen epoch (see Pretraining above).
- To choose the pretraining epoch, we compare every 10th epoch, not every
  epoch, and the quick probe for this choice runs without augmentation. This
  keeps the cost down. One 300-epoch ResNet-34 pretraining run took about 6
  hours on Colab.
- During our run, our Google Drive filled up, so some ResNet-34 pretraining
  checkpoints were not uploaded to it. Before the probes ran, we copied them
  back to their folders from Colab's local Drive cache, where they had been
  kept. Where a checkpoint was in both places, the two copies were identical.
- MedSigLIP's training data include MIMIC-CXR. So MedSigLIP may have seen
  images of our test patients during its own training.

## Reproduce

The Colab notebook `mimic4_medmod_ihm_colab_runs.ipynb` shows every real-data
run, including the runner commands.

Both scripts run on synthetic data with `--synthetic`. The MedSigLIP extension
also runs on synthetic data with `--synthetic ... --embeddings synthetic`.
This option uses random stand-in vectors, so the gated MedSigLIP model is not
needed. The scripts' docstrings have the details. The commands for real data
are:

```bash
# Early Fusion, one seed (repeat with --seed 1002, 2918, 5793, 7261, 84305)
python examples/mimic4_medmod_ihm_early_fusion.py \
    --ehr-root /data/mimiciv/1.0 --cxr-root /data/mimic-cxr-jpg/2.0.0 \
    --image-dir /data/mimic-cxr-jpg/2.0.0/resized --cache-dir /data/cache \
    --ehr-weights checkpoints/mortality/uni_ehr_all/best_checkpoint.pth.tar \
    --cxr-weights checkpoints/cxr_rad_full/best_checkpoint.pth.tar --seed 1002
# For MedSigLIP, pass --embeddings medsiglip_448.npz instead of --cxr-weights

# ConVIRT pretraining, then the probe on the chosen epoch NNN
python examples/mimic4_medmod_convirt.py pretrain --extracted-root ... --out runs/convirt
python examples/mimic4_medmod_convirt.py probe ... --checkpoint runs/convirt/epoch_NNN.pt
# For MedSigLIP, add --embeddings medsiglip_448.npz to both steps
# The Colab notebook chooses NNN with quick probes of the runner
# medmod_runners/convirt_probe.py (option --no-augment)

# Compute the MedSigLIP embeddings
python -m pyhealth.datasets.medmod.medsiglip --help
```

## References

Paper we reproduce:

- Elsharief, S., Shurrab, S., Al Jorf, B., Lechuga López, L. J., Geras, K. J.,
  and Shamout, F. E. (2025). MedMod: Multimodal benchmark for medical
  prediction tasks with electronic health records and chest X-ray scans.
  Proceedings of the Conference on Health, Inference, and Learning (CHIL),
  PMLR 287.
  https://proceedings.mlr.press/v287/elsharief25a.html
  Code: https://github.com/nyuad-cai/MedMod

Methods and models we use:

- Hayat, N., Geras, K. J., and Shamout, F. E. (2022). MedFuse: Multi-modal
  fusion with clinical time-series data and chest X-ray images. Machine
  Learning for Healthcare (MLHC). Code and encoder checkpoints:
  https://github.com/nyuad-cai/MedFuse
- Zhang, Y., Jiang, H., Miura, Y., Manning, C. D., and Langlotz, C. P. (2022).
  Contrastive learning of medical visual representations from paired images
  and text. Machine Learning for Healthcare (MLHC).
- Sellergren, A., et al. (2026). MedGemma Technical Report. Nature Medicine
  32, 3695–3704. https://doi.org/10.1038/s41591-026-04626-w (preprint
  arXiv:2507.05201). This report introduces MedSigLIP. The model
  `google/medsiglip-448` (https://huggingface.co/google/medsiglip-448) is used
  under the Health AI Developer Foundations terms of use,
  https://developers.google.com/health-ai-developer-foundations/terms

Data:

- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., and
  Mark, R. (2021). MIMIC-IV (version 1.0). PhysioNet.
  https://doi.org/10.13026/s6n6-xd98
- Johnson, A., Lungren, M., Peng, Y., Lu, Z., Mark, R., Berkowitz, S., and
  Horng, S. (2019). MIMIC-CXR-JPG - chest radiographs with structured labels
  (version 2.0.0). PhysioNet. https://doi.org/10.13026/8360-t248
- Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R.,
  Lungren, M. P., Deng, C., Mark, R. G., and Horng, S. (2019). MIMIC-CXR: A
  large publicly available database of labeled chest radiographs.
  arXiv:1901.07042. https://arxiv.org/abs/1901.07042
- Johnson, A., Pollard, T., Mark, R., Berkowitz, S., and Horng, S. (2019).
  MIMIC-CXR Database (version 2.0.0). PhysioNet.
  https://doi.org/10.13026/C2JT1Q
- Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R.,
  Lungren, M. P., Deng, C., Mark, R. G., and Horng, S. (2019). MIMIC-CXR, a
  de-identified publicly available database of chest radiographs with
  free-text reports. Scientific Data 6, 317.
  https://doi.org/10.1038/s41597-019-0322-0
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C.,
  Johnson, A., Mark, R. G., and Heldt, T. (2026). PhysioNet as a global
  platform for biomedical research. Nature Health.
  https://doi.org/10.1038/s44360-026-00096-z
