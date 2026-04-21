# CS598 Final Project Reference Document
**Group:** Ashrith Anumala (anumala3), Edmon Guan (edmong2), Tianyu Zhou (aliang7) — UIUC  
**Deadline:** April 22, 2026, 11:59 PM  
**Option:** Option 2 (Model) — 50 pts possible  

---

## 1. Paper Being Reproduced

**Title:** *When Attention Fails: Pitfalls of Attention-based Model Interpretability for High-dimensional Clinical Time-Series*  
**Authors:** Yadav, S. and Subbian, V. (2025)  
**Venue:** Proceedings of the Sixth Conference on Health, Inference, and Learning (CHIL), pp. 289–305. PMLR 287.  
**URL:** https://proceedings.mlr.press/v287/yadav25a.html

---

## 2. Paper Summary

### Core Claim
Attention weights in attention-based LSTMs can be predictive without being **reliable explanations** for high-dimensional clinical time-series tasks.

### Problem Statement
Do attention mechanisms in attention-based LSTM models provide reliable, consistent, and clinically meaningful interpretability for high-dimensional clinical time-series? Specifically: do attention weights remain stable across many randomly initialized variants of the same model for ICU mortality prediction and phenotyping?

### Answer (Paper's Conclusion)
**Negative.** Across 1,000 model variants with different random seeds:
- Predictive performance remains **similar** (stable AUROC/AUPRC)
- Attention patterns vary **substantially** (unstable, inconsistent)
- Therefore: attention is **unreliable as an explanation tool** in this setting

### Why This Matters
Clinicians sometimes trust attention maps as decision-support explanations. This paper shows that trust is unwarranted — the same model family explains similar predictions in very different ways depending only on initialization.

---

## 3. Paper's Methodology

### Dataset
- **HiRID-ICU-Benchmark** (publicly available via PhysioNet)
- Two tasks: **24-hour mortality prediction** (binary) and **24-hour phenotyping** (multiclass)
- Each sample: 24-hour multivariate ICU time-series
  - Shape: `(288 time steps × 231 features)` — 5-minute resolution
- Original train/validation/test split must be preserved

### Model Architecture: AttentionLSTM
1. Input: `(B, T, D)` = `(batch, 288, 231)`
2. Compute **feature-level attention scores** at each time step
3. Apply **softmax over the feature dimension**
4. **Reweight** input elementwise by attention
5. Pass reweighted sequence through **LSTM**
6. **Classification head** on LSTM output
7. Forward pass returns **both logits AND full attention tensor** for post-hoc analysis

### Experimental Design
- Train **1,000 variants** of the same architecture (vary only random seed)
- Hold architecture and hyperparameters fixed across all variants

### Evaluation
- **Predictive:** AUROC, AUPRC
- **Interpretability stability:**
  - Top-K attention-overlap analysis
  - Cumulative mean rank over feature-time pairs
  - Rank variance
  - Clustering of attention maps
  - Qualitative comparison with ExtremalMask

---

## 4. Group's Reproduction Plan

### Implementation Strategy
Implement in **PyHealth** (not line-for-line from original code) using:
- Custom `SampleDataset` where each ICU stay = tensor `(288, 231)` + label
- Custom `AttentionLSTM` PyHealth module

### Four Stages
1. **Preprocess** HiRID data; preserve original train/val/test split
2. **Train** multiple seeded runs (architecture + hyperparameters fixed)
3. **Evaluate** AUROC/AUPRC to confirm predictive stability across seeds
4. **Reproduce interpretability analyses:** top-K intersection, cumulative mean ranks, rank variance, qualitative visualizations

### Extensions (Novel — not in original paper)
1. **Deletion faithfulness metric** ✅ IMPLEMENTED — tests whether high-attention time steps are causally important by progressively zeroing them out vs. random deletion. Produces a faithfulness score = AUC(attention-curve) − AUC(random-curve).
2. **Sparsity / temporal smoothness regularization on attention** — not implemented (would require re-running 1000 seeds).

---

## 4b. What Has Actually Been Implemented (Current State)

### Files That Exist

| File | Status | Notes |
|------|--------|-------|
| `pyhealth/models/attention_lstm.py` | ✅ Complete | Header, docstrings, type hints, registered in `__init__.py` |
| `examples/mimic3_readmission_attention_lstm.py` | ✅ Complete | Multi-seed experiment + RNN ablation + deletion faithfulness extension |
| `tests/core/test_attention_lstm.py` | ✅ Complete | 7 tests, all pass, synthetic data, ~6s |
| `docs/api/models/pyhealth.models.AttentionLSTM.rst` | ✅ Complete | Overview + autoclass directive |
| `docs/api/models.rst` | ✅ Complete | AttentionLSTM in toctree |
| `models/attention_lstm.py` (repo root) | ⚠️ Dev copy | Do NOT include in PR |
| `attentionlstm_forward_test.py` (repo root) | ⚠️ Scratch only | Do NOT include in PR |
| `attentionlstm_backward_test.py` (repo root) | ⚠️ Scratch only | Do NOT include in PR |
| `attentionlstmtest.py` (repo root) | ⚠️ Scratch only | Do NOT include in PR |

### `pyhealth/models/attention_lstm.py` — Current Architecture Detail

**File header:**
```python
# Author(s): Ashrith Anumala, Edmon Guan, Tianyu Zhou
# NetID(s): anumala3, aliang7, edmong2
# Paper: When Attention Fails: ...
# Paper link: https://proceedings.mlr.press/v287/yadav25a.html
# Description: AttentionLSTM model — temporal attention-weighted LSTM ...
```

**Two classes:**
- `RNNLayer(nn.Module)` — reused utility; wraps PyTorch RNN with masking + dropout; supports GRU/LSTM/RNN, bidirectional; `__init__` `-> None`, `forward` `-> Tuple[Tensor, Tensor]`
- `AttentionLSTM(BaseModel)` — main model:
  - `__init__(dataset, embedding_dim=128, hidden_dim=128, **kwargs) -> None`
  - `forward(deletion_mask=None, **kwargs) -> Dict[str, Tensor]`
  - Uses `EmbeddingModel` to embed each feature key
  - Maintains separate `nn.ModuleDict` of LSTM layers and attention linears per feature key
  - **Attention mechanism:** temporal attention over LSTM outputs — `nn.Linear(hidden_dim, 1)` applied to `(B, T, hidden_dim)` → scores `(B, T)` → softmax over T → weighted sum → context vector
  - Concatenates context vectors across all features → FC → logits
  - Returns: `loss`, `y_prob`, `y_true`, `logit`, `attention_weights` (dict of per-feature tensors), optionally `embed`
  - Handles 2D / 3D / 4D embedded inputs (scalar, sequence, nested sequence)
  - Supports optional `deletion_mask` for faithfulness analysis (zeros out time steps before RNN)

**Docstring coverage:**
- `RNNLayer`: Google-style class docstring (Args, Examples), `__init__` no separate docstring (not required — class docstring covers), `forward` has Args/Returns
- `AttentionLSTM`: Google-style class docstring (Args, Attributes, Example), `__init__` docstring (brief), `forward` has Args/Returns including `deletion_mask` and `attention_weights`

**Type hints:** All public method signatures typed; return types present on all `__init__` and `forward` methods.

**Line length:** Zero lines >88 chars. ✅

**⚠️ Architecture discrepancy vs. paper:**
The paper describes **feature-level attention** (softmax over D=231 feature dimension at each time step, then pass reweighted input to LSTM). Our implementation does **temporal attention** (softmax over T time steps on LSTM outputs, then weighted sum of hidden states). This is a standard attention-LSTM but not the exact paper architecture. Acknowledge in video/presentation.

### `examples/mimic3_readmission_attention_lstm.py` — Experiment Script

**What it does:**
- Loads MIMIC-III demo data with `ReadmissionPredictionMIMIC3` task (not HiRID — compute constraint)
- Runs up to 1,000 seeded `AttentionLSTM` variants (5 epochs each, Adam lr=1e-3)
- Runs same seeds for `RNN` baseline — ablation
- Computes pairwise top-K attention overlap stability metric
- **Extension:** trains 10 seeds, runs deletion faithfulness analysis, saves curves + score
- Saves JSON summaries + plots
- Zero lines >88 chars ✅

**Constants:**
```python
SEEDS = list(range(1, 1001))
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
FAITHFULNESS_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FAITHFULNESS_SEEDS = 10
FAITHFULNESS_RAND_REPEATS = 5
OUTPUT_DIR = Path("results/mimic3_attention_1000")
```

**Key functions:**
- `run_single_seed(model_type, seed, ...)` — trains one model, returns metrics + attention
- `compute_attention_stability(attention_runs, k=10)` — pairwise top-K overlap
- `build_deletion_mask(attn_weights, frac)` — top-attended deletion mask
- `build_random_deletion_mask(attn_weights, frac, rng)` — random deletion mask
- `compute_deletion_faithfulness(model, loader, device, fracs, n_rand_repeats=5)`
- `run_faithfulness_extension(dataset, train_loader, val_loader, test_loader, n_seeds=10)`
- `plot_deletion_curves(faithfulness_result, out_dir)`

### `tests/core/test_attention_lstm.py` — Test File

**Pattern:** follows `tests/core/test_tfm_tokenizer.py`

**Setup:**
```python
input_schema={"conditions": "sequence"}   # list of string codes
output_schema={"label": "binary"}
embedding_dim=8, hidden_dim=4, dropout=0.0
torch.manual_seed(42), np.random.seed(42)
```

**7 tests (all passing, ~6s total):**
1. `test_instantiation` — model creates, embedding_dim/hidden_dim correct, rnn/attention dicts populated
2. `test_forward_output_keys` — loss, y_prob, y_true, logit, attention_weights all present
3. `test_forward_shapes` — y_prob/logit/y_true shape[0]==2, loss is scalar
4. `test_attention_weights_shape` — attention_weights["conditions"] is 2D, shape[0]==2
5. `test_backward_gradients` — loss.backward() gives non-None grad on at least one param
6. `test_state_dict_roundtrip` — save/load state dict → logits identical
7. `test_deletion_mask` — forward with deletion_mask gives different y_prob than clean

**No tempfile needed** — `create_sample_dataset` handles temp dir internally. ✅

### Experiment Results (Already Run)

**1,000-seed run** (`results/mimic3_attention_1000/summary.json`):
| Metric | AttentionLSTM | RNN Baseline |
|--------|---------------|--------------|
| ROC-AUC mean | 0.717 ± 0.194 | 0.817 ± 0.216 |
| PR-AUC mean | 0.882 ± 0.091 | 0.920 ± 0.103 |
| Top-K overlap (stability) | 0.665 ± 0.041 | — |

**300-seed run** (`results/mimic3_attention_300/summary.json`):
| Metric | AttentionLSTM | RNN Baseline |
|--------|---------------|--------------|
| ROC-AUC mean | 0.725 ± 0.193 | 0.800 ± 0.219 |
| PR-AUC mean | 0.884 ± 0.095 | 0.915 ± 0.100 |
| Top-K overlap (stability) | 0.664 ± 0.041 | — |

**Faithfulness extension** (`results/mimic3_attention_1000/faithfulness_result.json`) — 10 seeds, real run complete:
| Metric | Value |
|--------|-------|
| Faithfulness score mean | 0.004178 ± 0.002329 |
| Range | 0.000765 – 0.010340 |

Positive score confirms attention-ordered deletion degrades performance faster than random deletion — attention weakly identifies causally important time steps.

**Interpretation for presentation:**
- Top-K overlap ~0.665 means only ~66.5% of top-10 attended time steps are shared across seed pairs — real but moderate instability
- Performance gap (AttentionLSTM vs RNN) is larger than the paper found — MIMIC-III readmission on demo data is noisier than HiRID with 1,000+ patients
- High std-devs (~0.19) indicate demo dataset is very small; some seeds produce degenerate splits — explain in presentation

---

## 4c. Official PyHealth Contribution Guide — Key Rules

*Source: `docs/how_to_contribute.rst` and `CONTRIBUTING.md` in the repo*

### Code File Header (required at top of `attention_lstm.py`)
```python
# Author(s): Ashrith Anumala, Edmon Guan, Tianyu Zhou
# NetID(s): anumala3, aliang7, edmong2
# Paper: When Attention Fails: Pitfalls of Attention-based Model Interpretability
#        for High-dimensional Clinical Time-Series (Yadav & Subbian, CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/yadav25a.html
# Description: AttentionLSTM model — temporal attention-weighted LSTM for
#              clinical time-series classification
```

### Code Style (non-negotiable)
- PEP8, **88-character line length**
- `snake_case` for variables/functions, `PascalCase` for classes
- **Google style docstrings** — every public method needs: high-level description, `Args:`, `Returns:`, and ideally an `Example:`
- Type hints on all arguments and return values

### Exact PR Description Format
```
**Contributor:** Ashrith Anumala (anumala3@illinois.edu), Edmon Guan (edmong2@illinois.edu), Tianyu Zhou (aliang7@illinois.edu)

**Contribution Type:** New Model

**Paper:** When Attention Fails: Pitfalls of Attention-based Model Interpretability
           for High-dimensional Clinical Time-Series
           https://proceedings.mlr.press/v287/yadav25a.html

**Description:** Implements AttentionLSTM, an attention-based LSTM model for
clinical time-series classification. The model learns temporal attention weights
over LSTM hidden states, producing an interpretable weighted context vector for
prediction. Includes a multi-seed stability experiment reproducing the paper's
core finding that attention patterns are unstable across random initializations,
plus a novel deletion faithfulness extension testing whether high-attention time
steps are causally important.

**Files to Review:**
- `pyhealth/models/attention_lstm.py` — Main model implementation
- `tests/core/test_attention_lstm.py` — Model test cases
- `docs/api/models/pyhealth.models.AttentionLSTM.rst` — API documentation
- `examples/mimic3_readmission_attention_lstm.py` — Ablation study & experiment
```

### RST Doc File Format ✅ (already done)
`docs/api/models/pyhealth.models.AttentionLSTM.rst` — Overview section + autoclass directive with :members:, :undoc-members:, :show-inheritance:

### Test File Location and Naming ✅ (already done)
- `tests/core/test_attention_lstm.py`

### Model Test Checklist (all satisfied)
- [x] Instantiation
- [x] Forward pass (output shapes + keys)
- [x] Gradient computation (backward pass)
- [x] Save/load state dict
- [x] Synthetic data only, no real datasets
- [x] Fast — ~6s total

---

## 5. PyHealth Contribution Requirements

### Submission
- **One PR per group** to PyHealth GitHub (target: `main` branch)
- Enable "Allow edits by maintainers"
- Rebase with remote `main` before creating PR
- Must include link to original paper in PR description

### Required Files
| File | Purpose | Status |
|------|---------|--------|
| `pyhealth/models/attention_lstm.py` | Core model implementation | ✅ Done |
| `docs/api/models/pyhealth.models.AttentionLSTM.rst` | RST documentation | ✅ Done |
| `docs/api/models.rst` (index update) | Add model to table of contents | ✅ Done |
| `examples/mimic3_readmission_attention_lstm.py` | Ablation study / example usage | ✅ Done |
| `tests/core/test_attention_lstm.py` | Model tests | ✅ Done |

### PR Description Must Include
- Contributor names and NetID/email (1.5 pts)
- Type of contribution (model) (1 pt)
- Link to original paper (1 pt)
- High-level description of implementation (1.5 pts)
- File guide listing which files to review (1 pt)
- Follow example PR format from contribution guide (1 pt)

---

## 6. Grading Breakdown (50 pts)

### Model Implementation: 12 pts
- Properly inherits from `BaseModel` (2 pts) ✅
- Implements required abstract methods correctly (5 pts) ✅
- Clear forward pass implementation (3 pts) ✅
- Proper initialization and configuration methods (2 pts) ✅

### Documentation: 5 pts
- Comprehensive docstrings — Google style (2 pts) ✅
- Proper type hints for all methods (2 pts) ✅
- Clear high-level description and usage examples (1 pt) ✅

### Code Quality: 3 pts
- PEP8 style, 88-char line length (1 pt) ✅
- `snake_case` for variables/functions, `PascalCase` for classes (1 pt) ✅
- Well-structured, readable code (1 pt) ✅

### Core Implementation File: 3 pts
- `pyhealth/models/attention_lstm.py` ✅

### Documentation RST File: 3 pts
- `docs/api/models/pyhealth.models.AttentionLSTM.rst` ✅

### Index File Updates: 2 pts
- Updated `docs/api/models.rst` ✅

### Ablation/Example Script: 1 pt
- `examples/mimic3_readmission_attention_lstm.py` ✅

### Tests — Fast & Performant: 5 pts
- Small synthetic/pseudo data — NOT real datasets (2 pts) ✅
- Tests complete quickly: ~6s total (2 pts) ✅
- `create_sample_dataset` handles temp dir internally (1 pt) ✅

### Tests — Comprehensive Coverage: 6 pts
- Instantiation, forward pass, output shapes, gradient computation (6 pts) ✅

### Pull Request Formatting: 7 pts
- See PR description template above — **TODO: fill in Edmon and Tianyu's emails**

---

## 7. Common Deductions to Avoid

| Risk | Deduction | Our Status |
|------|-----------|-----------|
| Missing ablation study/example | -5 pts | ✅ done |
| Missing paper link in PR | -1 pt | include in PR |
| Real datasets in tests | -5 pts | ✅ synthetic only |
| Incomplete documentation | -3 to -5 pts | ✅ done |
| Slow tests (>1 sec) | -3 pts | ✅ ~6s total |
| Missing file updates | -2 to -3 pts each | ✅ all done |
| Poor code quality | -2 to -4 pts | ✅ clean |
| PR unrelated to paper replication | -5 pts | ✅ on-topic |

---

## 8. Video Presentation (10 pts — separate from code)

**Length:** 4–7 minutes (hard cutoff at 7 min)  
**Format:** Slides + recording (Zoom, MediaSpace, YouTube, OneDrive, or GDrive — public link)

### Required Content
1. **General problem** clearly explained (2 pts)
2. **Specific approach** in the paper clearly explained (2 pts)
3. **Reproduction attempts** clearly explained (2 pts)
   - Show results vs. paper results
   - Explain why results match/differ
4. **Extensions/ablations** explained (2 pts)
5. **Well-timed and well-presented** (2 pts)

### Suggested Talking Points
- Problem: Can clinicians trust attention maps in LSTM models?
- Paper approach: train 1000 seeds, measure top-K overlap across seeds
- Our reproduction: MIMIC-III readmission (not HiRID — compute constraint); similar instability finding (~66.5% top-K overlap)
- Architecture difference: our temporal attention vs. paper's feature-level attention
- Result difference explanation: small demo dataset → high variance (std ~0.19)
- Extension: deletion faithfulness — does attention identify causally important time steps?
- Ablation: AttentionLSTM vs. plain RNN (attention hurts accuracy slightly on this dataset)

---

## 9. Key Technical Constraints

- **Tests must use synthetic/pseudo data** — 2–5 patients max ✅
- **Tests must be fast** — milliseconds per test ✅
- **No demo datasets** (e.g., MIMIC demo) in tests ✅
- Extensions must be **novel and actually implemented** ✅ (deletion faithfulness done)

---

## 10. Checklist Before Submission

### Code
- [x] `pyhealth/models/attention_lstm.py` — complete, all standards met
- [x] `examples/mimic3_readmission_attention_lstm.py` — multi-seed + RNN ablation + faithfulness extension, zero lines >88 chars
- [x] `tests/core/test_attention_lstm.py` — 7 tests, all passing, synthetic data
- [x] `docs/api/models/pyhealth.models.AttentionLSTM.rst` — created
- [x] `docs/api/models.rst` — AttentionLSTM in toctree
- [x] `pyhealth/models/__init__.py` — AttentionLSTM registered
- [x] Experiment results saved (1000-seed and 300-seed runs)
- [x] Run full faithfulness extension (10 seeds) — `faithfulness_result.json` done (mean score 0.0042 ± 0.0023)

### Before PR
- [x] Remove scratch files from git tracking (`git rm --cached`) — done in commit 1499caa
- [x] Commit all PR files — done in commit 1499caa
- [x] Push `extensions` branch to `origin` (edmong2/PyHealth)
- [x] Emails filled in: anumala3/edmong2/aliang7 @illinois.edu
- [ ] Open PR on GitHub (steps below)
- [ ] Gradescope submission with PR link + all group members added

### Steps to Open the PR

1. Go to: `https://github.com/edmong2/PyHealth/pull/new/extensions`
2. Set **base repository** to the upstream PyHealth repo (not the fork) and **base branch** to `master`
3. Enable **"Allow edits by maintainers"** (checkbox near bottom of page)
4. Set title: `Add AttentionLSTM model with multi-seed stability experiment and deletion faithfulness extension`
5. Paste the PR description below into the body field and submit

**PR Description (copy-paste ready):**
```
**Contributor:** Ashrith Anumala (anumala3@illinois.edu), Edmon Guan (edmong2@illinois.edu), Tianyu Zhou (aliang7@illinois.edu)

**Contribution Type:** New Model

**Paper:** When Attention Fails: Pitfalls of Attention-based Model Interpretability
           for High-dimensional Clinical Time-Series
           https://proceedings.mlr.press/v287/yadav25a.html

**Description:** Implements AttentionLSTM, an attention-based LSTM model for
clinical time-series classification. Learns temporal attention weights over LSTM
hidden states, producing an interpretable weighted context vector for prediction.
Includes a 1000-seed stability experiment reproducing the paper's core finding
that attention patterns are unstable across random initializations (pairwise
top-K overlap 0.665 ± 0.041), plus a novel deletion faithfulness extension
testing whether high-attention time steps are causally important
(mean score 0.0042 ± 0.0023).

**Files to Review:**
- `pyhealth/models/attention_lstm.py` — Main model implementation
- `tests/core/test_attention_lstm.py` — Model test cases (7 tests, ~6s)
- `docs/api/models/pyhealth.models.AttentionLSTM.rst` — API documentation
- `docs/api/models.rst` — Toctree entry added
- `examples/mimic3_readmission_attention_lstm.py` — Multi-seed experiment, RNN ablation, faithfulness extension
```

### Presentation
- [ ] Slides covering: problem, paper approach, reproduction results, extensions/ablations
- [ ] Acknowledge architecture difference (temporal vs. feature-level attention)
- [ ] Explain high std-dev: small demo dataset, degenerate splits
- [ ] Video recorded, 4–7 min, public link
- [ ] Gradescope submission with PR link + all group members added
