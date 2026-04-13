# KEEP Implementation Comparison: Paper vs G2Lab vs Desmond vs Our PyHealth Port

**Date:** 2026-04-12
**Purpose:** Side-by-side comparison of four KEEP implementations to identify what to align, what to keep, and what to combine.

---

## Overview

| | **Paper** | **G2Lab/keep** | **Desmond's keep-mimic4** | **Our PyHealth port** |
|---|---|---|---|---|
| Type | Algorithm description | Research code | Standalone reproduction | PyHealth integration |
| Dataset | UK Biobank + MIMIC-IV | UK Biobank | MIMIC-IV | MIMIC-III + IV |
| Goal | Describe KEEP | Produce published numbers | Match paper exactly | Make KEEP consumable by PyHealth models |
| Data format | Pre-built OMOP pickles | Pre-built pickles | Athena CSVs + MIMIC | Athena CSVs + MIMIC |
| Output | — | numpy pickle | text file | text file |
| Tests | — | None | Planned | 51 unit tests |

---

## Graph Construction

### Root node

| | Root choice | # nodes | Source |
|---|---|---|---|
| **Paper** | `4274025` "Disease" | ~68K (unfiltered) | Appendix A.1.1 |
| **G2Lab** | Pre-built graph loaded from pickle | 5,686 (after UK Biobank count filter) | configs.py |
| **Desmond** | `4274025` "Disease" | 68,396 | Matches paper |
| **Our port** | All 187 standard SNOMED Condition roots | 65,375 | Broader interpretation |

**Issue:** Our port uses a broader root set. The paper is explicit: root is 4274025.

**Fix:** One-line change in `build_hierarchy_graph()` to require depth-from-4274025 instead of depth-from-any-root.

### Source table for hierarchy

| | Table used | Method |
|---|---|---|
| **Paper** | `CONCEPT_ANCESTOR` | `min_levels_of_separation` from root |
| **G2Lab** | (pre-built) | N/A |
| **Desmond** | `CONCEPT_ANCESTOR` | Matches paper |
| **Our port** | `CONCEPT_RELATIONSHIP` "Is a" edges | BFS from roots |

Functionally similar. CONCEPT_ANCESTOR is the transitive closure of "Is a" edges. Both produce valid hierarchy graphs.

### Orphan handling

| | Approach |
|---|---|
| **Paper** | Not discussed |
| **G2Lab** | (pre-built graph) |
| **Desmond** | Explicit orphan rescue: 42 unreachable nodes rescued via closest in-set ancestor |
| **Our port** | Not handled. Likely same bug |

**Issue:** We probably have orphan nodes in our graph that Desmond rescued. His discovery: some standard SNOMED Condition concepts have direct parents in the Observation domain, which get filtered out, orphaning downstream subtrees.

**Fix:** Adopt Desmond's orphan rescue logic.

---

## ICD-to-SNOMED Mapping

| | Approach | Handles multi-target? |
|---|---|---|
| **Paper** | Not explicitly described (UKBB data is native OMOP) | — |
| **G2Lab** | Not explicitly described | — |
| **Desmond** | `Dict[str, List[int]]` with all in-graph targets | Yes |
| **Our port** | `Dict[str, int]` with first match | No — silently drops alternatives |

**Issue:** 24% of ICD-10 codes map to multiple SNOMED concepts (combination codes like "A01.04 Typhoid arthritis" → both typhoid fever AND inflammatory arthritis). We silently keep one and drop the other.

**Fix:** Return `Dict[str, List[int]]`, update consumers to handle lists.

---

## Co-occurrence Matrix

### What goes in the diagonal

| | Diagonal |
|---|---|
| **Paper** | Not discussed |
| **G2Lab** | Patient counts per code (used by cui2vec for PMI calculation) |
| **Desmond** | Zero (standard GloVe) |
| **Our port** | Patient counts per code (matches G2Lab) |

**Investigation:** We found G2Lab uses the diagonal to store singleton counts because cui2vec reads them for PMI. Desmond's plan says "Diagonal is zero" but that may be a proposal mistake — he'd be diverging from the reference.

**Status:** Our choice matches G2Lab. Keep as-is.

### Roll-up procedure

| | Rollup |
|---|---|
| **Paper** | Dense roll-up: each code propagates to ALL ancestors |
| **G2Lab** | Implicit (inherited from UK Biobank preprocessing) |
| **Desmond** | Dense roll-up planned |
| **Our port** | Implemented in `rollup_codes()` — matches paper |

**Status:** Both ports correctly implement dense roll-up.

### Count filter

| | Filter |
|---|---|
| **Paper** | Applied implicitly via UK Biobank data |
| **G2Lab** | `_ct_filter` suffix on all pipeline files |
| **Desmond** | Explicit Story 5b: drop concepts with zero patients |
| **Our port** | **Not implemented** |

**Issue:** This is why our graph stays at 65K when the paper's effective vocabulary is 5,686. Without the count filter, we train on 60K concepts that never appear in patient records.

**Fix:** Add a count filter step that intersects the depth-5 graph with concepts observed in patient data.

---

## Regularized GloVe (Stage 2)

### Regularization distance

| | Distance | Lambda | Scaling |
|---|---|---|---|
| **Paper text** | L2 norm squared | 1e-3 | Per-element (Equation 4) |
| **G2Lab code** | Cosine (REG_NORM=None default) | 1e-5 | Sum over batch |
| **Desmond's plan** | L2 (reg_norm=2, paper-faithful) | 1e-3 (paper-faithful) | Not specified |
| **Our port (fixed)** | Cosine (default), L2 available | 1e-5 | Sum over batch (matches G2Lab) |

**The critical finding:** Paper text and G2Lab code disagree on distance, lambda, AND normalization convention. The code produced the published numbers.

**Our decision:** Default to G2Lab code (cosine, 1e-5, sum). Expose paper variant via `use_cosine_reg=False, lambd=1e-3`. Document the deviation.

**Desmond's decision:** Default to paper text (L2, 1e-3). He may hit reproduction issues if the code's choices are what produced the paper's numbers.

### Optimizer

| | Optimizer | LR |
|---|---|---|
| **Paper (Algorithm 1)** | AdamW | not specified |
| **G2Lab code** | Adagrad | 0.05 |
| **Desmond** | AdamW (paper-faithful) | matches paper |
| **Our port** | Adagrad (matches G2Lab) | 0.05 |

**Same pattern.** We picked code, Desmond picked paper. Neither has run it yet.

### Row vs row+col regularization

| | Regularizes |
|---|---|
| **G2Lab code** | Both `i_indices` and `j_indices` |
| **Desmond's plan** | Not explicit — "embedding_i + embeddings_u(i_indices)" suggests row only? |
| **Our port** | Both (fixed in recent bugfix) |

**Status:** We fixed this. Desmond's plan description is ambiguous; verify when reviewing his code.

---

## Node2Vec (Stage 1)

### Hyperparameters

All three match:
- Embedding dim: 100
- Walk length: 30
- Number of walks: 750
- p, q: 1, 1 (DeepWalk)
- Window: 10
- Min count: 1
- Batch words: 4096

**Status:** No disagreements here.

### Graph for Node2Vec

| | Graph used |
|---|---|
| **Paper** | Filtered (5,686 nodes) |
| **G2Lab** | Pre-filtered pickle |
| **Desmond** | Filtered (Story 5b → Story 4) |
| **Our port** | **Unfiltered (65K nodes)** |

**Issue:** We run Node2Vec on 65K nodes when most of them never appear in patient data. Slow and potentially harmful (walks pass through empty regions).

**Fix:** Apply count filter first, then run Node2Vec on the filtered graph.

---

## Patient Data Pipeline

### 2-occurrence filter

| | Filter |
|---|---|
| **Paper** | Required (Appendix A.4) |
| **G2Lab** | Implemented in create_cohort_sentence.py |
| **Desmond** | Planned |
| **Our port** | Implemented (`min_occurrences=2`) |

### Censoring rule

| | Censoring | Scope |
|---|---|---|
| **Paper** | 2nd occurrence date < outcome date | Extrinsic eval only |
| **G2Lab** | Implemented in `create_cohort_sentence.py` | Extrinsic eval only |
| **Desmond** | Planned | Extrinsic eval |
| **Our port** | **Intentionally not implemented** | N/A — handled by PyHealth |

**Decision (2026-04-13):** Do NOT add censoring to KEEP's embedding training pipeline.

**Why:**
1. **KEEP embeddings are population-level**, trained once and reused across any downstream task. They have no concept of an "index date" or "outcome."
2. **The paper itself uses complete patient history** for embedding training (Appendix A.4: "Co-occurrence is determined based on the patient's complete medical history, rather than being restricted to individual visits").
3. **PyHealth's task processors handle temporal cutoffs** automatically — when `MortalityPredictionMIMIC3` generates samples, it only includes events before each sample's timestamp.
4. **Adding censoring to embedding training would discard signal** for no benefit, weakening the embeddings.

**Where censoring actually applies:**
- Extrinsic evaluation (downstream prediction) → handled by PyHealth task generation
- Reproducing paper's exact UK Biobank tasks → would require a custom task extractor (not a KEEP pipeline change)

The censoring rule is documented in `extract_patient_codes_from_df`'s docstring as an explicit non-decision with rationale.

---

## PyHealth Integration

### SNOMED as code_mapping target

| | Support |
|---|---|
| **Paper** | N/A |
| **G2Lab** | N/A |
| **Desmond** | Via Story 9 custom task (outputs OMOP IDs directly) |
| **Our port** | `code_mapping=("ICD9CM", "SNOMED")` — first-class vocabulary |

**Advantage:** Ours integrates cleanly with existing PyHealth models. Desmond's requires writing a custom task per downstream use case.

### GRASP pretrained embeddings

| | Support |
|---|---|
| **Desmond** | Planned (Story 8) |
| **Our port** | Implemented (same fix) |

### Tests

| | Coverage |
|---|---|
| **Desmond** | Planned (Story 11) |
| **Our port** | 51 unit tests across 4 test files |

### Convenience runner

| | Wrapper |
|---|---|
| **Desmond** | 7 separate scripts run manually |
| **Our port** | `run_keep_pipeline()` — single function call |

---

## Executive Summary Table

| Aspect | Paper | G2Lab | Desmond | Our Port | Winner |
|---|---|---|---|---|---|
| Graph root | 4274025 | pre-built | 4274025 ✓ | 187 roots ✗ | Desmond / Paper |
| Graph source | CONCEPT_ANCESTOR | pre-built | CONCEPT_ANCESTOR | RELATIONSHIP "Is a" | Tie |
| Orphan rescue | n/a | n/a | Yes ✓ | No ✗ | Desmond |
| ICD multi-target | n/a | n/a | Yes ✓ | No ✗ | Desmond |
| Diagonal counts | n/a | Yes | No | Yes ✓ | Us / G2Lab |
| Dense rollup | Yes | implicit | Yes | Yes ✓ | Tie |
| Count filter | implicit | Yes | Yes ✓ | No ✗ | Desmond |
| Reg distance | L2 | Cosine | L2 | Cosine | TBD (empirical) |
| Lambda | 1e-3 | 1e-5 | 1e-3 | 1e-5 | TBD (empirical) |
| Optimizer | AdamW | Adagrad | AdamW | Adagrad | TBD (empirical) |
| Row+col reg | — | Yes | ambiguous | Yes ✓ | Us / G2Lab |
| Censoring | Eval only | Eval only | Eval only | N/A in training (PyHealth handles eval) | Ours (correct scope) |
| Node2Vec on filtered graph | Yes | Yes | Yes | No ✗ | Desmond |
| PyHealth integration | — | — | minimal | **Full** ✓ | Us |
| Tests | — | — | planned | **51 tests** ✓ | Us |
| Convenience API | — | — | scripts | `run_keep_pipeline()` ✓ | Us |

---

## The Paper vs Code Divergence

Three places where paper text disagrees with G2Lab code that produced the published numbers:

| Aspect | Paper says | Code does |
|---|---|---|
| Reg distance | L2 squared | Cosine |
| Lambda | 1e-3 | 1e-5 |
| Optimizer | AdamW | Adagrad |

**Desmond's strategy:** Trust the paper.
**Our strategy:** Trust the code.

Only empirical results (running both variants and comparing AUPRC to published Table 4) can settle this. Neither of us has done that yet.

---

## Recommended Action Items

### High priority (fix in our port)

1. **Change root to 4274025 "Disease"** — 1-line fix, paper-faithful
2. **Add count filter** — reduces 65K → ~5K nodes, matches paper's effective vocabulary
3. **Add orphan rescue** — prevents 42+ unreachable nodes (adopt Desmond's logic)
4. **Fix multi-target ICD mapping** — return lists instead of silently dropping
5. **Re-run Node2Vec on filtered graph** — currently wastes compute on 60K empty nodes

### Medium priority

6. **Add censoring rule** — for fair extrinsic eval comparison
7. **Expose optimizer and reg_distance as parameters** — enable paper/code variant comparison

### Not priorities (handle at integration time)

- PyHealth integration (we already have it)
- Tests (we already have them)
- Convenience runner (we already have it)

---

## Integration Plan: 5 Concrete Steps

### Step 1: Read Desmond's actual code

His plan describes what he'll build. We need to see the real code.

```
https://github.com/ddhangdd/keep-mimic4

Files to read, in priority order:
  1. keep_pipeline/scripts/build_omop_graph.py          ← root 4274025 + orphan rescue
  2. keep_pipeline/scripts/build_icd_to_omop_mapping.py ← multi-target handling
  3. keep_pipeline/scripts/apply_count_filter.py        ← the 68K → ~5K reduction
  4. keep_pipeline/scripts/train_keep_glove.py          ← his reg loss (paper or code?)
```

For each, note: implemented or planned? exact hyperparameters? matches the paper?

### Step 2: Port his uncontroversial fixes into our pipeline

Four things both paper and code agree on:

| Fix | Current state | Target state | Files touched |
|-----|---------------|--------------|---------------|
| Root = 4274025 | BFS from 187 roots | BFS from 4274025 only | `build_omop_graph.py` |
| Count filter | 65K unfiltered nodes | Filter to observed concepts | `run_pipeline.py` |
| Orphan rescue | ~42 unreachable nodes | Rescue via closest ancestor | `build_omop_graph.py` |
| Multi-target ICD | `Dict[str, int]` (drops 24%) | `Dict[str, List[int]]` | `build_omop_graph.py`, `build_cooccurrence.py`, `export_embeddings.py` |

### Step 3: Write intrinsic eval script

Lives in our port. Takes any `keep_vectors.txt` and computes paper Table 2/3 metrics.

```
pyhealth/medcode/pretrained_embeddings/keep_emb/intrinsic_eval.py

Functions:
  resnik_correlation(embeddings, graph)           → paper target 0.68
  cooccurrence_correlation(embeddings, cooc)      → paper target 0.62
  comorbidity_benchmark(embeddings, known_pairs)  → paper target 2.13

Per-disease correlations (paper methodology):
  K1=10 most similar from embedding, K2=150 random
  RUNS=250 for bootstrap distribution
  Significance threshold: 95th percentile of null (excludes known pairs)
```

This is the validation tool. Both implementations run through it.

### Step 4: Run both pipelines, compare scores

```
Our port:
  run_keep_pipeline() → our_keep_vectors.txt
  intrinsic_eval(our_keep_vectors.txt) → score X

Desmond's port:
  his scripts → desmond_keep_vectors.txt
  intrinsic_eval(desmond_keep_vectors.txt) → score Y

Compare X, Y, and paper's 0.68.
Whichever is closer wins the hyperparameter debate empirically.
```

### Step 5: Ablation with validated embeddings

Whoever's file scores closer becomes the ablation input:

```python
model = GRASP(
    code_mapping=("ICD9CM", "SNOMED"),
    pretrained_emb_path="keep_vectors.txt",  # the validated winner
)
```

Run A1 (random), A2 (code_mapping), A3 (KEEP). This is the paper headline.

---

## This Week's Concrete Tasks

Order of operations (~1 week of work):

```
Day 1:  Read Desmond's 4 scripts
Day 2:  Port root = 4274025 (TDD: write test first)
Day 3:  Port count filter (TDD)
Day 4:  Port orphan rescue (TDD)
Day 5:  Port multi-target ICD mapping (TDD)
Day 6:  Write intrinsic eval script
Day 7:  Verify our pipeline still produces valid output
```

All of this lives in our port. We adopt Desmond's ideas without cloning his repo.

---

## The Coordination Message

Send to Desmond:

> "I'm adopting your four data-pipeline fixes (root 4274025, count filter,
> orphan rescue, multi-target ICD). You keep paper-faithful hyperparams.
> Our pipelines will now run the SAME data through different hyperparams.
> When both produce keep_vectors.txt, we score them against paper's 0.68
> Resnik target. Winner's hyperparameters become the PyHealth default."

Honest framing: not duplicating work, adopting his strengths (data pipeline correctness) while keeping ours (PyHealth integration, tests).

---

## Collaboration Strategy with Desmond

**Don't duplicate. Coordinate.**

### Shared artifacts
- One `CONCEPT.csv` + one `CONCEPT_RELATIONSHIP.csv` — both use the same Athena version
- One SNOMED graph (if we adopt root 4274025 + count filter + orphan rescue from him)
- One ICD-to-SNOMED mapping (with multi-target support)
- One `keep_vectors.txt` output format

### Division of labor
- **Desmond:** Paper-faithful pipeline. Adopts our PyHealth integration for downstream eval.
- **Us:** PyHealth integration. Adopts his graph construction and multi-target mapping.

### Merge point
The `keep_vectors.txt` file. If both our pipelines produce files with the same format and token space (SNOMED concept IDs), either can be loaded through our PyHealth integration. This gives the team two independent reproductions to cross-validate.

### For the paper
Report BOTH variants in the ablation table:
- KEEP (paper-faithful): L2, lambda=1e-3, AdamW
- KEEP (code-faithful): Cosine, lambda=1e-5, Adagrad

This is a contribution — the original repo doesn't document the deviation. Running both and reporting the delta is novel work.

---

## The Question That Decides Everything

**Does the paper's published AUPRC match G2Lab's code (with cosine, 1e-5, Adagrad) or the paper text (L2, 1e-3, AdamW)?**

This is an empirical question. The first team to run both variants on real data answers it. Until then, we're making educated guesses.

Whoever runs the full pipeline first wins the framing: "our choice matches the paper" vs "our choice matches the code". Both are defensible. Only the numbers settle it.
