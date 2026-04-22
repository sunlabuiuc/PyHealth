# Evidence Retrieval Benchmark Results

All ablations run on the bundled `SyntheticEHRNotesDataset` (5 patients, 8 note-condition samples, 5 positives, 3 negatives). Each row reports binary metrics on the Pass-1 note-level decision.

Backends compared: **offline stub** vs **gpt-4o-mini**.

## Ablation I — sequential vs single-prompt

| Backend | Prompt style | Accuracy | Precision | Recall | FP | Explanations |
|---|---|:-:|:-:|:-:|:-:|:-:|
| offline stub | sequential | 0.875 | 0.833 | 1.000 | 1 | 6/8 |
| offline stub | single | 0.875 | 0.833 | 1.000 | 1 | 6/8 |
| gpt-4o-mini | sequential | 1.000 | 1.000 | 1.000 | 0 | 5/8 |
| gpt-4o-mini | single | 1.000 | 1.000 | 1.000 | 0 | 8/8 |

## Ablation II — LLM vs CBERT-lite IR baseline

| Model | Accuracy | Precision | Recall | FP |
|---|:-:|:-:|:-:|:-:|
| offline stub (sequential) | 0.875 | 0.833 | 1.000 | 1 |
| gpt-4o-mini (sequential) | 1.000 | 1.000 | 1.000 | 0 |
| CBERT-lite IR baseline | 0.625 | 0.750 | 0.600 | 1 |

## Ablation III — note-length budget sweep (novel axis)

| Backend | max_note_chars | Accuracy | Precision | Recall | FP |
|---|:-:|:-:|:-:|:-:|:-:|
| offline stub | 80 | 0.750 | 0.800 | 0.800 | 1 |
| offline stub | 160 | 0.875 | 0.833 | 1.000 | 1 |
| offline stub | 320 | 0.875 | 0.833 | 1.000 | 1 |
| offline stub | 4000 | 0.875 | 0.833 | 1.000 | 1 |
| gpt-4o-mini | 80 | 1.000 | 1.000 | 1.000 | 0 |
| gpt-4o-mini | 160 | 1.000 | 1.000 | 1.000 | 0 |
| gpt-4o-mini | 320 | 1.000 | 1.000 | 1.000 | 0 |
| gpt-4o-mini | 4000 | 1.000 | 1.000 | 1.000 | 0 |
