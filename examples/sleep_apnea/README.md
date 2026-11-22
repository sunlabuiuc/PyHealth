# Mock NCH / CHAT data

Synthetic stand-ins for the NSRR NCH Sleep DataBank and CHAT studies, for developing the
PyHealth port of `apnea-missing-modality` before we have data access.

From the PyHealth repo root, with PyHealth installed (`pip install -e .`):

```
python examples/sleep_apnea/make_mock_apnea_data.py --force                                            # out/  (20 min studies, ~375 MB)
python examples/sleep_apnea/make_mock_apnea_data.py --minutes 6 --out examples/sleep_apnea/tiny --force # tiny/ (~100 MB)
```

Output is deterministic for a given `--seed` (default 598). Both output folders are gitignored;
regenerate them locally rather than committing them.

## Layout

| Stage | Path | Matches |
|---|---|---|
| 1. raw | `raw/nchsdb/sleep_data/<pat>_<study>.edf` + `.tsv` (onset, duration, description) | NCH download (`D:\nchsdb`) |
| | `raw/nchsdb/health_data/SLEEP_STUDY.csv`, `DEMOGRAPHIC.csv` | NCH health data (subset of columns) |
| | `raw/nchsdb/AHI.csv` (Study, AHI) | the `D:\Data\AHI.csv` the original reads |
| | `raw/chat/polysomnography/edfs/{baseline,followup}/chat-<visit>-<id>.edf` | CHAT download |
| | `raw/chat/polysomnography/annotations-events-nsrr/<visit>/chat-<visit>-<id>-nsrr.xml` | NSRR XML annotations |
| | `raw/chat/polysomnography/edfs/baseline/chat-baseline-<id>-nsrr.tsv` | the TSV the original preprocessing reads (baseline only) |
| | `raw/chat/datasets/chat-{baseline,followup}-dataset-mock.csv` | CHAT subject tables (subset of columns) |
| 2. preprocessed | `preprocessed/nch_30x64/<pat>_<study>_<apneaSec>_<hypopneaSec>.npz` | `data/nch/preprocessing.py` output |
| | `preprocessed/chat_b_30x64/chat-baseline-<id>.edf_<a>_<h>.npz` | `data/chat/preprocessing.py` output |
| 3. folds | `folds/nch_30x64_{0..4}.npz`, `folds/chat_b_30x64_{0..4}.npz` | what `train_test.py` loads (`x`, `y_apnea`, `y_hypopnea`) |

`manifest.json` lists every study's ground-truth events, why studies were skipped, and fold contents.

The mock raw data loads with `NCHSleepDataset` and `CHATDataset`:

```python
from pyhealth.datasets import CHATDataset, NCHSleepDataset

if __name__ == "__main__":  # needed on Windows: PyHealth starts worker processes
    nch = NCHSleepDataset(root="examples/sleep_apnea/out/raw/nchsdb")
    nch.stats()   # 8 patients, 10 studies
    chat = CHATDataset(root="examples/sleep_apnea/out/raw/chat")
    chat.stats()  # 8 patients, 10 studies (8 baseline + 2 follow-up)
```

Stage 2 `.npz`: `data` (epochs, channels, samples). NCH has 11 channels at 64 Hz (1920 samples).
CHAT has 15 re-referenced channels at 128 Hz (3840 samples). `labels_apnea` and `labels_hypopnea`
give the number of seconds of each event type in each 30 s epoch. Wake epochs are removed.

Stage 3 `x`: (N, 1920, 9). Columns: EOG, EEG C4, nasal pressure, airflow, SpO2, CO2, ECG,
R-peak amplitude, RRI. That is `modality.py` index 0=eog, 1=eeg, 3=resp, 4=spo2, 5=co2, 7-8=ecg.

## Edge cases included on purpose
- NCH patients 10012 and 10071 each have two studies (patient-level folds).
- NCH 10063 is recorded at 400 Hz and CHAT 300003 at 256 Hz, so resampling is exercised.
- NCH 10104 and CHAT 300007 have no events, so they are skipped (low AHI / no respiratory events).
- NCH 10117 has no `Capno` channel and CHAT 300008 has no `Cap` channel, so they are skipped for insufficient channels.
- CHAT 300001 and 300002 also have a follow-up study, with XML annotations only. Like the original, stages 2 and 3
  use baseline studies only.

## Differences from the original code
- R-peaks are found with `scipy.signal.find_peaks` instead of biosppy's Hamilton segmenter.
- Epoch start times are read before resampling.
- The NCH `data.inp_dim` bug is not reproduced, and the CHAT off-by-one (`file_list[patient - 1]`) is fixed.
- CHAT fold files are named `chat_b_30x64_<k>.npz`, which is what `main_*.py` loads. The original
  dataloader writes `chat_b_30x64__<k>`.
- Every study uses one sampling rate for all channels. Real NSRR EDFs often mix rates
  (for example, SpO2 at a lower rate).
