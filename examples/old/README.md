# Archived examples

These examples are kept for reference only. They don't run on the current
PyHealth release, or they duplicate a maintained example, and they're no longer
updated. Each file sits at the same relative path it had under `examples/`.

For current material, start with the [tutorial series](../tutorials/series/README.md)
or the [examples index](../README.md).

| File | Why it was archived | Use instead |
|---|---|---|
| `cardiology_detection_isAR_SparcNet.py` | `CardiologyDataset` hasn't been ported to the 2.x dataset API (no `set_task`); 1.x task function | none yet |
| `eeg/sleep_staging/sleep_staging_ISRUC_SparcNet.py` | `ISRUCDataset` hasn't been ported to 2.x (no `set_task`) | `eeg/sleep_staging/sleep_staging_sleepEDF_contrawr.py` |
| `eeg/sleep_staging/sleep_staging_shhs_contrawr.py` | `SHHSDataset` hasn't been ported to 2.x (no `set_task`) | `eeg/sleep_staging/contrawr_sleepedf.ipynb` |
| `conformal_eeg/tuev_eeg_quickstart.ipynb` | duplicate with a stale layer size; fails on 2.x | `eeg/tuh_eeg/tuev_eeg_event_classification.ipynb`, `conformal_eeg/contrawr_tuev_eeg.ipynb` |
| `conformal_eeg/test_tfm_tuev_inference.py`, `conformal_eeg/test_tfm_tuab_inference.py` | personal checkpoint checks; need unreleased weight files | `conformal_eeg/*_conformal.py --model tfm` |
| `cxr/covid19cxr_tutorial_display.py` | near-copy of the COVID-19 CXR tutorial | `cxr/covid19cxr_tutorial.py` |
| `graph_torchvision_model.ipynb` | `Graph_TorchvisionModel` can't be constructed on 2.x; 1.x arguments | `cxr/cnn_cxr.ipynb` |
| `kg_embedding.ipynb`, `lm_embedding_huggingface.ipynb`, `lm_embedding_openai.ipynb`, `prepare_mapping.ipynb` | maintainer scripts that built the medcode resource files; need licensed UMLS data, private checkpoints or a paid API | `medcode.py`, `medcode_icd_mappings.py` |
| `medical_coding_mimic3.ipynb` | unfinished fragment (no training) | `tutorials/series/07_clinical_text.ipynb` |
| `benchmark_pyhealth.ipynb` | internal timing notebook that compared different workloads | `benchmark_perf/` |
| `clinical_tasks/dka_mimic4_stagenet.py` | duplicate of `clinical_tasks/dka_mimic4.py` | `clinical_tasks/dka_mimic4.py` |
| `clinical_tasks/t1d_mimic4_stagenet.py` | duplicate of `clinical_tasks/t1dka_mimic4.py` | `clinical_tasks/t1dka_mimic4.py` |
| `interpretability/interpret_playground.py` | scratch copy of the Transformer interpretation benchmark | `interpretability/mp_transformer_mimic4_interpret.py` |
| `interpretability/interpret_demo.ipynb` | 1.x model arguments and removed dataset methods | `interpretability/*_transformer_mimic4_interpret.py`, `cxr/covid19cxr_tutorial.py` |
| `interpretability/shap_stagenet_mimic4.ipynb` | Colab copy of the script; installs a personal fork | `interpretability/shap_stagenet_mimic4.py` |
| `drug_recommendation/drug_recommendation_mimic3_micron.ipynb` | copy of the script; installs a personal fork | `drug_recommendation/drug_recommendation_mimic3_micron.py` |
| `mortality_prediction/mortality_mimic3_rnn_baseline_cached.ipynb` | duplicate of the RNN mortality script | `mortality_prediction/mortality_mimic3_rnn.py` |
| `mortality_prediction/mimic3_mortality_prediction_cached.ipynb` | personal scratch notebook (no model) | `mortality_prediction/mortality_mimic3_rnn.py` |
| `mortality_prediction/timeseries_mimic4.py` | fragment of the notebook's first cell | `mortality_prediction/timeseries_mimic4.ipynb` |
| `test.py`, `memtest.py` | personal scratch scripts | — |
