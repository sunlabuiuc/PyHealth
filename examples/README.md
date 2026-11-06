# PyHealth examples

**New to PyHealth? Start with the [tutorial series](tutorials/series/README.md).**
It runs on Colab with public or synthetic data and goes from a first model to
contributing a dataset.

The examples here are shorter and more specialised: one model, task, dataset or
feature each. Each one was checked against PyHealth 2.1 and runs end to end on
demo or synthetic data. Its header (the module docstring, or the first cell of a
notebook) says:

- what it shows,
- which dataset it needs, and whether that dataset is credentialed,
- how to run it and where to set the data path,
- how to try it without credentials.

Most MIMIC/eICU examples can be tried on the demo data in
[`test-resources/core/`](../test-resources/core/) (`mimic3demo`, `mimic4demo`,
`eicudemo`) or on the public synthetic MIMIC-III that some scripts load by
default. Those demos only show that the code path works: they're far too small
for meaningful metrics, and some tasks get a single label class from them.

Scripts call `set_task`, which starts worker processes, so run them as scripts
(`python examples/<file>.py`). Don't paste them into a REPL without the
`if __name__ == "__main__":` guard.

Older examples that no longer run, or that duplicated a maintained one, are in
[`old/`](old/README.md).

## By task

| Folder | Contents |
|---|---|
| [`mortality_prediction/`](mortality_prediction/) | In-hospital mortality on MIMIC-III, MIMIC-IV and eICU: RNN, StageNet, StageAttentionNet, Transformer, AdaCare, ConCare, GRASP, Agent, TCN, XGBoost, EHRMamba, multimodal (labs, notes, CXR) |
| [`readmission/`](readmission/) | Readmission on MIMIC-III, MIMIC-IV, eICU and OMOP, plus fairness metrics |
| [`length_of_stay/`](length_of_stay/) | Length of stay on MIMIC-III, MIMIC-IV and eICU: RNN, StageNet, StageAttentionNet, Transformer, EHRMamba |
| [`drug_recommendation/`](drug_recommendation/) | GAMENet, SafeDrug, MoleRec, MICRON, RETAIN, AdaCare, Transformer on MIMIC-III/IV, eICU and OMOP; Optuna tuning |
| [`clinical_tasks/`](clinical_tasks/) | Diabetic ketoacidosis prediction on MIMIC-IV (all diabetes and type 1 only) |
| [`interpretability/`](interpretability/) | Integrated Gradients, DeepLift, GIM, SHAP, LIME, attention methods, and faithfulness metrics (comprehensiveness and sufficiency) |
| [`cxr/`](cxr/) | Chest X-ray classification (ChestX-ray14, COVID-19), saliency maps, conformal prediction, ViT, VAE and GAN image generation |
| [`eeg/`](eeg/) | EEG events and abnormality (TUEV, TUAB), sleep staging (Sleep-EDF), EEGBCI pattern discovery; SparcNet, ContraWR, BIOT, TFM-Tokenizer |
| [`conformal_eeg/`](conformal_eeg/) | Conformal prediction on TUEV/TUAB: conventional, k-means cluster, neighbourhood (NCP), covariate shift |
| [`ecg/`](ecg/) | PTB-XL multi-label ECG classification, including ProtoECGNet |
| [`benchmark_perf/`](benchmark_perf/README.md) | Speed and memory benchmarks against PyHealth 1.x, pandas and meds_reader |

## Top-level notebooks

| Notebook | Shows |
|---|---|
| `datasets_overview.ipynb` | Loading MIMIC-III, MIMIC-IV and other datasets; what a `SampleDataset` holds |
| `tutorial_stagenet_comprehensive.ipynb` | StageNet end to end on MIMIC-IV, with Integrated Gradients and DeepLift |
| `transformer_mimic4.ipynb`, `cnn_mimic4.ipynb`, `deepr_mimic4.ipynb`, `agent_mimic4.ipynb`, `concare_mimic4_example.ipynb`, `gat_mimic4.ipynb`, `gcn_mimic4.ipynb`, `ehrmamba_mimic4.ipynb` | One model each, mortality on MIMIC-IV |
| `gamenet_mimic3.ipynb`, `molerec_mimic3.ipynb`, `tcn_mimic3_codes.ipynb` | Drug recommendation and code-based models on MIMIC-III |
| `medlink_mimic3.ipynb` | MedLink patient record linkage |
| `graphcare_tutorial.ipynb` | GraphCare on synthetic data (needs `torch-geometric`) |
| `benchmark_ehrshot_xgboost.ipynb` | EHRShot benchmark tasks with XGBoost |
| `synthea_end_to_end.ipynb` | Generating Synthea data and loading it into PyHealth (needs Java) |
| `smart_processor_clinical_text_tutorial.ipynb`, `text_embedding_tutorial.ipynb` | Clinical text processors and text embeddings |
| `time_image_processor_tutorial.ipynb`, `vision_embedding_tutorial.ipynb` | Image processors and image embeddings |
| `sparcnet_sleepedf.ipynb` | SparcNet on Sleep-EDF |

## Top-level scripts

**Data and processing**

| Script | Shows |
|---|---|
| `split_aware_set_task_mimic3demo.py` | `set_task(task, split=...)`: fit processors on training patients only |
| `reuse_train_processors.py`, `nested_sequence_fit_on_train.py` | Fitting processors on training data and reusing them |
| `processor_vocab_editing.py`, `custom_tensor_processor.py`, `label_schema_forms.py` | Customising processors and output schemas |
| `split_by_patient_order.py`, `split_ratio_validation.py`, `sample_dataset_subset.py` | Patient splits and dataset subsets |
| `event_timestamps_mimic3demo.py` | Events with and without timestamps |
| `dataset_cache_refresh.py`, `dataset_cache_shared_tmpdir.py`, `raw_fields_disk_cache.py`, `require_cache_dir.py` | How dataset caching works, and keeping caches in controlled storage |
| `omop_dataset_demo.py`, `meds_demo.py`, `verify_meds_mortality.py`, `mimic4fhir_mpf_ehrmamba.py` | OMOP, MEDS and FHIR data |
| `medcode.py`, `medcode_icd_mappings.py` | Medical code lookup, ICD translation and groupers |
| `survival_preprocess_support2_demo.py` | SUPPORT2 survival preprocessing |
| `logging_setup.py` | Controlling PyHealth's logging |

**Training, evaluation and calibration**

| Script | Shows |
|---|---|
| `reproducible_training.py` | Seeding for identical results |
| `imbalanced_outcome_pos_weight.py` | Class weighting for rare outcomes |
| `calibration_and_bootstrap_ci.py`, `ece_metrics.py`, `logistic_recalibration_new_site.py`, `mimic4_califorest.py` | Calibration metrics, bootstrap confidence intervals, recalibration |
| `label_shift_conformal.py` | Conformal prediction under label shift |
| `multiclass_ranking_metrics.py` | Ranking metrics for multiclass models |
| `lime_stagenet_mimic4.py` | LIME explanations (more methods in `interpretability/`) |
| `autoimprove_synthetic_mimic3.py`, `autoimprove_eicu_ihm48.py` | Automatic pipeline improvement with `AutoImprover` |

**Models**

| Script | Shows |
|---|---|
| `halo_mimic3.py`, `medgan_mimic3.py`, `gpt2_mimic3.py` | Synthetic EHR generation |
| `patient_linkage_mimic3_medlink.py` | MedLink record linkage (needs `pytrec_eval`) |
| `physionet_deid_ner_transformer_deid.py` | Transformer de-identification (PhysioNet, credentialed) |
| `mimic4_mortality_medfuse.py` | MedFuse ablation (synthetic data) |
| `support_survival_survival_mdn.py` (with `survival_mdn_data.py`, `survival_mdn_metrics.py`, results in `survival_mdn_results/`) | Survival mixture density network on SUPPORT |
| `yaib_mimic4_rnn_v2.py` | RNN_V2 on the five MIMIC-IV YAIB tasks |
