Tutorials
========================

 We provide the following tutorials to help users get started with our pyhealth. Please bear with us as we update the documentation on how to use pyhealth 2.0.


`Tutorial 0: Introduction to pyhealth.data <https://colab.research.google.com/drive/17nOzjIjKiAbC8bsntZ3h9xy2Vq4bKpuv?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_data.ipynb>`__

`Tutorial 1: Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1vI_oljc7rU5ocsC26ITM7HUgD5SGZkFE?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/orig_tutorial_pyhealth_datasets.ipynb>`__

`Tutorial 2: Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1QB0acnGb-wOuK53UNSgHxjCW74QeYjUl?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=CxESe1gYWU4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=4>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/orig_tutorial_pyhealth_tasks.ipynb>`__

`Tutorial 3: Introduction to pyhealth.models <https://colab.research.google.com/drive/1cUTSfFL1wLUXDBtJGTAWntolvcmxrDGo?usp=drive_link>`_  `[Video] <https://www.youtube.com/watch?v=fRc0ncbTgZA&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=6>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_model.ipynb>`__

`Tutorial 4: Introduction to pyhealth.trainer <https://colab.research.google.com/drive/1up_SL0BxxHPO9pmjKQ98w1GbpiB7LySp?usp=drive_link>`_  `[Video] <https://www.youtube.com/watch?v=5Hyw3of5pO4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=7>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_trainer.ipynb>`__

`Tutorial 5: Introduction to pyhealth.metrics <https://colab.research.google.com/drive/1bO0h5BR62_kQ7zFOgzQmt5vb8jqJ0rV-?usp=drive_link>`_  `[Video] <https://www.youtube.com/watch?v=d-Kx_xCwre4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=8>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_metrics.ipynb>`__

`Tutorial 6: Introduction to pyhealth.tokenizer <https://colab.research.google.com/drive/1jhJ11MLUafhflQAz8HSrWiOEYlIhhvc_?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=CeXJtf0lfs0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=10>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_tokenizer.ipynb>`__

`Tutorial 7: Introduction to pyhealth.medcode <https://colab.research.google.com/drive/1Tw1AUS53fotH1EYr4Abp7qYN3zDBeUbC?usp=drive_link>`_ `[Video] <https://www.youtube.com/watch?v=MmmfU6_xkYg&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=9>`__ `[Source] <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorials/tutorial_pyhealth_medcode.ipynb>`__


Data Access Guide
=======================

For information on how to access and download the datasets supported by PyHealth, please refer to our `Datasets Overview Notebook <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/datasets_overview.ipynb>`_.

Additionally, for detailed tutorials on accessing PhysioNet and MIMIC datasets, see the `Getting MIMIC access` section of the `DL4H course instructions <https://docs.google.com/document/d/1NHgXzSPINafSg8Cd_whdfSauFXgh-ZflZIw5lu6k2T0/edit?tab=t.5pba851jxeg6>`_.


`Pipeline 1: Chest Xray Classification <https://colab.research.google.com/drive/18vK23gyI1LjWbTgkq4f99yDZA3A7Pxp9?usp=sharing>`_ 

`Pipeline 2: Medical Coding <https://colab.research.google.com/drive/1ThYP_5ng5xPQwscv5XztefkkoTruhjeK?usp=sharing>`_ 

`Pipeline 3: Medical Transcription Classification <https://colab.research.google.com/drive/1bjk_IArc2ZmXGR6u6Qzyf7kh70RdiY9c?usp=sharing>`_ 

`Pipeline 4: Mortality Prediction <https://colab.research.google.com/drive/1b9xRbxUz-HLzxsrvxdsdJ868ajGQCY6U?usp=sharing>`_ 

`Pipeline 5: Readmission Prediction <https://colab.research.google.com/drive/1h0pAymUlPQfkLFryI9QI37-HAW1tRxGZ?usp=sharing>`_ 

.. `Pipeline 5: Phenotype Prediction <https://colab.research.google.com/drive/10CSb4F4llYJvv42yTUiRmvSZdoEsbmFF>`_

Multimodal & Smart Processors
------------------------------

These examples demonstrate PyHealth's unified multimodal architecture using
:class:`~pyhealth.processors.TemporalFeatureProcessor` subclasses and
:class:`~pyhealth.models.UnifiedMultimodalEmbeddingModel`.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Notebook / File
     - Description
   * - `smart_processor_clinical_text_tutorial.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/smart_processor_clinical_text_tutorial.ipynb>`_
     - End-to-end tutorial: HuggingFace tokenizer inside ``TupleTimeTextProcessor``,
       canonical ``("tuple_time_text", kwargs)`` schema form, EmbeddingModel with 3D inputs,
       and gradient flow through BERT-tiny in ``MLP``, ``Transformer``, ``RNN``, ``MultimodalRNN``
   * - ``examples/`` (see ``time_image_processor``\* files)
     - ``TimeImageProcessor`` for serial chest X-rays with timestamps

**Key APIs:**

- :class:`~pyhealth.processors.TemporalFeatureProcessor` — ABC for all temporal processors
- :class:`~pyhealth.processors.ModalityType` — ``CODE / TEXT / IMAGE / NUMERIC / AUDIO / SIGNAL``
- :class:`~pyhealth.processors.TemporalTimeseriesProcessor` — timeseries with preserved timestamps
- :func:`~pyhealth.datasets.collate_temporal` — universal DataLoader collator for dict-output processors
- :class:`~pyhealth.models.UnifiedMultimodalEmbeddingModel` — temporally-aligned multimodal sequence embeddings


----------

Additional Examples
===================

.. warning::
   **Compatibility Notice**: Not all examples below have been updated to PyHealth 2.0. However, they remain useful references for understanding workflows and implementation patterns. If you encounter compatibility issues, please refer to the tutorials above or consult the updated API documentation.

The ``examples/`` directory contains additional code examples demonstrating various tasks, models, and techniques. These examples show how to use PyHealth in real-world scenarios.

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

Mortality Prediction
--------------------

These examples are located in ``examples/mortality_prediction/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `mortality_prediction/mortality_mimic3_rnn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_rnn.py>`_
     - RNN for mortality prediction on MIMIC-III
   * - `mortality_prediction/mortality_mimic3_stagenet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_stagenet.py>`_
     - StageNet for mortality prediction on MIMIC-III
   * - `mortality_prediction/mortality_mimic3_adacare.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_adacare.ipynb>`_
     - AdaCare for mortality prediction on MIMIC-III (notebook)
   * - `mortality_prediction/mortality_mimic3_agent.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_agent.py>`_
     - Agent model for mortality prediction on MIMIC-III
   * - `mortality_prediction/mortality_mimic3_concare.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_concare.py>`_
     - ConCare for mortality prediction on MIMIC-III
   * - `mortality_prediction/mortality_mimic3_grasp.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_grasp.py>`_
     - GRASP for mortality prediction on MIMIC-III
   * - `mortality_prediction/mortality_mimic3_tcn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic3_tcn.py>`_
     - Temporal Convolutional Network for mortality prediction
   * - `mortality_prediction/mortality_mimic4_stagenet_v2.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mortality_mimic4_stagenet_v2.py>`_
     - StageNet for mortality prediction on MIMIC-IV (v2)
   * - `mortality_prediction/timeseries_mimic4.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/timeseries_mimic4.py>`_
     - Time series analysis on MIMIC-IV

Readmission Prediction
----------------------

These examples are located in ``examples/readmission/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `readmission/readmission_mimic3_rnn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/readmission/readmission_mimic3_rnn.py>`_
     - RNN for readmission prediction on MIMIC-III
   * - `readmission/readmission_mimic3_fairness.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/readmission/readmission_mimic3_fairness.py>`_
     - Fairness-aware readmission prediction on MIMIC-III
   * - `readmission/readmission_omop_rnn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/readmission/readmission_omop_rnn.py>`_
     - RNN for readmission prediction on OMOP dataset

Survival Prediction
-------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `survival_preprocess_support2_demo.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/survival_preprocess_support2_demo.py>`_
     - Survival probability prediction preprocessing with SUPPORT2 dataset. Demonstrates feature extraction (demographics, vitals, labs, scores, comorbidities) and ground truth survival probability labels for 2-month and 6-month horizons. Shows how to decode processed tensors back to human-readable features.

Drug Recommendation
-------------------

These examples are located in ``examples/drug_recommendation/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `drug_recommendation/drug_recommendation_mimic3_safedrug.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_safedrug.py>`_
     - SafeDrug for drug recommendation on MIMIC-III
   * - `drug_recommendation/drug_recommendation_mimic3_molerec.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_molerec.py>`_
     - MoleRec for drug recommendation on MIMIC-III
   * - `drug_recommendation/drug_recommendation_mimic3_gamenet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_gamenet.py>`_
     - GAMENet for drug recommendation on MIMIC-III
   * - `drug_recommendation/drug_recommendation_mimic3_transformer.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_transformer.py>`_
     - Transformer for drug recommendation on MIMIC-III
   * - `drug_recommendation/drug_recommendation_mimic3_micron.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_micron.py>`_
     - MICRON for drug recommendation on MIMIC-III
   * - `drug_recommendation/drug_recommendation_mimic4_gamenet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic4_gamenet.py>`_
     - GAMENet for drug recommendation on MIMIC-IV
   * - `drug_recommendation/drug_recommendation_mimic4_retain.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic4_retain.py>`_
     - RETAIN for drug recommendation on MIMIC-IV
   * - `drug_recommendation/drug_recommendation_eicu_transformer.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_eicu_transformer.py>`_
     - Transformer for drug recommendation on eICU

EEG and Sleep Analysis
----------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `eeg/sleep_staging/sleep_staging_sleepEDF_contrawr.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/eeg/sleep_staging/sleep_staging_sleepEDF_contrawr.py>`_
     - ContraWR for sleep staging on SleepEDF
   * - `eeg/sleep_staging/sleep_staging_shhs_contrawr.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/eeg/sleep_staging/sleep_staging_shhs_contrawr.py>`_
     - ContraWR for sleep staging on SHHS
   * - `eeg/sleep_staging/sleep_staging_ISRUC_SparcNet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/eeg/sleep_staging/sleep_staging_ISRUC_SparcNet.py>`_
     - SparcNet for sleep staging on ISRUC
   * - `eeg/eeg_models/SparcNet_eeg_events_classification.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/eeg/eeg_models/SparcNet_eeg_events_classification.py>`_
     - SparcNet for EEG event detection
   * - `eeg/eeg_models/SparcNet_eeg_abnormal_classification.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/eeg/eeg_models/SparcNet_eeg_abnormal_classification.py>`_
     - SparcNet for EEG abnormality detection
   * - `cardiology_detection_isAR_SparcNet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cardiology_detection_isAR_SparcNet.py>`_
     - SparcNet for cardiology arrhythmia detection

Image Analysis (Chest X-Ray)
----------------------------

These examples are located in ``examples/cxr/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `cxr/covid19cxr_tutorial.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/covid19cxr_tutorial.py>`_
     - ViT training, conformal prediction & interpretability for COVID-19 CXR
   * - `cxr/covid19cxr_conformal.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/covid19cxr_conformal.py>`_
     - Conformal prediction for COVID-19 CXR classification
   * - `cxr/cnn_cxr.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/cnn_cxr.ipynb>`_
     - CNN for chest X-ray classification (notebook)
   * - `cxr/chestxray14_binary_classification.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/chestxray14_binary_classification.ipynb>`_
     - Binary classification on ChestX-ray14 dataset (notebook)
   * - `cxr/chestxray14_multilabel_classification.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/chestxray14_multilabel_classification.ipynb>`_
     - Multi-label classification on ChestX-ray14 dataset (notebook)
   * - `cxr/ChestXrayClassificationWithSaliency.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/ChestXrayClassificationWithSaliency.ipynb>`_
     - Chest X-ray classification with saliency maps (notebook)
   * - `cxr/chextXray_image_generation_VAE.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/chextXray_image_generation_VAE.py>`_
     - VAE for chest X-ray image generation
   * - `cxr/ChestXray-image-generation-GAN.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cxr/ChestXray-image-generation-GAN.ipynb>`_
     - GAN for chest X-ray image generation (notebook)

Interpretability
----------------

These examples are located in ``examples/interpretability/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `interpretability/integrated_gradients_mortality_mimic4_stagenet.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/integrated_gradients_mortality_mimic4_stagenet.py>`_
     - Integrated Gradients for StageNet interpretability
   * - `interpretability/deeplift_stagenet_mimic4.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/deeplift_stagenet_mimic4.py>`_
     - DeepLift attributions for StageNet on MIMIC-IV
   * - `interpretability/gim_stagenet_mimic4.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/gim_stagenet_mimic4.py>`_
     - GIM attributions for StageNet on MIMIC-IV
   * - `interpretability/gim_transformer_mimic4.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/gim_transformer_mimic4.py>`_
     - GIM attributions for Transformer on MIMIC-IV
   * - `interpretability/shap_stagenet_mimic4.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/shap_stagenet_mimic4.py>`_
     - SHAP attributions for StageNet on MIMIC-IV
   * - `interpretability/interpretability_metrics.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/interpretability_metrics.py>`_
     - Evaluating attribution methods with metrics
   * - `interpretability/interpret_demo.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/interpret_demo.ipynb>`_
     - Interactive interpretability demonstrations (notebook)
   * - `interpretability/shap_stagenet_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/interpretability/shap_stagenet_mimic4.ipynb>`_
     - SHAP attributions for StageNet (notebook)

Patient Linkage
---------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `patient_linkage_mimic3_medlink.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/patient_linkage_mimic3_medlink.py>`_
     - MedLink for patient record linkage on MIMIC-III

Length of Stay
--------------

These examples are located in ``examples/length_of_stay/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `length_of_stay/length_of_stay_mimic3_rnn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/length_of_stay/length_of_stay_mimic3_rnn.py>`_
     - RNN for length of stay prediction on MIMIC-III
   * - `length_of_stay/length_of_stay_mimic4_rnn.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/length_of_stay/length_of_stay_mimic4_rnn.py>`_
     - RNN for length of stay prediction on MIMIC-IV

Advanced Topics
---------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - `omop_dataset_demo.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/omop_dataset_demo.py>`_
     - Working with OMOP Common Data Model
   * - `medcode.py <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/medcode.py>`_
     - Medical code vocabulary and mappings
   * - `benchmark_ehrshot_xgboost.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/benchmark_ehrshot_xgboost.ipynb>`_
     - EHRShot benchmark with XGBoost (notebook)

Notebooks (Interactive)
------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Notebook File
     - Description
   * - `tutorial_stagenet_comprehensive.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/tutorial_stagenet_comprehensive.ipynb>`_
     - Comprehensive StageNet tutorial
   * - `mortality_prediction/mimic3_mortality_prediction_cached.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/mimic3_mortality_prediction_cached.ipynb>`_
     - Cached mortality prediction workflow
   * - `mortality_prediction/timeseries_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/timeseries_mimic4.ipynb>`_
     - Time series analysis on MIMIC-IV
   * - `transformer_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/transformer_mimic4.ipynb>`_
     - Transformer models on MIMIC-IV
   * - `cnn_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/cnn_mimic4.ipynb>`_
     - CNN models on MIMIC-IV
   * - `gat_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/gat_mimic4.ipynb>`_
     - Graph Attention Networks on MIMIC-IV
   * - `gcn_mimic4.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/gcn_mimic4.ipynb>`_
     - Graph Convolutional Networks on MIMIC-IV
   * - `drug_recommendation/safedrug_mimic3.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/safedrug_mimic3.ipynb>`_
     - SafeDrug interactive notebook
   * - `molerec_mimic3.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/molerec_mimic3.ipynb>`_
     - MoleRec interactive notebook
   * - `drug_recommendation/drug_recommendation_mimic3_micron.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/drug_recommendation/drug_recommendation_mimic3_micron.ipynb>`_
     - MICRON interactive notebook
   * - `kg_embedding.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/kg_embedding.ipynb>`_
     - Knowledge graph embeddings
   * - `lm_embedding_huggingface.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/lm_embedding_huggingface.ipynb>`_
     - Language model embeddings with HuggingFace
   * - `lm_embedding_openai.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/lm_embedding_openai.ipynb>`_
     - Language model embeddings with OpenAI
   * - `prepare_mapping.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/prepare_mapping.ipynb>`_
     - Data preprocessing and mapping utilities
   * - `graph_torchvision_model.ipynb <https://github.com/sunlabuiuc/PyHealth/blob/master/examples/graph_torchvision_model.ipynb>`_
     - Using Torchvision models with graph data
