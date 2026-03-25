Tutorials
========================

 We provide the following tutorials to help users get started with our pyhealth. Please bear with us as we update the documentation on how to use pyhealth 2.0.


`Tutorial 0: Introduction to pyhealth.data <https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2>`_ 

`Tutorial 1: Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1voSx7wEfzXfEf2sIfW6b-8p1KqMyuWxK?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`_ 

`Tutorial 2: Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1kKkkBVS_GclHoYTbnOtjyYnSee79hsyT?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=CxESe1gYWU4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=4>`_ 

`Tutorial 3: Introduction to pyhealth.models <https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=fRc0ncbTgZA&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=6>`_ 

`Tutorial 4: Introduction to pyhealth.trainer <https://colab.research.google.com/drive/1L1Nz76cRNB7wTp5Pz_4Vp4N2eRZ9R6xl?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=5Hyw3of5pO4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=7>`_ 

`Tutorial 5: Introduction to pyhealth.metrics <https://colab.research.google.com/drive/1Mrs77EJ92HwMgDaElJ_CBXbi4iABZBeo?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=d-Kx_xCwre4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=8>`_

`Tutorial 6: Introduction to pyhealth.tokenizer <https://colab.research.google.com/drive/1bDOb0A5g0umBjtz8NIp4wqye7taJ03D0?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=CeXJtf0lfs0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=10>`_

`Tutorial 7: Introduction to pyhealth.medcode <https://colab.research.google.com/drive/1xrp_ACM2_Hg5Wxzj0SKKKgZfMY0WwEj3?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=MmmfU6_xkYg&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=9>`_


Data Access Guide
=======================

For information on how to access and download the datasets supported by PyHealth, please refer to our `Datasets Overview Notebook <https://github.com/sunlabuiuc/PyHealth/blob/main/examples/datasets_overview.ipynb>`_.

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
   * - ``smart_processor_clinical_text_tutorial.ipynb``
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
   * - ``mortality_prediction/mortality_mimic3_rnn.py``
     - RNN for mortality prediction on MIMIC-III
   * - ``mortality_prediction/mortality_mimic3_stagenet.py``
     - StageNet for mortality prediction on MIMIC-III
   * - ``mortality_prediction/mortality_mimic3_adacare.ipynb``
     - AdaCare for mortality prediction on MIMIC-III (notebook)
   * - ``mortality_prediction/mortality_mimic3_agent.py``
     - Agent model for mortality prediction on MIMIC-III
   * - ``mortality_prediction/mortality_mimic3_concare.py``
     - ConCare for mortality prediction on MIMIC-III
   * - ``mortality_prediction/mortality_mimic3_grasp.py``
     - GRASP for mortality prediction on MIMIC-III
   * - ``mortality_prediction/mortality_mimic3_tcn.py``
     - Temporal Convolutional Network for mortality prediction
   * - ``mortality_prediction/mortality_mimic4_stagenet_v2.py``
     - StageNet for mortality prediction on MIMIC-IV (v2)
   * - ``mortality_prediction/timeseries_mimic4.py``
     - Time series analysis on MIMIC-IV

Readmission Prediction
----------------------

These examples are located in ``examples/readmission/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``readmission/readmission_mimic3_rnn.py``
     - RNN for readmission prediction on MIMIC-III
   * - ``readmission/readmission_mimic3_fairness.py``
     - Fairness-aware readmission prediction on MIMIC-III
    * - ``readmission/readmission_omop_rnn.py``
     - RNN for readmission prediction on OMOP dataset

Survival Prediction
-------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``survival_preprocess_support2_demo.py``
     - Survival probability prediction preprocessing with SUPPORT2 dataset. Demonstrates feature extraction (demographics, vitals, labs, scores, comorbidities) and ground truth survival probability labels for 2-month and 6-month horizons. Shows how to decode processed tensors back to human-readable features.

Drug Recommendation
-------------------

These examples are located in ``examples/drug_recommendation/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``drug_recommendation/drug_recommendation_mimic3_safedrug.py``
     - SafeDrug for drug recommendation on MIMIC-III
   * - ``drug_recommendation/drug_recommendation_mimic3_molerec.py``
     - MoleRec for drug recommendation on MIMIC-III
   * - ``drug_recommendation/drug_recommendation_mimic3_gamenet.py``
     - GAMENet for drug recommendation on MIMIC-III
   * - ``drug_recommendation/drug_recommendation_mimic3_transformer.py``
     - Transformer for drug recommendation on MIMIC-III
   * - ``drug_recommendation/drug_recommendation_mimic3_micron.py``
     - MICRON for drug recommendation on MIMIC-III
   * - ``drug_recommendation/drug_recommendation_mimic4_gamenet.py``
     - GAMENet for drug recommendation on MIMIC-IV
   * - ``drug_recommendation/drug_recommendation_mimic4_retain.py``
     - RETAIN for drug recommendation on MIMIC-IV
   * - ``drug_recommendation/drug_recommendation_eICU_transformer.py``
     - Transformer for drug recommendation on eICU

EEG and Sleep Analysis
----------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``sleep_staging_sleepEDF_contrawr.py``
     - ContraWR for sleep staging on SleepEDF
   * - ``sleep_staging_shhs_contrawr.py``
     - ContraWR for sleep staging on SHHS
   * - ``sleep_staging_ISRUC_SparcNet.py``
     - SparcNet for sleep staging on ISRUC
   * - ``EEG_events_SparcNet.py``
     - SparcNet for EEG event detection
   * - ``EEG_isAbnormal_SparcNet.py``
     - SparcNet for EEG abnormality detection
   * - ``cardiology_detection_isAR_SparcNet.py``
     - SparcNet for cardiology arrhythmia detection

Image Analysis (Chest X-Ray)
----------------------------

These examples are located in ``examples/cxr/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``cxr/covid19cxr_tutorial.py``
     - ViT training, conformal prediction & interpretability for COVID-19 CXR
   * - ``cxr/covid19cxr_conformal.py``
     - Conformal prediction for COVID-19 CXR classification
   * - ``cxr/cnn_cxr.ipynb``
     - CNN for chest X-ray classification (notebook)
   * - ``cxr/chestxray14_binary_classification.ipynb``
     - Binary classification on ChestX-ray14 dataset (notebook)
   * - ``cxr/chestxray14_multilabel_classification.ipynb``
     - Multi-label classification on ChestX-ray14 dataset (notebook)
   * - ``cxr/ChestXrayClassificationWithSaliency.ipynb``
     - Chest X-ray classification with saliency maps (notebook)
   * - ``cxr/chextXray_image_generation_VAE.py``
     - VAE for chest X-ray image generation
   * - ``cxr/ChestXray-image-generation-GAN.ipynb``
     - GAN for chest X-ray image generation (notebook)

Interpretability
----------------

These examples are located in ``examples/interpretability/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``integrated_gradients_mortality_mimic4_stagenet.py``
     - Integrated Gradients for StageNet interpretability
   * - ``interpretability/deeplift_stagenet_mimic4.py``
     - DeepLift attributions for StageNet on MIMIC-IV
   * - ``interpretability/gim_stagenet_mimic4.py``
     - GIM attributions for StageNet on MIMIC-IV
   * - ``interpretability/gim_transformer_mimic4.py``
     - GIM attributions for Transformer on MIMIC-IV
   * - ``interpretability/shap_stagenet_mimic4.py``
     - SHAP attributions for StageNet on MIMIC-IV
   * - ``interpretability/interpretability_metrics.py``
     - Evaluating attribution methods with metrics
   * - ``interpretability/interpret_demo.ipynb``
     - Interactive interpretability demonstrations (notebook)
   * - ``interpretability/shap_stagenet_mimic4.ipynb``
     - SHAP attributions for StageNet (notebook)

Patient Linkage
---------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``patient_linkage_mimic3_medlink.py``
     - MedLink for patient record linkage on MIMIC-III

Length of Stay
--------------

These examples are located in ``examples/length_of_stay/``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``length_of_stay/length_of_stay_mimic3_rnn.py``
     - RNN for length of stay prediction on MIMIC-III
   * - ``length_of_stay/length_of_stay_mimic4_rnn.py``
     - RNN for length of stay prediction on MIMIC-IV

Advanced Topics
---------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Example File
     - Description
   * - ``omop_dataset_demo.py``
     - Working with OMOP Common Data Model
   * - ``medcode.py``
     - Medical code vocabulary and mappings
   * - ``benchmark_ehrshot_xgboost.ipynb``
     - EHRShot benchmark with XGBoost (notebook)

Notebooks (Interactive)
------------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Notebook File
     - Description
   * - ``tutorial_stagenet_comprehensive.ipynb``
     - Comprehensive StageNet tutorial
   * - ``mimic3_mortality_prediction_cached.ipynb``
     - Cached mortality prediction workflow
   * - ``mortality_prediction/timeseries_mimic4.ipynb``
     - Time series analysis on MIMIC-IV
   * - ``transformer_mimic4.ipynb``
     - Transformer models on MIMIC-IV
   * - ``cnn_mimic4.ipynb``
     - CNN models on MIMIC-IV
   * - ``gat_mimic4.ipynb``
     - Graph Attention Networks on MIMIC-IV
   * - ``gcn_mimic4.ipynb``
     - Graph Convolutional Networks on MIMIC-IV
   * - ``safedrug_mimic3.ipynb``
     - SafeDrug interactive notebook
   * - ``molerec_mimic3.ipynb``
     - MoleRec interactive notebook
   * - ``drug_recommendation/drug_recommendation_mimic3_micron.ipynb``
     - MICRON interactive notebook
   * - ``kg_embedding.ipynb``
     - Knowledge graph embeddings
   * - ``lm_embedding_huggingface.ipynb``
     - Language model embeddings with HuggingFace
   * - ``lm_embedding_openai.ipynb``
     - Language model embeddings with OpenAI
   * - ``prepare_mapping.ipynb``
     - Data preprocessing and mapping utilities
   * - ``graph_torchvision_model.ipynb``
     - Using Torchvision models with graph data


----------
