Development logs
======================
We track the new development here:

**Dec 29, 2023**

..code-blocks:: rst

    1. add GAN models and demos in pyhealth PR #256
    2. 

**Dec 20, 2023**

..code-blocks:: rst

    1. add graph neural network models for drug recommendation in PR #251
    2. add regression_metrics_fn for some unsupervised learning tasks.
    3. add VAE models in pyhealth PR #253

**Nov 28, 2023**

..code-blocks:: rst

    1. fix ddi metric calculation error raised by issue #249.

**Nov 24, 2023**

..code-blocks:: rst

    1. add ddi metrics to multilabel_metrics_fn per issue #247
    2. add medlink as a new medical record link pipeline in PR #240
    3. add chefer transformer in PR #239
    4. add KG embedding in PR #234
    5. add LLM for Pyhealth in PR #231

**Sep 01, 2023**

.. code-block:: rst

    1. add Base fairness metrics and example `#216`.

**July 22, 2023**

.. code-block:: rst

    1. add Temple University TUEV and TUAB two datasets in `#194`.
    2. add the EEG six event detection and abnormal EEG detection tasks.


**July 1, 2023**

.. code-block:: rst

    1. add six ECG datasets: "cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl", "st_petersburg_incart" (from 
        PhysioNet Cardiology Challenge 2020 https://physionet.org/content/challenge-2020/1.0.2/ `#176`
    2. add ECG binary classification tasks (for five symptom categories: Arrhythmias symptom, Bundle branch blocks and fascicular blocks symptom, 
        Axis deviations symptom, Conduction delays symptom, Wave abnormalities symptom) `#176`


**May 31, 2023**

.. code-block:: rst

    1. add SHHS dataset and its sleep staging task.

**May 25, 2023**

.. code-block:: rst

    1. add dirichlet calibration `PR #159`

**May 9, 2023**

.. code-block:: rst

    1. add MIMIC-Extract dataset  `#136`
    2. add new maintainer members for pyhealth: Junyi Gao and Benjamin Danek

**May 6, 2023**

.. code-block:: rst

    1. add new parser functions (admissionDx, diagnosisStrings) and prediction tasks for eICU dataset `#148`

**Apr 27, 2023**

.. code-block:: rst

    1. add MoleRec model (WWW'23) for drug recommendation `#122`

**Apr 26, 2023**

.. code-block:: rst
 
    1. fix bugs in GRASP model `#141`
    2. add pandas install <2 constraints `#135` 
    3. add hcpcsevents table process in MIMIC4 dataset `#134`
    
**Apr 10, 2023**

.. code-block:: rst

    1. fix Ambiguous datetime usage in eICU (https://github.com/sunlabuiuc/PyHealth/pull/132)

**Mar 26, 2023**    

.. code-block:: rst

    1. add the entire uncertainty quantification module (https://github.com/sunlabuiuc/PyHealth/pull/111)

**Feb 26, 2023**

.. code-block:: rst
 
    1. add 6 EHR predictiom model: Adacare, Concare, Stagenet, TCN, Grasp, Agent

**Feb 24, 2023**

.. code-block:: rst
 
    1. add unittest for omop dataset
    2. add github action triggered manually, check `#104`

**Feb 19, 2023**

.. code-block:: rst
 
    1. add unittest for eicu dataset
    2. add ISRUC dataset (and task function) for signal learning

**Feb 12, 2023**

.. code-block:: rst
 
    1. add unittest for mimiciii, mimiciv
    2. add SHHS datasets for sleep staging task
    3. add SparcNet model for signal classification task

**Feb 08, 2023**

.. code-block:: rst
 
    1. complete the biosignal data support, add ContraWR [1] model for general purpose biosignal classification task ([1] Yang, Chaoqi, Danica Xiao, M. Brandon Westover, and Jimeng Sun. 
        "Self-supervised eeg representation learning for automatic sleep staging."
        arXiv preprint arXiv:2110.15278 (2021).)

**Feb 07, 2023**

.. code-block:: rst
 
    1. Support signal dataset processing and split: add SampleSignalDataset, BaseSignalDataset. Use SleepEDFcassette dataset as the first signal dataset. Use example/sleep_staging_sleepEDF_contrawr.py
    2. rename the dataset/ parts: previous BaseDataset becomes BaseEHRDataset and SampleDatast becomes SampleEHRDataset. Right now, BaseDataset will be inherited by BaseEHRDataset and BaseSignalDataset. SampleBaseDataset will be inherited by SampleEHRDataset and SampleSignalDataset.

**Feb 06, 2023**

.. code-block:: rst
 
    1. improve readme style
    2. add the pyhealth live 06 and 07 link to pyhealth live

**Feb 01, 2023**

.. code-block:: rst
 
    1. add unittest of PyHealth MedCode and Tokenizer

**Jan 26, 2023**

.. code-block:: rst
 
    1. accelerate MIMIC-IV, eICU and OMOP data loading by using multiprocessing (pandarallel)

**Jan 25, 2023**

.. code-block:: rst

    1. accelerate the MIMIC-III data loading process by using multiprocessing (pandarallel)

**Jan 24, 2023**

.. code-block:: rst

    1. Fix the code typo in pyhealth/tasks/drug_recommendation.py for issue `#71`.
    2. update the pyhealth live schedule 

**Jan 22, 2023**

.. code-block:: rst

    1. Fix the list of list of vector problem in RNN, Transformer, RETAIN, and CNN
    2. Add initialization examples for RNN, Transformer, RETAIN, CNN, and Deepr
    3. (minor) change the parameters from "Type" and "level" to "type_" and "dim_"
    4. BPDanek adds the "__repr__" function to medcode for better print understanding
    5. add unittest for pyhealth.data

**Jan 21, 2023**

.. code-block:: rst

    1. Added a new model, Deepr (models.Deepr)

**Jan 20, 2023**

.. code-block:: rst

    1. add the pyhealth live 05
    2. add slack channel invitation in pyhealth live page

**Jan 13, 2023**

.. code-block:: rst

    1. add the pyhealth live 03 and 04 video link to the nagivation
    2. add future pyhealth live schedule

**Jan 8, 2023**

.. code-block:: rst

    1. Changed BaseModel.add_feature_transform_layer in models/base_model.py so that it accepts special_tokens if necessary
    2. fix an int/float bug in dataset checking (transform int to float and then process them uniformly)

**Dec 26, 2022**

.. code-block:: rst

    1. add examples to pyhealth.data, pyhealth.datasets
    2. improve jupyter notebook tutorials 0, 1, 2


**Dec 21, 2022**

.. code-block:: rst

    1. add the development logs to the navigation
    2. add the pyhealth live schedule to the nagivation
