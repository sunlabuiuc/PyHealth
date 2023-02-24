Development logs
======================
We track the new development here:


**Feb 24, 2023**

.. code-block:: bash
 
    1. add unittest for omop dataset
    2. add github action triggered manually, check #104

**Feb 19, 2023**

.. code-block:: bash
 
    1. add unittest for eicu dataset
    2. add ISRUC dataset (and task function) for signal learning

**Feb 12, 2023**

.. code-block:: bash
 
    1. add unittest for mimiciii, mimiciv
    2. add SHHS datasets for sleep staging task
    3. add SparcNet model for signal classification task

**Feb 08, 2023**

.. code-block:: bash
 
    1. complete the biosignal data support, add ContraWR [1] model for general purpose biosignal classification task ([1] Yang, Chaoqi, Danica Xiao, M. Brandon Westover, and Jimeng Sun. 
        "Self-supervised eeg representation learning for automatic sleep staging."
        arXiv preprint arXiv:2110.15278 (2021).)

**Feb 07, 2023**

.. code-block:: bash
 
    1. Support signal dataset processing and split: add SampleSignalDataset, BaseSignalDataset. Use SleepEDFcassette dataset as the first signal dataset. Use example/sleep_staging_sleepEDF_contrawr.py
    2. rename the dataset/ parts: previous BaseDataset becomes BaseEHRDataset and SampleDatast becomes SampleEHRDataset. Right now, BaseDataset will be inherited by BaseEHRDataset and BaseSignalDataset. SampleBaseDataset will be inherited by SampleEHRDataset and SampleSignalDataset.

**Feb 06, 2023**

.. code-block:: bash
 
    1. improve readme style
    2. add the pyhealth live 06 and 07 link to pyhealth live

**Feb 01, 2023**

.. code-block:: bash
 
    1. add unittest of PyHealth MedCode and Tokenizer

**Jan 26, 2023**

.. code-block:: bash
 
    1. accelerate MIMIC-IV, eICU and OMOP data loading by using multiprocessing (pandarallel)

**Jan 25, 2023**

.. code-block:: bash

    1. accelerate the MIMIC-III data loading process by using multiprocessing (pandarallel)

**Jan 24, 2023**

.. code-block:: bash

    1. Fix the code typo in pyhealth/tasks/drug_recommendation.py for issue #71.
    2. update the pyhealth live schedule 

**Jan 22, 2023**

.. code-block:: bash

    1. Fix the list of list of vector problem in RNN, Transformer, RETAIN, and CNN
    2. Add initialization examples for RNN, Transformer, RETAIN, CNN, and Deepr
    3. (minor) change the parameters from "Type" and "level" to "type_" and "dim_"
    4. BPDanek adds the "__repr__" function to medcode for better print understanding
    5. add unittest for pyhealth.data

**Jan 21, 2023**

.. code-block:: bash

    1. Added a new model, Deepr (models.Deepr)

**Jan 20, 2023**

.. code-block:: bash

    1. add the pyhealth live 05
    2. add slack channel invitation in pyhealth live page

**Jan 13, 2023**

.. code-block:: bash

    1. add the pyhealth live 03 and 04 video link to the nagivation
    2. add future pyhealth live schedule

**Jan 8, 2023**

.. code-block:: bash

    1. Changed BaseModel.add_feature_transform_layer in models/base_model.py so that it accepts special_tokens if necessary
    2. fix an int/float bug in dataset checking (transform int to float and then process them uniformly)

**Dec 26, 2022**

.. code-block:: bash

    1. add examples to pyhealth.data, pyhealth.datasets
    2. improve jupyter notebook tutorials 0, 1, 2


**Dec 21, 2022**

.. code-block:: bash

    1. add the development logs to the navigation
    2. add the pyhealth live schedule to the nagivation
