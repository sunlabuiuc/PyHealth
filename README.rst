Welcome to PyHealth!
====================================

.. image:: https://img.shields.io/pypi/v/pyhealth.svg?color=brightgreen
   :target: https://pypi.org/project/pyhealth/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pyhealth/badge/?version=latest
   :target: https://pyhealth.readthedocs.io/en/latest/
   :alt: Documentation status
   

.. image:: https://img.shields.io/github/stars/sunlabuiuc/pyhealth.svg
   :target: https://github.com/sunlabuiuc/pyhealth/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/sunlabuiuc/pyhealth.svg?color=blue
   :target: https://github.com/sunlabuiuc/pyhealth/network
   :alt: GitHub forks


.. image:: https://pepy.tech/badge/pyhealth
   :target: https://pepy.tech/project/pyhealth
   :alt: Downloads


.. image:: https://pepy.tech/badge/pyhealth/month
   :target: https://pepy.tech/project/pyhealth
   :alt: Downloads



.. -----


.. **Build Status & Coverage & Maintainability & License**

.. .. image:: https://travis-ci.org/yzhao062/pyhealth.svg?branch=master
..    :target: https://travis-ci.org/yzhao062/pyhealth
..    :alt: Build Status


.. .. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
..    :target: https://ci.appveyor.com/project/yzhao062/pyhealth/branch/master
..    :alt: Build status


.. .. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
..    :target: https://codeclimate.com/github/yzhao062/pyhealth/maintainability
..    :alt: Maintainability


.. .. image:: https://img.shields.io/github/license/yzhao062/pyhealth
..    :target: https://github.com/yzhao062/pyhealth/blob/master/LICENSE
..    :alt: License

PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to deploy and more flexible and customizable. `[Tutorials] <https://pyhealth.readthedocs.io/>`_
 
 **[News!]** We are running the "PyHealth Live" gathering at 8 PM CST every Wednesday night! Welcome to join over `zoom <https://illinois.zoom.us/j/87450975602?pwd=ckQyaHhkRitlUzlwYUY3NjdEQ0pFdz09>`_. Check out `PyHealth Live <https://github.com/sunlabuiuc/PyHealth/blob/master/docs/live.rst>`_ for more information and watch the `Live Videos <https://www.youtube.com/playlist?list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV>`_.


.. image:: figure/poster.png
   :width: 810

1. Installation
-----------------

- You could install from PyPi:

.. code-block:: bash

    pip install pyhealth

- or from github source:

.. code-block:: bash

    pip install .


2. Introduction
--------------------------
``pyhealth`` provides these functionalities (we are still enriching some modules):

.. image:: figure/overview.png
   :width: 770

You can use the following functions independently:

- **Dataset**: ``MIMIC-III``, ``MIMIC-IV``, ``eICU``, ``OMOP-CDM``, ``customized EHR datasets``, etc.
- **Tasks**: ``diagnosis-based drug recommendation``, ``patient hospitalization and mortality prediction``, ``length stay forecasting``, etc. 
- **ML models**: ``CNN``, ``LSTM``, ``GRU``, ``LSTM``, ``RETAIN``, ``SafeDrug``, ``Deepr``, etc.

*Build a healthcare AI pipeline can be as short as 10 lines of code in PyHealth*.


3. Build ML Pipelines
--------------------------

All healthcare tasks in our package follow a **five-stage pipeline**: 

.. image:: figure/five-stage-pipeline.png
   :width: 810

We try hard to make sure each stage is as separate as possibe, so that people can customize their own pipeline by only using our data processing steps or the ML models.

Module 1: pyhealth.datasets process the datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pyhealth.datasets`` provides a clean structure for the dataset, independent from the tasks. We support ``MIMIC-III``, ``MIMIC-IV`` and ``eICU``, as well as the standard ``OMOP-formatted data``.

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset
    mimic3base = MIMIC3Dataset(
        # root directory of the dataset
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/", 
        # raw CSV table name
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        # map all NDC codes to ATC 3rd level codes in these tables
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    )

Module 2: pyhealth.tasks define a healthcare task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``pyhealth.tasks`` defines how to process each patient's data into a set of samples for the tasks. In the package, we provide several task examples, such as ``drug recommendation`` and ``length of stay prediction``.

.. code-block:: python

    from pyhealth.tasks import drug_recommendation_mimic3_fn
    mimic3sample = mimic3base.set_task(task_fn=drug_recommendation_mimic3_fn) # use default task

    from pyhealth.datasets import split_by_patient, get_dataloader
    train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

* **STEP 3: <pyhealth.models>** provides the healthcare ML models using ``<pyhealth.models>``. This module also provides model layers, such as ``pyhealth.models.RETAINLayer`` for building customized ML architectures. Our model layers can used as easily as ``torch.nn.Linear``.

.. code-block:: python

    from pyhealth.models import Transformer

    model = Transformer(
        dataset=mimic3sample,
        feature_keys=["conditions", "procedures"],
        label_key="drugs",
        mode="multilabel",
    )

* **STEP 4: <pyhealth.trainer>** is the training manager with ``train_loader``, the ``val_loader``, ``val_metric``, and specify other arguemnts, such as epochs, optimizer, learning rate, etc. The trainer will automatically save the best model and output the path in the end.

.. code-block:: python
    
    from pyhealth.trainer import Trainer

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="pr_auc_samples",
    )

* **STEP 5: <pyhealth.metrics>** provides several **common evaluation metrics** (refer to `Doc <https://pyhealth.readthedocs.io/en/latest/api/metrics.html>`_ and see what are available) and **special metrics** in healthcare, such as drug-drug interaction (DDI) rate.

.. code-block:: python
    
    trainer.evaluate(test_loader)

3.2 Medical Code Map
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **<pyhealth.codemap>** provides two core functionalities: (i) looking up information for a given medical code (e.g., name, category, sub-concept); (ii) mapping codes across coding systems (e.g., ICD9CM to CCSCM). **This module can be independently applied to your research.**

* For code mapping between two coding systems

.. code-block:: python

    from pyhealth.medcode import CrossMap

    codemap = CrossMap.load("ICD9CM", "CCSCM")
    codemap.map("82101") # use it like a dict

    codemap = CrossMap.load("NDC", "ATC")
    codemap.map("00527051210")

* For code ontology lookup within one system

.. code-block:: python

    from pyhealth.medcode import InnerMap

    icd9cm = InnerMap.load("ICD9CM")
    icd9cm.lookup("428.0") # get detailed info
    icd9cm.get_ancestors("428.0") # get parents

3.3 Medical Code Tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **<pyhealth.tokenizer>** is used for transformations between string-based tokens and integer-based indices, based on the overall token space. We provide flexible functions to tokenize 1D, 2D and 3D lists. **This module can be independently applied to your research.**

.. code-block:: python

    from pyhealth.tokenizer import Tokenizer

    # Example: we use a list of ATC3 code as the token
    token_space = ['A01A', 'A02A', 'A02B', 'A02X', 'A03A', 'A03B', 'A03C', 'A03D', \
            'A03F', 'A04A', 'A05A', 'A05B', 'A05C', 'A06A', 'A07A', 'A07B', 'A07C', \
            'A12B', 'A12C', 'A13A', 'A14A', 'A14B', 'A16A']
    tokenizer = Tokenizer(tokens=token_space, special_tokens=["<pad>", "<unk>"])

    # 2d encode 
    tokens = [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', 'B035', 'C129']]
    indices = tokenizer.batch_encode_2d(tokens) # [[8, 9, 10, 11], [12, 1, 1, 0]]

    # 2d decode 
    indices = [[8, 9, 10, 11], [12, 1, 1, 0]]
    tokens = tokenizer.batch_decode_2d(indices) # [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', '<unk>', '<unk>']]

..

4. Tutorials
------------

 We provide the following tutorials to help users get started with our pyhealth.


`Tutorial 0: Introduction to pyhealth.data <https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing>`_ 

`Tutorial 1: Introduction to pyhealth.datasets <https://colab.research.google.com/drive/18kbzEQAj1FMs_J9rTGX8eCoxnWdx4Ltn?usp=sharing>`_ 

`Tutorial 2: Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1r7MYQR_5yCJGpK_9I9-A10HmpupZuIN-?usp=sharing>`_ 

`Tutorial 3: Introduction to pyhealth.models <https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing>`_ 

`Tutorial 4: Introduction to pyhealth.trainer <https://colab.research.google.com/drive/1L1Nz76cRNB7wTp5Pz_4Vp4N2eRZ9R6xl?usp=sharing>`_ 

`Tutorial 5: Introduction to pyhealth.metrics <https://colab.research.google.com/drive/1Mrs77EJ92HwMgDaElJ_CBXbi4iABZBeo?usp=sharing>`_ 

`Tutorial 6: Introduction to pyhealth.tokenizer <https://colab.research.google.com/drive/1bDOb0A5g0umBjtz8NIp4wqye7taJ03D0?usp=sharing>`_

`Tutorial 7: Introduction to pyhealth.medcode <https://colab.research.google.com/drive/1xrp_ACM2_Hg5Wxzj0SKKKgZfMY0WwEj3?usp=sharing>`_

 The following tutorials will help users build their own task pipelines.

`Pipeline 1: Drug Recommendation <https://colab.research.google.com/drive/10CSb4F4llYJvv42yTUiRmvSZdoEsbmFF?usp=sharing>`_ 

`Pipeline 2: Length of Stay Prediction <https://colab.research.google.com/drive/1JoPpXqqB1_lGF1XscBOsDHMLtgvlOYI1?usp=sharing>`_ 

`Pipeline 3: Readmission Prediction <https://colab.research.google.com/drive/1bhCwbXce1YFtVaQLsOt4FcyZJ1_my7Cs?usp=sharing>`_ 

`Pipeline 4: Mortality Prediction <https://colab.research.google.com/drive/1Qblpcv4NWjrnADT66TjBcNwOe8x6wU4c?usp=sharing>`_ 

 The following tutorials will help users to explore advanced features of pyhealth.

`Advanced Tutorial 1: Fit your dataset into our pipeline <https://colab.research.google.com/drive/1UurxwAAov1bL_5OO3gQJ4gAa_paeJwJp?usp=sharing>`_

`Advanced Tutorial 2: Define your own healthcare task <https://colab.research.google.com/drive/1gK6zPXvfFGBM1uNaLP32BOKrnnJdqRq2?usp=sharing>`_ 

`Advanced Tutorial 3: Adopt customized model into pyhealth <https://colab.research.google.com/drive/1F_NJ90GC8_Eq-vKTf7Tyziew4gWjjKoH?usp=sharing>`_ 

`Advanced Tutorial 4: Load your own processed data into pyhealth and try out our ML models <https://colab.research.google.com/drive/1ZRnKch2EyJLrI3G5AvDXVpeE2wwgBWfw?usp=sharing>`_



----


5. Datasets
--------------------------
We provide the processing files for the following open EHR datasets:

===================  =======================================  ========================================  ======================================================================================================== 
Dataset              Module                                   Year                                      Information                                                             
===================  =======================================  ========================================  ========================================================================================================
MIMIC-III            ``pyhealth.datasets.MIMIC3BaseDataset``  2016                                      `MIMIC-III Clinical Database <https://physionet.org/content/mimiciii/1.4//>`_    
MIMIC-IV             ``pyhealth.datasets.MIMIC4BaseDataset``  2020                                      `MIMIC-IV Clinical Database <https://physionet.org/content/mimiciv/0.4/>`_  
eICU                 ``pyhealth.datasets.eICUBaseDataset``    2018                                      `eICU Collaborative Research Database <https://eicu-crd.mit.edu//>`_                 
OMOP                 ``pyhealth.datasets.OMOPBaseDataset``                                              `OMOP-CDM schema based dataset <https://www.ohdsi.org/data-standardization/the-common-data-model/>`_                                    
===================  =======================================  ========================================  ========================================================================================================


6. Machine/Deep Learning Models and Benchmarks
------------------------------------------------

==================================    ================  =================================  ======  ===========================================================================================================================================
Model Name                            Type              Module                             Year    Reference
==================================    ================  =================================  ======  ===========================================================================================================================================
Convolutional Neural Network (CNN)    deep learning     ``pyhealth.models.CNN``            1989    `Handwritten Digit Recognition with a Back-Propagation Network <https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf>`_
Recurrent Neural Nets (RNN)           deep Learning     ``pyhealth.models.RNN``            2011    `Recurrent neural network based language model <http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf>`_
Transformer                           deep Learning     ``pyhealth.models.Transformer``    2017    `Atention is All you Need <https://arxiv.org/abs/1706.03762>`_
RETAIN                                deep Learning     ``pyhealth.models.RETAIN``         2016    `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_
GAMENet                               deep Learning     ``pyhealth.models.GAMENet``        2019    `GAMENet: Graph Attention Mechanism for Explainable Electronic Health Record Prediction <https://arxiv.org/abs/1809.01852>`_
MICRON                                deep Learning     ``pyhealth.models.MICRON``         2021    `Change Matters: Medication Change Prediction with Recurrent Residual Networks <https://www.ijcai.org/proceedings/2021/0513>`_
SafeDrug                              deep Learning     ``pyhealth.models.SafeDrug``       2021    `SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations <https://arxiv.org/abs/2105.02711>`_
==================================    ================  =================================  ======  ===========================================================================================================================================

* Check the `interactive map on benchmark EHR predictive tasks <https://pyhealth.readthedocs.io/en/latest/index.html#benchmark-on-healthcare-tasks>`_.

7. Citing PyHealth
----------------------------------

.. code-block:: bibtex

    @software{pyhealth2022github,
        author = {Chaoqi Yang and Zhenbang Wu and Patrick Jiang and Jimeng Sun},
        title = {{PyHealth}: A Deep Learning Toolkit for Healthcare Predictive Modeling},
        url = {https://github.com/sunlabuiuc/PyHealth},
        year = {2022},
    }

