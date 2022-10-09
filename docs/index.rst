.. PyHealth documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyHealth!
====================================

.. **Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/pyhealth.svg?color=brightgreen
   :target: https://pypi.org/project/pyhealth/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pyhealth/badge/?version=latest
   :target: https://pyhealth.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/yzhao062/pyhealth/master
   :alt: MyBinder


.. image:: https://img.shields.io/github/stars/yzhao062/pyhealth.svg
   :target: https://github.com/yzhao062/pyhealth/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyhealth.svg?color=blue
   :target: https://github.com/yzhao062/pyhealth/network
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

PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to deploy and more flexible and customizable.
 
----------


Introduction
--------------------------

.. .. image:: https://raw.githubusercontent.com/yzhao062/PyHealth/master/docs/images/logo.png
..    :target: https://raw.githubusercontent.com/yzhao062/PyHealth/master/docs/images/logo.png
..    :alt: PyHealth Logo
..    :align: center

PyHealth can support **diverse electronic health records (EHRs)** such as MIMIC and eICU and all OMOP-CDM based databases and provide **various advanced deep learning algorithms** for handling **important healthcare tasks** such as diagnosis-based drug recommendation, patient hospitalization and mortality prediction, and ICU length stay forecasting, etc.  

*Build a healthcare AI pipeline can be as short as 10 lines of code in PyHealth*.


Modules
--------------------------

All healthcare tasks in our package follow a **five-stage pipeline**: 

 dataset process -> task-specific process -> build ML/DL model -> model training -> inference

We try hard to make sure each stage is as separate as possibe, so that people can customize their own pipeline by only using our data processing steps or the ML models. 

We have implements one module for each of the steps (we build a pipeline training RETAIN on MIMIC-III for drug recommendation task):

* **pyhealth.datasets** provides a clean structure for the dataset. Any instance from this module is independent from downstream tasks. We support the processing scripts for three open datasets, MIMIC-III, MIMIC-IV and eICU, as well as the standard OMOP-formatted data. The output instance contains a unified ``dictionary-based structure``.
.. code-block:: python

    from pyhealth.datasets import MIMIC3BaseDataset
    base_ds = MIMIC3BaseDataset(root='./mimiciii/1.4')

* **pyhealth.tasks** inputs the ``<pyhealth.datasets>`` object and further processes it (clean up irrelevant info.) and finally provides another structure for the specific task. This module bridges the dataset instance and the downstream models.
.. code-block:: python

    from pyhealth.tasks import DrugRecDataset
    drugrec_ds = DrugRecDataset(base_ds)

    from pyhealth.data.split import split_by_pat
    train_loader, val_loader, test_loader = split_by_pat(drugrec_ds, [0.8, 0.1, 0.1])

* **pyhealth.models** provides the state-of-the-art healthcare ML models.
.. code-block:: python
    
    from pyhealth.models import RETAIN
    model = RETAIN("drug_recommendation", voc_size, tokenizers, emb_dim=64)

* **trainer.py** is the training manager for deep learning models.
.. code-block:: python
    
    from pyhealth.trainer import Trainer
    from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel

    trainer = Trainer(enable_logging=True)
    trainer.fit(model, train_loader, val_loader, epochs=50, \
        evaluate_fn=evaluate_multilabel, monitor="jaccard")
    # Best model saved to: ../output/221004-015401/best.ckpt

* **pyhealth.evaluator** lists the evaluator for each healthcare tasks with detailed metrics specification.
.. code-block:: python
    
    model = trainer.load(model, path="../output/221004-015401/best.ckpt") 
    evaluate_multilabel(model, val_loader)
    # {'ddi': 0.07266, 'jaccard': 0.4767, 'prauc': 0.7385, 'f1': 0.6366}


.. Demo - drug recommendation
.. --------------------------
.. * Step 1: dataset process

.. .. code-block:: python

..     from pyhealth.datasets import MIMIC3BaseDataset
..     base_ds = MIMIC3BaseDataset(root='./mimiciii/1.4')


.. * Step 2: task-specific process

.. .. code-block:: python

..     from pyhealth.tasks import DrugRecDataset
..     drugrec_ds = DrugRecDataset(base_ds)

..     # dataset split
..     from pyhealth.data.split import split_by_pat
..     train_loader, val_loader, test_loader = split_by_pat(
..                                                 drugrec_ds,
..                                                 ratios = [2/3, 1/6, 1/6], 
..                                                 batch_size = 64, 
..                                                 seed = 12345,
..                                             )

.. * Step 3: build ML/DL model

.. .. code-block:: python
    
..     # DL model
..     from pyhealth.models import RETAIN
..     model = RETAIN(
..         task = "drug_recommendation",
..         voc_size = drugrec_ds.voc_size,
..         tokenizers = drugrec_ds.tokenizers,
..         emb_dim = 64,
..     )

..     # or ML model
..     from sklearn.linear_model import LogisticRegression
..     model = MLModel(
..         task = "drug_recommendation",
..         classifier = LogisticRegression(random_state=0, max_iter=10, verbose=1),
..         voc_size = drugrec_ds.voc_size,
..         tokenizers = drugrec_ds.tokenizers
..     )

.. * Step 4: model training

.. .. code-block:: python
    
..     from pyhealth.trainer import Trainer
..     from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel

..     # for DL model
..     trainer = Trainer(enable_logging=True)
..     trainer.fit(
..         model,
..         train_loader=train_loader,
..         epochs=50,
..         evaluate_fn=evaluate_multilabel,
..         eval_loader=val_loader,
..         monitor="jaccard",
..     )

..     # for ML model
..     model.fit(
..         train_loader=train_loader,
..         evaluate_fn=evaluate_multilabel,
..         eval_loader=val_loader,
..         monitor="jaccard",
..     )

.. * Step 5: inference

.. .. code-block:: python
    
..     # for DL model, load the best model
..     model = trainer.load(model, path="../output/221004-015401/best.ckpt") 
    
..     # for ML model, load the best model
..     model.load(path="../output/221002-170055/best.ckpt")

..     evaluate_multilabel(model, val_loader)
..     """
..     {'loss': 1.0,
..     'ddi': 0.07266116297106122,
..     'jaccard': 0.47675539329773964,
..     'prauc': 0.738596427730712,
..     'f1': 0.6366047688629042}
..     """

* We also provide medical code mapping functionality via **pyhealth.codemap**.

.. code-block:: python

    from pyhealth.codemap import InnerMap # map within one code system
    ICD = InnerMap('icd-10')
    ICD['I50. 9'] # heart failure
    ICD.children('I50. 9')

    from pyhealth.codemap import CrossMap # map between two code systems
    NDC10_to_RxNorm = CrossMap('NDC10', 'RXCUI')
    NDC10_to_RXCUI['7641315306'] # AZITHROMYCIN tablet

Users can **customize their healthcare AI pipeline** as simply as calling one module

* process your OMOP data via ``pyhealth.datasets.omop``
* process the open eICU (e.g., MIMIC) data via ``pyhealth.datasets.eICU``
* process the dataset for drug recommendation task via ``pyhealth.tasks.DrugRecommendation``
* use the healthcare ML models (e.g., RETAIN) via ``pyhealth.models.RETAIN``.
* code map between for conditions and medicaitons via ``pyhealth.codemap``.

.. **Citing PyHealth**\ :

.. `PyHealth paper <https://arxiv.org/abs/2101.04209>`_ is under review at
.. `JMLR <http://www.jmlr.org/>`_ (machine learning open-source software track).
.. If you use PyHealth in a scientific publication, we would appreciate
.. citations to the following paper::

..     @article{
..     }



.. **Key Links and Resources**\ :


.. * `View the latest codes on Github <https://github.com/ycq091044/PyHealth-OMOP>`_
.. * `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyhealth/master>`_
.. * `Check out the PyHealth paper <https://github.com/yzhao062/pyhealth>`_



----


Datasets
--------------------------
We provide the following datasets for general purpose healthcare AI research:

===================  =======================================  ========================================  ======================================================================================================== 
Dataset              Module                                   Year                                      Information                                                             
===================  =======================================  ========================================  ========================================================================================================
MIMIC-III            ``pyhealth.datasets.MIMIC3BaseDataset``  2016                                      `MIMIC-III Clinical Database <https://physionet.org/content/mimiciii/1.4//>`_    
MIMIC-IV             ``pyhealth.datasets.MIMIC4BaseDataset``  2020                                      `MIMIC-IV Clinical Database <https://physionet.org/content/mimiciv/0.4/>`_  
eICU                 ``pyhealth.datasets.eICUBaseDataset``    2018                                      `eICU Collaborative Research Database <https://eicu-crd.mit.edu//>`_                 
OMOP                 ``pyhealth.datasets.OMOPBaseDataset``                                              `OMOP-CDM schema based dataset <https://www.ohdsi.org/data-standardization/the-common-data-model/>`_                                    
===================  =======================================  ========================================  ========================================================================================================


Machine/Deep Learning Models
--------------------------

============================    ================  =================================  ======  ===========================================================================================================================================
Model Name                      Type              Module                             Year    Reference                                                                                    
============================    ================  =================================  ======  ===========================================================================================================================================
Logistic Regression (LR)        classifical ML    ``pyhealth.models.MLModel``                ``sklearn.linear_model.LogisticRegression``                                                                    
Random Forest (RF)              classifical ML    ``pyhealth.models.MLModel``                ``sklearn.ensemble.RandomForestClassifier``                                                                
Neural Networks (NN)            classifical ML    ``pyhealth.models.MLModel``                ``sklearn.neural_network.MLPClassifier``                                                  
Recurrent Neural Nets (RNN)     deep Learning     ``pyhealth.models.RNN``            2011    `Recurrent neural network based language model <http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf>`_
Transformer                     deep Learning     ``pyhealth.models.Transformer``    2017    `Atention is All you Need <https://arxiv.org/abs/1706.03762>`_        
RETAIN                          deep Learning     ``pyhealth.models.RETAIN``         2016    `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_
GAMENet                         deep Learning     ``pyhealth.models.GAMENet``        2019    `GAMENet: Graph Attention Mechanism for Explainable Electronic Health Record Prediction <https://arxiv.org/abs/1809.01852>`_
MICRON                          deep Learning     ``pyhealth.models.MICRON``         2021    `Change Matters: Medication Change Prediction with Recurrent Residual Networks <https://www.ijcai.org/proceedings/2021/0513>`_
SafeDrug                        deep Learning     ``pyhealth.models.SafeDrug``       2021    `SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations <https://arxiv.org/abs/2105.02711>`_
============================    ================  =================================  ======  ===========================================================================================================================================


Benchmark on Healthcare Tasks
-------------------------------

* **Current results on drug recommendation**. We conduct 2/3 : 1/6 : 1/6 split on MIMIC-III using five-fold cross validation, following `GAMENet <https://arxiv.org/abs/1809.01852>`_, `MICRON <https://www.ijcai.org/proceedings/2021/0513>`_, and `SafeDrug <https://arxiv.org/abs/2105.02711>`_.

===================================     ========    =========      ==========      ==========
Model Name                              DDI         Jaccard         PRAUC           Macro-F1
===================================     ========    =========      ==========      ==========
LR                                      0.0734      0.4979          0.7673          0.6550
RF                                      0.0783      0.4482          0.7295          0.6119
NN                                      0.0732      0.4756          0.7394          0.6355
RNN                                     0.0785      0.4721          0.7445          0.6313
Transformer                             0.0791      0.4991          0.7692          0.6552
RETAIN                                  0.0770      0.5068          0.7727          0.6627
GAMENet                                 0.0760      0.4620          0.7378          0.6221
MICRON                                  0.0733      0.5042          0.7693          0.6599
SafeDrug (DDI hyperparameter: 0.08)     0.0792      0.4709          0.7413          0.6299
SafeDrug (DDI hyperparameter: 0.06)     0.0614      0.4682          0.7420          0.6276
SafeDrug (DDI hyperparameter: 0.04)     0.0513      0.4594          0.7390          0.6189
SafeDrug (DDI hyperparameter: 0.02)     0.0376      0.4448          0.7290          0.6051
===================================     ========    =========      ==========      ==========
(contribute your model by **sending a commit** to ``pyhealth.models``)




----


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Getting Started

   install
   tutorials
   usecase


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Documentation

   api/data
   api/datasets
   api/models
   api/tasks
   api/metrics


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
   faq
..    contribution
   whats_new


.. .. bibliography:: references.bib
..    :cited:
..    :labelprefix: A
..    :keyprefix: a-


.. .. rubric:: References

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
