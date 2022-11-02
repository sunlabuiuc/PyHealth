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
   :target: https://pyhealth.readthedocs.io/en/latest/
   :alt: Documentation status
   

.. image:: https://img.shields.io/github/stars/yzhao062/pyhealth.svg
   :target: https://github.com/sunlabuiuc/pyhealth/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/yzhao062/pyhealth.svg?color=blue
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

PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to develop, test and validate. Your development process becomes more flexible and more customizable. `[GitHub] <https://github.com/sunlabuiuc/PyHealth>`_
 
----------

 **[News!]** Current PyHealth is a brand-new version with more flexible and independent function calls (such as users may only use the data preprocessing parts or only the ML model layers as convenient as torch.nn.Linear). The `old documents <https://pyhealth.readthedocs.io/en/archived/>`_ are no longer maintained.


Introduction `[Video] <https://drive.google.com/file/d/1g8wt8jBhZpmlUDG9l05RtfPWGOlOJK94/view?usp=share_link>`_
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

 load dataset -> define task function -> build ML/DL model -> model training -> inference

! We try hard to make sure each stage is as separate as possibe, so that people can customize their own pipeline by only using our data processing steps or the ML models. Each step will call one module and we introduce them using an example.

An ML Pipeline Example 
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **STEP 1: <pyhealth.datasets>** provides a clean structure for the dataset, independent from the tasks. We support ``MIMIC-III``, ``MIMIC-IV`` and ``eICU``, as well as the standard ``OMOP-formatted data``. The dataset is stored in a unified ``Patient-Visit-Event`` structure.
.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset
    mimic3dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/mimiciii/1.4/", 
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": "ATC"}, # map all NDC codes to ATC codes in these tables
    )

* **STEP 2: <pyhealth.tasks>** inputs the ``<pyhealth.datasets>`` object and defines how to process each pateint's data into a set of samples for the tasks. In the package, we provide several task examples, such as ``drug recommendation`` and ``length of stay prediction``.
.. code-block:: python

    from pyhealth.tasks import drug_recommendation_mimic3_fn
    from pyhealth.datasets.splitter import split_by_patient
    from torch.utils.data import DataLoader
    from pyhealth.utils import collate_fn_dict

    mimic3dataset.set_task(task_fn=drug_recommendation_mimic3_fn) # use default drugrec task
    train_ds, val_ds, test_ds = split_by_patient(mimic3dataset, [0.8, 0.1, 0.1])

    # create dataloaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn_dict)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn_dict)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn_dict)

* **STEP 3: <pyhealth.models>** provides the healthcare ML models using ``<pyhealth.datasets>``. This module also provides model layers, such as ``pyhealth.models.RETAINLayer`` for building customized ML architectures. Our model layers can used as easily as ``torch.nn.Linear``.
.. code-block:: python
    
    from pyhealth.models import Transformer

    device = "cuda:0"
    model = Transformer(
        dataset=mimic3dataset,
        tables=["conditions", "procedures"],
        mode="multilabel",
    )
    model.to(device)

* **STEP 4: <pyhealth.trainer>** is the training manager with ``train_loader``, the ``val_loader``, ``val_metric``, and specify other arguemnts, such as epochs, optimizer, learning rate, etc. The trainer will automatically save the best model and output the path in the end.
.. code-block:: python
    
    from pyhealth.trainer import Trainer
    from pyhealth.metrics import pr_auc_multilabel
    import torch

    trainer = Trainer(enable_logging=True, output_path="../output", device=device)
    trainer.fit(model,
        train_loader=train_loader,
        epochs=10,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        val_loader=val_loader,
        val_metric=pr_auc_multilabel,
    )
    # Best model saved to: ../output/221004-015401/best.ckpt

* **STEP 5: <pyhealth.metrics>** provides: (i) **common evaluation metrics** and the usage is the same as ``<pyhealth.metrics>``; (ii) **metrics (weighted by patients)** for patient-level tasks; (iii) **special metrics** in healthcare, such as drug-drug interaction (DDI) rate.

.. code-block:: python
    
    from pyhealth.evaluator import evaluate
    from pyhealth.metrics import accuracy_multilabel, jaccard_multilabel, f1_multilabel

    # load best model and do inference
    model = trainer.load_best_model(model)
    y_gt, y_prob, y_pred = evaluate(model, test_loader, device)

    jaccard = jaccard_multilabel(y_gt, y_pred)
    accuracy = accuracy_multilabel(y_gt, y_pred)
    f1 = f1_multilabel(y_gt, y_pred)
    prauc = pr_auc_multilabel(y_gt, y_prob)

    print("jaccard: ", jaccard)
    print("accuracy: ", accuracy)
    print("f1: ", f1)
    print("prauc: ", prauc)


Medical Code Map
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **<pyhealth.codemap>** provides two core functionalities: (i) looking up information for a given medical code (e.g., name, category, sub-concept); (ii) mapping codes across coding systems (e.g., ICD9CM to CCSCM). **This module can be easily applied to your research.**

* For code mapping between two coding systems
.. code-block:: python

    from pyhealth.medcode import CrossMap
    codemap = CrossMap("ICD9CM", "CCSCM")
    codemap.map("82101") # use it like a dict

    codemap = CrossMap("NDC", "ATC", level=3)
    codemap.map("00527051210")

* For code ontology lookup within one system
.. code-block:: python

    from pyhealth.medcode import InnerMap
    ICD9CM = InnerMap("ICD9CM")
    ICD9CM.lookup("428.0") # get detailed info
    ICD9CM.get_ancestors("428.0") # get parents

Medical Code Tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **<pyhealth.tokenizer>** is used for transformations between string-based tokens and integer-based indices, based on the overall token space. We provide flexible functions to tokenize 1D, 2D and 3D lists. This module can be used in many other scenarios.

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

----

Users can **customize their healthcare AI pipeline** as simply as calling one module

* process your OMOP data via ``pyhealth.datasets``
* process the open eICU (e.g., MIMIC) data via ``pyhealth.datasets``
* define your own task on existing databases via ``pyhealth.tasks``
* use existing healthcare models or build upon it (e.g., RETAIN) via ``pyhealth.models``.
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
-----------------------------

==================================    ================  =================================  ======  ===========================================================================================================================================
Model Name                            Type              Module                             Year    Reference
==================================    ================  =================================  ======  ===========================================================================================================================================
Logistic Regression (LR)              classic ML        ``pyhealth.models.MLModel``                ``sklearn.linear_model.LogisticRegression``
Random Forest (RF)                    classic ML        ``pyhealth.models.MLModel``                ``sklearn.ensemble.RandomForestClassifier``
Neural Networks (NN)                  classic ML        ``pyhealth.models.MLModel``                ``sklearn.neural_network.MLPClassifier``
Convolutional Neural Network (CNN)    deep learning     ``pyhealth.models.CNN``            1989    `Handwritten Digit Recognition with a Back-Propagation Network <https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf>`_
Recurrent Neural Nets (RNN)           deep Learning     ``pyhealth.models.RNN``            2011    `Recurrent neural network based language model <http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf>`_
Transformer                           deep Learning     ``pyhealth.models.Transformer``    2017    `Atention is All you Need <https://arxiv.org/abs/1706.03762>`_
RETAIN                                deep Learning     ``pyhealth.models.RETAIN``         2016    `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_
GAMENet                               deep Learning     ``pyhealth.models.GAMENet``        2019    `GAMENet: Graph Attention Mechanism for Explainable Electronic Health Record Prediction <https://arxiv.org/abs/1809.01852>`_
MICRON                                deep Learning     ``pyhealth.models.MICRON``         2021    `Change Matters: Medication Change Prediction with Recurrent Residual Networks <https://www.ijcai.org/proceedings/2021/0513>`_
SafeDrug                              deep Learning     ``pyhealth.models.SafeDrug``       2021    `SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations <https://arxiv.org/abs/2105.02711>`_
==================================    ================  =================================  ======  ===========================================================================================================================================


Benchmark on Healthcare Tasks
----------------------------------

* Here is our `benchmark doc <https://docs.google.com/spreadsheets/d/1c4OwCSDaEt7vGmocidq1hK2HCTeB6ZHDzAZvlubpi08/edit#gid=1602645797>`_ on healthcare tasks. You can also check this below.

We also provide function for leaderboard generation, check it out in our `github repo <https://github.com/sunlabuiuc/PyHealth/tree/master/leaderboard>`_.


.. raw:: html

   <iframe allowtransparency="true" style="background: ##FFFFFF;" allowfullscreen=true
   width=100% height='700' src="https://docs.google.com/spreadsheets/d/e/2PACX-1vSndhECH_KZmWbivw0ewl8qhgukMwuZ2sdf0wPF4cvf1l4ZnEEIrkGkqLed8z82xICvSxUcW3-X0K2b/pubhtml?gid=1602645797&amp;single=true&amp;widget=true&amp;headers=false"></iframe>
..
.. -------------------------------

.. * **Current results on drug recommendation**. We conduct 2/3 : 1/6 : 1/6 split on MIMIC-III using five-fold cross validation, following `GAMENet <https://arxiv.org/abs/1809.01852>`_, `MICRON <https://www.ijcai.org/proceedings/2021/0513>`_, and `SafeDrug <https://arxiv.org/abs/2105.02711>`_.

.. ===================================     ========    =========      ==========      ==========
.. Model Name                              DDI         Jaccard         PRAUC           Macro-F1
.. ===================================     ========    =========      ==========      ==========
.. LR                                      0.0734      0.4979          0.7673          0.6550
.. RF                                      0.0783      0.4482          0.7295          0.6119
.. NN                                      0.0732      0.4756          0.7394          0.6355
.. RNN                                     0.0785      0.4721          0.7445          0.6313
.. CNN                                     0.0785      0.4394          0.7364          0.6313
.. Transformer                             0.0791      0.4991          0.7692          0.6552
.. RETAIN                                  0.0770      0.5068          0.7727          0.6627
.. GAMENet                                 0.0760      0.4620          0.7378          0.6221
.. MICRON                                  0.0733      0.5042          0.7693          0.6599
.. SafeDrug (DDI hyperparameter: 0.08)     0.0792      0.4709          0.7413          0.6299
.. SafeDrug (DDI hyperparameter: 0.06)     0.0614      0.4682          0.7420          0.6276
.. SafeDrug (DDI hyperparameter: 0.04)     0.0513      0.4594          0.7390          0.6189
.. SafeDrug (DDI hyperparameter: 0.02)     0.0376      0.4448          0.7290          0.6051
.. ===================================     ========    =========      ==========      ==========
.. (contribute your model by **sending a commit** to ``pyhealth.models``)





.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Getting Started

   install
   tutorials
..    usecase


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Documentation

   api/data
   api/datasets
   api/tasks
   api/models
   api/trainer
   api/evaluator
   api/tokenizer
   api/metrics
   api/medcode


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
..    faq
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
