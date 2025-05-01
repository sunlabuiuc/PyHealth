.. PyHealth documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyHealth!
====================================

.. **Deployment & Documentation & Stats**
.. important::

   * **Join our PyHealth Discord Community!** We are actively looking for contributors and want to get to know our users better! `Click here to join Discord <https://discord.gg/mpb835EHaX>`_
   * **Signup for our mailing list!** We will email any significant PyHealth changes that are soon to come! `Click here to subscribe <https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header>`_

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


.. image:: https://static.pepy.tech/badge/pyhealth
   :target: https://pepy.tech/project/pyhealth
   :alt: Downloads


.. image:: https://img.shields.io/badge/Tutorials-Google%20Colab-red
   :target: https://pyhealth.readthedocs.io/en/latest/tutorials.html
   :alt: Tutorials


.. image:: https://img.shields.io/badge/YouTube-16%20Videos-red
   :target: https://www.youtube.com/playlist?list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV
   :alt: YouTube


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

.. Go to KDD'23 Tutorial Link https://sunlabuiuc.github.io/PyHealth/
-----------------------------------------------------------------

PyHealth is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to develop, test and validate. Your development process becomes more flexible and more customizable. `[GitHub] <https://github.com/sunlabuiuc/PyHealth>`_ 


----------

 **[News!]** We are continueously implemeting good papers and benchmarks into PyHealth, checkout the `[Planned List] <https://docs.google.com/spreadsheets/d/1PNMgDe-llOm1SM5ZyGLkmPysjC4wwaVblPLAHLxejTw/edit#gid=159213380>`_. Welcome to pick one from the list and send us a PR or add more influential and new papers into the plan list.


Introduction `[Video] <https://www.youtube.com/watch?v=1Ir6hzU4Nro&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=1>`_
--------------------------------------------------------------------------------------------------------------------

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
    mimic3base = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )

User could also store their own dataset into our ``<pyhealth.datasets.SampleBaseDataset>`` structure and then follow the same pipeline below, see `Tutorial <https://colab.research.google.com/drive/1b9xRbxUz-HLzxsrvxdsdJ868ajGQCY6U?usp=sharing>`_

* **STEP 2: <pyhealth.tasks>** inputs the ``<pyhealth.datasets>`` object and defines how to process each patient's data into a set of samples for the tasks. In the package, we provide several task examples, such as ``mortality prediction`` and ``length of stay prediction``.

.. code-block:: python

    from pyhealth.tasks import MortalityPredictionMIMIC3
    from pyhealth.datasets import split_by_patient, get_dataloader
    mimic3_mortality_prediction = MortalityPredictionMIMIC3()
    mimic3sample = mimic3base.set_task(task_fn=mimic3_mortality_prediction) # use default task
    train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])

    # create dataloaders (torch.data.DataLoader)
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

* **STEP 3: <pyhealth.models>** provides the healthcare ML models using ``<pyhealth.models>``. This module also provides model layers, such as ``pyhealth.models.RETAINLayer`` for building customized ML architectures. Our model layers can used as easily as ``torch.nn.Linear``.

.. code-block:: python

    from pyhealth.models import RNN


    model = RNN(
      dataset=samples,
    )

* **STEP 4: <pyhealth.trainer>** is the training manager with ``train_loader``, the ``val_loader``, ``val_metric``, and specify other arguemnts, such as epochs, optimizer, learning rate, etc. The trainer will automatically save the best model and output the path in the end.

.. code-block:: python

    from pyhealth.trainer import Trainer

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="roc_auc",
    )

* **STEP 5: <pyhealth.metrics>** provides several **common evaluation metrics** (refer to `Doc <https://pyhealth.readthedocs.io/en/latest/api/metrics.html>`_ and see what are available) and **special metrics** in healthcare, such as drug-drug interaction (DDI) rate.

.. code-block:: python

    trainer.evaluate(test_loader)


Medical Code Map
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

Medical Code Tokenizer
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
MIMIC-III            ``pyhealth.datasets.MIMIC3Dataset``      2016                                      `MIMIC-III Clinical Database <https://physionet.org/content/mimiciii/1.4//>`_
MIMIC-IV             ``pyhealth.datasets.MIMIC4Dataset``      2020                                      `MIMIC-IV Clinical Database <https://physionet.org/content/mimiciv/0.4/>`_
eICU                 ``pyhealth.datasets.eICUDataset``        2018                                      `eICU Collaborative Research Database <https://eicu-crd.mit.edu//>`_
OMOP                 ``pyhealth.datasets.OMOPDataset``                                                  `OMOP-CDM schema based dataset <https://www.ohdsi.org/data-standardization/the-common-data-model/>`_
SleepEDF             ``pyhealth.datasets.SleepEDFDataset``    2018                                      `Sleep-EDF dataset <https://physionet.org/content/sleep-edfx/1.0.0/>`_
SHHS                 ``pyhealth.datasets.SHHSDataset``        2016                                      `Sleep Heart Health Study dataset <https://sleepdata.org/datasets/shhs>`_
ISRUC                ``pyhealth.datasets.ISRUCDataset``       2016                                      `ISRUC-SLEEP dataset <https://sleeptight.isr.uc.pt/?page_id=48>`_
===================  =======================================  ========================================  ========================================================================================================


Machine/Deep Learning Models
-----------------------------

==================================    ================  =================================  ======  ============================================================================================================================================================================  =======================================================================================================================================================================================
Model Name                            Type              Module                             Year    Summary                                                                                                                                                                       Reference
==================================    ================  =================================  ======  ============================================================================================================================================================================  =======================================================================================================================================================================================
Multi-layer Perceptron                deep learning     ``pyhealth.models.MLP``            1986    MLP treats each feature as static                                                                                                                                             `Backpropagation: theory, architectures, and applications <https://www.taylorfrancis.com/books/mono/10.4324/9780203763247/backpropagation-yves-chauvin-david-rumelhart>`_
Convolutional Neural Network (CNN)    deep learning     ``pyhealth.models.CNN``            1989    CNN runs on the conceptual patient-by-visit grids                                                                                                                             `Handwritten Digit Recognition with a Back-Propagation Network <https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf>`_
Recurrent Neural Nets (RNN)           deep Learning     ``pyhealth.models.RNN``            2011    RNN (includes LSTM and GRU) can run on any sequential level (e.g., visit by visit sequences)                                                                                  `Recurrent neural network based language model <http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf>`_
Transformer                           deep Learning     ``pyhealth.models.Transformer``    2017    Transformer can run on any sequential level (e.g., visit by visit sequences)                                                                                                  `Atention is All you Need <https://arxiv.org/abs/1706.03762>`_
RETAIN                                deep Learning     ``pyhealth.models.RETAIN``         2016    RETAIN uses two RNN to learn patient embeddings while providing feature-level and visit-level importance.                                                                     `RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism <https://arxiv.org/abs/1608.05745>`_
GAMENet                               deep Learning     ``pyhealth.models.GAMENet``        2019    GAMENet uses memory networks, used only for drug recommendation task                                                                                                          `GAMENet: Graph Attention Mechanism for Explainable Electronic Health Record Prediction <https://arxiv.org/abs/1809.01852>`_
MICRON                                deep Learning     ``pyhealth.models.MICRON``         2021    MICRON predicts the future drug combination by instead predicting the changes w.r.t. the current combination, used only for drug recommendation task                          `Change Matters: Medication Change Prediction with Recurrent Residual Networks <https://www.ijcai.org/proceedings/2021/0513>`_
SafeDrug                              deep Learning     ``pyhealth.models.SafeDrug``       2021    SafeDrug encodes drug molecule structures by graph neural networks, used only for drug recommendation task                                                                    `SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations <https://arxiv.org/abs/2105.02711>`_
MoleRec                               deep Learning     ``pyhealth.models.MoleRec``        2023    MoleRec encodes drug molecule in a substructure level as well as the patient's information into a drug combination representation, used only for drug recommendation task     `MoleRec: Combinatorial Drug Recommendation with Substructure-Aware Molecular Representation Learning <https://www.researchgate.net/publication/368454974_MoleRec_Combinatorial_Drug_Recommendation_with_Substructure-Aware_Molecular_Representation_Learning>`_
Deepr                                 deep Learning     ``pyhealth.models.Deepr``          2017    Deepr is based on 1D CNN. General purpose.                                                                                                                                    `Deepr : A Convolutional Net for Medical Records <https://arxiv.org/abs/1607.07519>`_
ContraWR Encoder (STFT+CNN)           deep Learning     ``pyhealth.models.ContraWR``       2021    ContraWR encoder uses short time Fourier transform (STFT) + 2D CNN, used for biosignal learning                                                                               `Self-supervised EEG Representation Learning for Automatic Sleep Staging <https://arxiv.org/abs/2110.15278>`_
SparcNet (1D CNN)                     deep Learning     ``pyhealth.models.SparcNet``       2023    SparcNet is based on 1D CNN, used for biosignal learning                                                                                                                      `Development of Expert-level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation <#>`_
TCN                                   deep learning     ``pyhealth.models.TCN``            2018    TCN is based on dilated 1D CNN. General purpose                                                                                                                               `Temporal Convolutional Networks <https://arxiv.org/abs/1803.01271>`_
AdaCare                               deep learning     ``pyhealth.models.AdaCare``        2020    AdaCare uses CNNs with dilated filters to learn enriched patient embedding. It uses feature calibration module to provide the feature-level and visit-level interpretability  `AdaCare: Explainable Clinical Health Status Representation Learning via Scale-Adaptive Feature Extraction and Recalibration <https://arxiv.org/abs/1911.12205>`_
ConCare                               deep learning     ``pyhealth.models.ConCare``        2020    ConCare uses transformers to learn patient embedding and calculate inter-feature correlations.                                                                                `ConCare: Personalized Clinical Feature Embedding via Capturing the Healthcare Context <https://arxiv.org/abs/1911.12216>`_
StageNet                              deep learning     ``pyhealth.models.StageNet``       2020    StageNet uses stage-aware LSTM to conduct clinical predictive tasks while learning patient disease progression stage change unsupervisedly                                    `StageNet: Stage-Aware Neural Networks for Health Risk Prediction <https://arxiv.org/abs/2001.10054>`_
Dr. Agent                             deep learning     ``pyhealth.models.Agent``          2020    Dr. Agent uses two reinforcement learning agents to learn patient embeddings by mimicking clinical second opinions                                                            `Dr. Agent: Clinical predictive model via mimicked second opinions <https://academic.oup.com/jamia/article/27/7/1084/5858308>`_
GRASP                                 deep learning     ``pyhealth.models.GRASP``          2021    GRASP uses graph neural network to identify latent patient clusters and uses the clustering information to learn patient                                                      `GRASP: Generic Framework for Health Status Representation Learning Based on Incorporating Knowledge from Similar Patients <https://ojs.aaai.org/index.php/AAAI/article/view/16152>`_
==================================    ================  =================================  ======  ============================================================================================================================================================================  =======================================================================================================================================================================================


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Getting Started

   install
   tutorials
   advance_tutorials


.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: Documentation

   api/data
   api/datasets
   api/tasks
   api/models
   api/trainer
   api/tokenizer
   api/metrics
   api/medcode
   api/calib


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   live
   log
   about



.. .. bibliography:: references.bib
..    :cited:
..    :labelprefix: A
..    :keyprefix: a-


.. .. rubric:: References

..    Indices and tables
..    ==================

..    * :ref:`genindex`
..    * :ref:`modindex`
..    * :ref:`search`
