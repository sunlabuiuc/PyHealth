.. PyHealth documentation master file, created by
   sphinx-quickstart on Wed Aug  5 21:17:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyHealth's documentation!
====================================


**Deployment & Documentation & Stats**

.. image:: https://img.shields.io/pypi/v/pyhealth.svg?color=brightgreen
   :target: https://pypi.org/project/pyhealth/
   :alt: PyPI version


.. image:: https://readthedocs.org/projects/pyhealth/badge/?version=latest
   :target: https://pyhealth.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


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


-----


**Build Status & Coverage & Maintainability & License**

.. image:: https://travis-ci.org/yzhao062/pyhealth.svg?branch=master
   :target: https://travis-ci.org/yzhao062/pyhealth
   :alt: Build Status


.. image:: https://circleci.com/gh/yzhao062/PyHealth.svg?style=svg
   :target: https://circleci.com/gh/yzhao062/PyHealth
   :alt: Circle CI


.. image:: https://ci.appveyor.com/api/projects/status/1kupdy87etks5n3r/branch/master?svg=true
   :target: https://ci.appveyor.com/project/yzhao062/pyhealth/branch/master
   :alt: Build status


.. image:: https://api.codeclimate.com/v1/badges/bdc3d8d0454274c753c4/maintainability
   :target: https://codeclimate.com/github/yzhao062/pyhealth/maintainability
   :alt: Maintainability


.. image:: https://img.shields.io/github/license/yzhao062/pyhealth
   :target: https://github.com/yzhao062/pyhealth/blob/master/LICENSE
   :alt: License


-----


**Development Status**: **As of 08/05/2020, PyHealth is under active development and in its alpha stage. Please follow, star, and fork to get the latest functions**!


**PyHealth** is a comprehensive and flexible **Python library** for **healthcare AI**, designed for both **ML researchers** and **medical practitioners**.
The library is proudly developed and maintained by researchers at `Carnegie Mellon University <https://www.cmu.edu/>`_, `IQVIA <https://www.iqvia.com/>`_, and `University of Illinois at Urbana-Champaign <https://illinois.edu/>`_.
PyHealth makes many important healthcare tasks become accessible, such as **phenotyping prediction**, **mortality prediction**,
**ICU length stay forecasting**, etc. Running these prediction tasks with deep learning models can be as short as 10 lines of code.



PyHealth comes with three major modules: (i) *data preprocessing module*; (ii) *learning module*
and (iii) *evaluation module*. Typically, one can run the data prep module to prepare the data, then feed to the learning module for prediction, and finally assess
the result with the evaluation module.
Users can use the full system as mentioned or just selected modules based on the own need:

* **Deep learning researchers** may directly use the processed data along with the proposed new models.
* **Medical personnel**, may leverage our data preprocessing module to convert the medical data to the format that learning models could digest, and then perform the inference tasks to get insights from the data.


PyHealth is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various datasets and algorithms.
* **Advanced models**\ , including **latest deep learning models** and **classical machine learning models**.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Customizable modules and flexible design**: each module may be turned on/off or totally replaced by custom functions. The trained models can be easily exported and reloaded for fast exexution and deployment.

**API Demo for LSTM on Phenotyping Prediction**\ :


   .. code-block:: python


       # load pre-processed CMS dataset
       from pyhealth.data.expdata_generator import cms as cms_expdata_generator

       cur_dataset = cms_expdata_generator(exp_id=exp_id, sel_task='phenotyping')
       cur_dataset.get_exp_data()
       cur_dataset.load_exp_data()

       # initialize the model for training
       from pyhealth.models.lstm import LSTM
       clf = LSTM(exp_id)
       clf.fit(cur_dataset.train, cur_dataset.valid)

       # load the best model for inference
       clf.load_model()
       clf.inference(cur_dataset.test)
       pred_results = clf.get_results()

       # evaluate the model
       from pyhealth import evaluation
       evaluator = evaluation.__dict__['phenotyping']
       r = evaluator(pred_results['hat_y'], pred_results['y'])


**Citing PyHealth**\ :

`PyHealth paper <https://github.com/yzhao062/pyhealth>`_ is under review at
`JMLR <http://www.jmlr.org/>`_ (machine learning open-source software track).
If you use PyHealth in a scientific publication, we would appreciate
citations to the following paper::

    @article{zhao2020pyhealth,
      author  = {Zhao, Yue and Qiao, Zhi and Xiao, Cao and Glass, Lucas and Hu, Xiyang and Sun, Jimeng},
      title   = {PyHealth: A Python Library for Healthcare AI},
      year    = {2020},
    }

or::

    Zhao, Y., Qiao, Z., Xiao, C., Glass, L., Hu, X and Sun, J., 2020. PyHealth: A Python Library for Healthcare AI.


**Key Links and Resources**\ :


* `View the latest codes on Github <https://github.com/yzhao062/pyhealth>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyhealth/master>`_
* `Check out the PyHealth paper <https://github.com/yzhao062/pyhealth>`_



----


Preprocessed Datasets & Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**(i) Preprocessed Datasets** (customized data preprocessing function is provided in the example folders):

===================  ================  ======================================================================================================    ======================================================    ===============================================================================================================
Type                 Abbr              Description                                                                                               Processed Function                                        Link
===================  ================  ======================================================================================================    ======================================================    ===============================================================================================================
EHR-ICU              MIMIC III         A relational database containing tables of data relating to patients who stayed within ICU.               \\examples\\data_generation\\dataloader_mimic             https://mimic.physionet.org/gettingstarted/overview/
EHR-ICU              MIMIC_demo        The MIMIC-III demo database is limited to 100 patients and excludes the noteevents table.                 \\examples\\data_generation\\dataloader_mimic_demo        https://mimic.physionet.org/gettingstarted/demo/
EHU-Claim            CMS               DE-SynPUF: CMS 2008-2010 Data Entrepreneurs Synthetic Public Use File                                     \\examples\\data_generation\\dataloader_cms               https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs
===================  ================  ======================================================================================================    ======================================================    ===============================================================================================================

You may download the above datasets at the links. The structure of the generated datasets can be found in datasets folder:

* \\datasets\\cms\\x_datat\\...csv
* \\datasets\\cms\\y_data\\phenotyping.csv
* \\datasets\\cms\\y_data\\mortality.csv

The processed datasets (X,y) should be put in x_data, y_data correspondingly, to be appropriately digested by deep learning models.

**(ii) Machine Learning and Deep Learning Models** :

===================  ================  ======================================================================================================  =====  ========================================
Type                 Abbr              Algorithm                                                                                               Year   Ref
===================  ================  ======================================================================================================  =====  ========================================
Classical Models     LogisticReg       Logistic Regression                                                                                     N/A
Classical Models     XGBoost           XGBoost: A scalable tree boosting system                                                                2016   [#Chen2016Xgboost]_
Neural Networks      LSTM              Long short-term memory                                                                                  1997   [#Hochreiter1997Long]_
Neural Networks      GRU               Gated recurrent unit                                                                                    2014   [#Cho2014Learning]_
Neural Networks      RETAIN            RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism         2016   [#Choi2016RETAIN]_
Neural Networks      Dipole            Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks  2017   [#Ma2017Dipole]_
Neural Networks      tLSTM             Patient Subtyping via Time-Aware LSTM Networks                                                          2017   [#Baytas2017tLSTM]_
Neural Networks      RAIM              RAIM: Recurrent Attentive and Intensive Model of Multimodal Patient Monitoring Data                     2018   [#Xu2018RAIM]_
Neural Networks      StageNet          StageNet: Stage-Aware Neural Networks for Health Risk Prediction                                        2020   [#Gao2020StageNet]_
===================  ================  ======================================================================================================  =====  ========================================

Examples of running ML and DL models can be found below, or directly at \\examples\\learning_examples\\


**(iii) Evaluation Metrics** :

=======================  =======================  ======================================================================================================  ===============================================
Type                     Abbr                     Metric                                                                                                  Method
=======================  =======================  ======================================================================================================  ===============================================
Binary Classification    average_precision_score  Compute micro/macro average precision (AP) from prediction scores                                       pyhealth.evaluation.xxx.get_avg_results
Binary Classification    roc_auc_score            Compute micro/macro ROC AUC score from prediction scores                                                pyhealth.evaluation.xxx.get_avg_results
Binary Classification    recall, precision, f1    Get recall, precision, and f1 values                                                                    pyhealth.evaluation.xxx.get_predict_results
Multi Classification     To be done here
=======================  =======================  ======================================================================================================  ===============================================


**(iv) Supported Tasks**:

=======================  =======================  ======================================================================================================  =========================================================
Type                     Abbr                     Description                                                                                             Method
=======================  =======================  ======================================================================================================  =========================================================
Multi-classification     phenotyping              Predict the diagnosis code of a patient based on other information, e.g., procedures                    \\examples\\data_generation\\generate_phenotyping_xxx.py
Binary Classification    mortality prediction     Predict whether a patient may pass away during the hospital                                             \\examples\\data_generation\\generate_mortality_xxx.py
Regression               ICU stay length pred     Forecast the length of an ICU stay                                                                      \\examples\\data_generation\\generate_icu_length_xxx.py
=======================  =======================  ======================================================================================================  =========================================================




Algorithm Benchmark
^^^^^^^^^^^^^^^^^^^

**The comparison among of implemented models** will be made available later
with a benchmark paper. TBA soon :)



----


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   example


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   api_cc
   api


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Additional Information

   about
   faq
   whats_new


----


.. rubric:: References

.. bibliography:: references.bib
   :cited:
   :labelprefix: A
   :keyprefix: a-



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
