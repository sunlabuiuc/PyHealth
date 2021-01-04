A Python Library for Health Predictive Models (PyHealth)
========================================================


**Deployment & Documentation & Stats**

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


.. image:: https://raw.githubusercontent.com/yzhao062/PyHealth/master/docs/images/logo.png
   :target: https://raw.githubusercontent.com/yzhao062/PyHealth/master/docs/images/logo.png
   :alt: PyHealth Logo
   :align: center

**Development Status**: **As of 11/09/2020, PyHealth is under active development and in its alpha stage. Please follow, star, and fork to get the latest functions**!


**PyHealth** is a comprehensive **Python package** for **healthcare AI**, designed for both **ML researchers** and **healthcare and medical practitioners**.
**PyHealth** accepts diverse healthcare data such as longitudinal electronic health records (EHRs), continuous signials (ECG, EEG), and clinical notes (to be added), and supports various predictive modeling methods using deep learning and other advanced machine learning algorithms published in the literature. 

The library is proudly developed and maintained by researchers from `Carnegie Mellon University <https://www.cmu.edu/>`_, `IQVIA <https://www.iqvia.com/>`_, and `University of Illinois at Urbana-Champaign <https://illinois.edu/>`_.
PyHealth makes many important healthcare tasks become accessible, such as **phenotyping prediction**, **mortality prediction**,
and **ICU length stay forecasting**, etc. Running these prediction tasks with deep learning models can be as short as 10 lines of code in PyHealth.


**PyHealth comes with three major modules**: (i) *data preprocessing module*; (ii) *learning module*
and (iii) *evaluation module*. Typically, one can run the data prep module to prepare the data, then feed to the learning module for model training and prediction, and finally assess the results with the evaluation module.
Users can use the full system as mentioned or just selected modules based on their own needs:

* **Deep learning researchers** may directly use the processed data along with the proposed new models.
* **Healthcare and Medical personnel**, may leverage our data preprocessing module to convert the medical data to the format that machine learning models could digest, and then perform the inference tasks to get insights from the data. This package can support them in various health analytics tasks including disease detection, risk prediction, patient subtyping, health monitoring, etc.


PyHealth is featured for:

* **Unified APIs, detailed documentation, and interactive examples** across various types of datasets and algorithms.
* **Advanced models**\ , including **latest deep learning models** and **classical machine learning models**.
* **Wide coverage**, supporting **sequence data**, **image data**, **series data** and **text data** like clinical notes.
* **Optimized performance with JIT and parallelization** when possible, using `numba <https://github.com/numba/numba>`_ and `joblib <https://github.com/joblib/joblib>`_.
* **Customizable modules and flexible design**: each module may be turned on/off or totally replaced by custom functions. The trained models can be easily exported and reloaded for fast execution and deployment.

**API Demo for LSTM on Phenotyping Prediction**\ :


   .. code-block:: python


       # load pre-processed CMS dataset
       from pyhealth.data.expdata_generator import sequencedata as expdata_generator

       expdata_id = '2020.0810.data.mortality.mimic'
       cur_dataset = expdata_generator(exp_id=exp_id)
       cur_dataset.get_exp_data(sel_task='mortality', )
       cur_dataset.load_exp_data()

       # initialize the model for training
       from pyhealth.models.sequence.lstm import LSTM
       # enable GPU
       expmodel_id = 'test.model.lstm.0001'
       clf = LSTM(expmodel_id=expmodel_id, n_batchsize=20, use_gpu=True, n_epoch=100)
       clf.fit(cur_dataset.train, cur_dataset.valid)

       # load the best model for inference
       clf.load_model()
       clf.inference(cur_dataset.test)
       pred_results = clf.get_results()

       # evaluate the model
       from pyhealth.evaluation.evaluator import func
       r = func(pred_results['hat_y'], pred_results['y'])
       print(r)



**Citing PyHealth**\ :

`PyHealth paper <https://github.com/yzhao062/pyhealth>`_ is under review at
`JMLR <http://www.jmlr.org/>`_ (machine learning open-source software track).
If you use PyHealth in a scientific publication, we would appreciate
citations to the following paper::

    @article{zhao2020pyhealth,
      author  = {Zhao, Yue and Qiao, Zhi and Xiao, Cao and Glass, Lucas and Sun, Jimeng},
      title   = {PyHealth: A Python Library for Healthcare AI},
      year    = {2020},
    }

or::

    Zhao, Y., Qiao, Z., Xiao, C., Glass, L. and Sun, J., 2020. PyHealth: A Python Library for Healthcare AI.


**Key Links and Resources**\ :


* `View the latest codes on Github <https://github.com/yzhao062/pyhealth>`_
* `Execute Interactive Jupyter Notebooks <https://mybinder.org/v2/gh/yzhao062/pyhealth/master>`_
* `Check out the PyHealth paper <https://github.com/yzhao062/pyhealth>`_



**Table of Contents**\ :


* `Installation <#installation>`_
* `API Cheatsheet & Reference <#api-cheatsheet--reference>`_
* `Preprocessed Datasets & Implemented Algorithms <#preprocessed-datasets--implemented-algorithms>`_
* `Quick Start for Data Processing <#quick-start-for-data-processing>`_
* `Quick Start for Running Predictive Models <#quick-start-for-running-predictive-models>`_
* `Algorithm Benchmark <#algorithm-benchmark>`_
* `Blueprint & Development Plan <#blueprint--development-plan>`_
* `How to Contribute <#how-to-contribute>`_
* `Inclusion Criteria <#inclusion-criteria>`_

----


Installation
^^^^^^^^^^^^

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as PyHealth is updated frequently:

.. code-block:: bash

   pip install pyhealth            # normal install
   pip install --upgrade pyhealth  # or update if needed
   pip install --pre pyhealth      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyhealth.git
   cd pyhealth
   pip install .


**Required Dependencies**\ :


* Python 3.5, 3.6, or 3.7
* combo>=0.0.8
* joblib
* numpy>=1.13
* numba>=0.35
* pandas>=0.25
* scipy>=0.20
* scikit_learn>=0.20
* tqdm
* torch (this should be installed manually)
* xgboost (this should be installed manually)
* xlrd >= 1.0.0
* zipfile36
* PyWavelets
* torch
* torchvision
* xgboost

**Warning 1**\ :
PyHealth has multiple neural network based models, e.g., LSTM, which are
implemented in PyTorch. However, PyHealth does **NOT** install these DL libraries for you.
This reduces the risk of interfering with your local copies.
If you want to use neural-net based models, please make sure PyTorch is installed.
Similarly, models depending on **xgboost**, would **NOT** enforce xgboost installation by default.

----


API Cheatsheet & Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^

Full API Reference: (https://pyhealth.readthedocs.io/en/latest/pyhealth.html). API cheatsheet for most learning models:

* **fit(X_train, X_valida)**\ : Fit a learning model.
* **inference(X)**\ : Predict on X using the fitted estimator.
* **evaluator(y, y^hat)**\ : Model evaluation.

Model load and reload:

* **load_model()**\ : Load the best model so far.


Preprocessed Datasets & Implemented Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**(i) Preprocessed Datasets** (customized data preprocessing function is provided in the example folders):

====================  ================  ======================================================================================================    ======================================================    ===============================================================================================================
Type                  Abbr              Description                                                                                               Processed Function                                        Link
====================  ================  ======================================================================================================    ======================================================    ===============================================================================================================
Sequence: EHR-ICU     MIMIC III         A relational database containing tables of data relating to patients who stayed within ICU.               \\examples\\data_generation\\dataloader_mimic             https://mimic.physionet.org/gettingstarted/overview/
Sequence: EHR-ICU     MIMIC_demo        The MIMIC-III demo database is limited to 100 patients and excludes the noteevents table.                 \\examples\\data_generation\\dataloader_mimic_demo        https://mimic.physionet.org/gettingstarted/demo/
Sequence: EHU-Claim   CMS               DE-SynPUF: CMS 2008-2010 Data Entrepreneurs Synthetic Public Use File                                     \\examples\\data_generation\\dataloader_cms               https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs
Image: Chest X-ray    Pediatric         Pediatric Chest X-ray Pneumonia (Bacterial vs Viral vs Normal) Dataset                                    N/A                                                       https://academictorrents.com/details/951f829a8eeb4d2839c4a535db95078a9175010b
Series: ECG           PhysioNet         AF Classification from a short single lead ECG recording Dataset.                                         N/A                                                       https://archive.physionet.org/challenge/2017/#challenge-data
====================  ================  ======================================================================================================    ======================================================    ===============================================================================================================

You may download the above datasets at the links. The structure of the generated datasets can be found in datasets folder:

* \\datasets\\cms\\x_data\\...csv
* \\datasets\\cms\\y_data\\phenotyping.csv
* \\datasets\\cms\\y_data\\mortality.csv


The processed datasets (X,y) should be put in x_data, y_data correspondingly, to be appropriately digested by deep learning models. We include some sample datasets under \\datasets folder.

**(ii) Machine Learning and Deep Learning Models** :

===================  ================  ========================================  ======================================================================================================  =====  ========================================
Type                 Abbr              Class                                     Algorithm                                                                                               Year   Ref
===================  ================  ========================================  ======================================================================================================  =====  ========================================
Classical Models     RandomForest      pyhealth.models.sequence.rf               Random Forests                                                                                          2000   [#Breiman2001Random]_
Classical Models     XGBoost           pyhealth.models.sequence.xgboost          XGBoost: A scalable tree boosting system                                                                2016   [#Chen2016Xgboost]_
Neural Networks      LSTM              pyhealth.models.sequence.lstm             Long short-term memory                                                                                  1997   [#Hochreiter1997Long]_
Neural Networks      GRU               pyhealth.models.sequence.gru              Gated recurrent unit                                                                                    2014   [#Cho2014Learning]_
Neural Networks      RETAIN            pyhealth.models.sequence.retain           RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism         2016   [#Choi2016RETAIN]_
Neural Networks      Dipole            pyhealth.models.sequence.dipole           Dipole: Diagnosis Prediction in Healthcare via Attention-based Bidirectional Recurrent Neural Networks  2017   [#Ma2017Dipole]_
Neural Networks      tLSTM             pyhealth.models.sequence.tlstm            Patient Subtyping via Time-Aware LSTM Networks                                                          2017   [#Baytas2017tLSTM]_
Neural Networks      RAIM              pyhealth.models.sequence.raim             RAIM: Recurrent Attentive and Intensive Model of Multimodal Patient Monitoring Data                     2018   [#Xu2018RAIM]_
Neural Networks      StageNet          pyhealth.models.sequence.stagenet         StageNet: Stage-Aware Neural Networks for Health Risk Prediction                                        2020   [#Gao2020StageNet]_
===================  ================  ========================================  ======================================================================================================  =====  ========================================

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


Quick Start for Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We propose the idea of standard template, a formalized schema for healthcare datasets.
Ideally, as long as the data is scanned as the template we defined, the downstream
task processing and the use of ML models will be easy and standard. In short, it has the following structure:
**add a figure here**. The dataloader for different datasets can be found in examples/data_generation.
Using `"examples/data_generation/dataloader_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/dataloader_mimic_demo_parallel.py>`_
as an exmaple:

#. First read in patient, admission, and event tables.

   .. code-block:: python


       from pyhealth.utils.utility import read_csv_to_df
       patient_df = read_csv_to_df(os.path.join('data', 'mimic-iii-clinical-database-demo-1.4', 'PATIENTS.csv'))
       admission_df = read_csv_to_df(os.path.join('data', 'mimic-iii-clinical-database-demo-1.4', 'ADMISSIONS.csv'))
       ...

#. Then invoke the parallel program to parse the tables in n_jobs cores.

   .. code-block:: python


       from pyhealth.data.base_mimic import parallel_parse_tables
       all_results = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=True)(
       delayed(parallel_parse_tables)(
            patient_df=patient_df,
            admission_df=admission_df,
            icu_df=icu_df,
            event_df=event_df,
            event_mapping_df=event_mapping_df,
            duration=duration,
            save_dir=save_dir)
        for i in range(n_jobs))

#. The processed sequential data will be saved in the prespecified directory.

   .. code-block:: python

      with open(patient_data_loc, 'w') as outfile:
          json.dump(patient_data_list, outfile)

The provided examples in PyHealth mainly focus on scanning the data tables in the schema we have, and **generate episode datasets**.
For instance, `"examples/data_generation/dataloader_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/dataloader_mimic_demo_parallel.py>`_
demonstrates the basic procedure of processing MIMIC III demo datasets.

#. The next step is to generate episode/sequence data for mortality prediction. See `"examples/data_generation/generate_mortality_prediction_mimic_demo.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/data_generation/generate_mortality_prediction_mimic_demo.py>`_

   .. code-block:: python

      with open(patient_data_loc, 'w') as outfile:
          json.dump(patient_data_list, outfile)

By this step, the dataset has been processed for generating X, y for phenotyping prediction. **It is noted that the API across most datasets are similar**.
One may easily replicate this procedure by calling the data generation scripts in \\examples\\data_generation. You may also modify the parameters in the
scripts to generate the customized datasets.

**Preprocessed datasets are also available at \\datasets\\cms and \\datasets\\mimic**.


----


Quick Start for Running Predictive Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Before running examples, you need the datasets. Please download from the GitHub repository `"datasets" <https://github.com/yzhao062/PyHealth/tree/master/datasets>`_.
You can either unzip them manually or running our script `"00_extract_data_run_before_learning.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/learning_models/00_extract_data_run_before_learning.py>`_

`"examples/learning_models/example_sequence_gpu_mortality.py" <https://github.com/yzhao062/pyhealth/blob/master/examples/learning_models/example_sequence_gpu_mortality.py>`_
demonstrates the basic API of using GRU for mortality prediction. **It is noted that the API across all other algorithms are consistent/similar**.

**If you do not have the preprocessed datasets yet, download the \\datasets folder (cms.zip and mimic.zip) from PyHealth repository, and run \\examples\\learning_models\\extract_data_run_before_learning.py to prepare/unzip the datasets.**


#. Setup the datasets. X and y should be in x_data and y_data, respectively.

   .. code-block:: python

      # load pre-processed CMS dataset
      from pyhealth.data.expdata_generator import sequencedata as expdata_generator

      expdata_id = '2020.0810.data.mortality.mimic'
      cur_dataset = expdata_generator(exp_id=exp_id)
      cur_dataset.get_exp_data(sel_task='mortality', )
      cur_dataset.load_exp_data()


#. Initialize a LSTM model, you may set up the parameters of the LSTM, e.g., n_epoch, learning_rate, etc,.

   .. code-block:: python

      # initialize the model for training
      from pyhealth.models.sequence.lstm import LSTM
      # enable GPU
      expmodel_id = 'test.model.lstm.0001'
      clf = LSTM(expmodel_id=expmodel_id, n_batchsize=20, use_gpu=True, n_epoch=100)

#. Model loading, Load the saved model, default for 'best', maybe can personally set via '0', 'latest', etc.

   .. code-block:: python

      clf.load_model()

#. Model training, parameters are learnt on the train datasets and verified on valid datasets

   .. code-block:: python

      clf.fit(cur_dataset.train, cur_dataset.valid)

#. Model inferring, make prediction on the test datasets

   .. code-block:: python

      clf.inference(cur_dataset.test)
      pred_results = clf.get_results()


#. Evaluation on the model. Multiple metrics are supported.

   .. code-block:: python

      # evaluate the model
      from pyhealth.evaluation.evaluator import func
      r = func(pred_results['hat_y'], pred_results['y'])
      print(r)



Algorithm Benchmark
^^^^^^^^^^^^^^^^^^^

**The comparison among of implemented models** will be made available later
with a benchmark paper. TBA soon :)


Blueprint & Development Plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The long term goal of PyHealth is to become a comprehensive healthcare AI toolkit that supports
beyond EHR data, but also the images and clinical notes.

- The compatibility and the support of OMOP format datasets
- Model persistence (save, load, and portability)
- The release of a benchmark paper with PyHealth


----

Reference
^^^^^^^^^

.. [#Baytas2017tLSTM] Baytas, I.M., Xiao, C., Zhang, X., Wang, F., Jain, A.K. and Zhou, J., 2017, August. Patient subtyping via time-aware lstm networks. In *KDD*.

.. [#Breiman2001Random] Breiman, L., 2001. Random forests. *Machine learning*, 45(1), pp.5-32.

.. [#Chen2016Xgboost] Chen, T. and Guestrin, C., 2016, August. Xgboost: A scalable tree boosting system. In *KDD*.

.. [#Cho2014Learning] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. and Bengio, Y., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

.. [#Choi2016RETAIN] Choi, E., Bahadori, M.T., Sun, J., Kulas, J., Schuetz, A. and Stewart, W., 2016. Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. In Advances in Neural Information Processing Systems (pp. 3504-3512).

.. [#Gao2020StageNet] Gao, J., Xiao, C., Wang, Y., Tang, W., Glass, L.M. and Sun, J., 2020, April. StageNet: Stage-Aware Neural Networks for Health Risk Prediction. In Proceedings of The Web Conference 2020 (pp. 530-540).

.. [#Hochreiter1997Long] Hochreiter, S. and Schmidhuber, J., 1997. Long short-term memory. *Neural computation*, 9(8), pp.1735-1780.

.. [#Ma2017Dipole] Ma, F., Chitta, R., Zhou, J., You, Q., Sun, T. and Gao, J., 2017, August. Dipole: Diagnosis prediction in healthcare via attention-based bidirectional recurrent neural networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1903-1911).

.. [#Xu2018RAIM] Xu, Y., Biswal, S., Deshpande, S.R., Maher, K.O. and Sun, J., 2018, July. Raim: Recurrent attentive and intensive model of multimodal patient monitoring data. In Proceedings of the 24th ACM SIGKDD international conference on Knowledge Discovery & Data Mining (pp. 2565-2573).
