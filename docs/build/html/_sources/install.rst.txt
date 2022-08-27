Installation
============

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

**Warning 1**\ :
PyHealth has multiple neural network based models, e.g., LSTM, which are
implemented in PyTorch. However, PyHealth does **NOT** install these DL libraries for you.
This reduces the risk of interfering with your local copies.
If you want to use neural-net based models, please make sure PyTorch is installed.
Similarly, models depending on **xgboost**, would **NOT** enforce xgboost installation by default.

----