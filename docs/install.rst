Installation
============

You could install our package by:

.. code-block:: bash

   git clone https://github.com/ycq091044/PyHealth-OMOP.git
   cd pyhealth
   pip install .


**Required Dependencies**\ :


* Python 3.5, 3.6, 3.7 or 3.8
* numpy>=1.13
* numba>=0.35
* pandas>=0.25
* scikit_learn>=0.20
* torch==1.12
* pytorch-lightning==1.6

**Warning 1**\ :
PyHealth has multiple neural network based models, e.g., LSTM, which are
implemented in PyTorch. However, PyHealth does **NOT** install these DL libraries for you.
This reduces the risk of interfering with your local copies.
If you want to use neural-net based models, please make sure PyTorch is installed.
Similarly, models depending on **xgboost**, would **NOT** enforce xgboost installation by default.


**CUDA Setting**\ :

To run PyHealth, you also need CUDA and cudatoolkit that support your GPU well. `More info <https://developer.nvidia.com/cuda-gpus/>`_

For example, if you use NVIDIA RTX A6000 as your GPU for training, you should install a compatible cudatoolkit using:

.. code-block:: bash

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch.

----

.. toctree::
   :maxdepth: 4

   pyhealth.data