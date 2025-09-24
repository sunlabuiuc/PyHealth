Installation
============

You could install from PyPi for the older stable version of PyHealth:

.. code-block:: bash

    pip install pyhealth

or our most-updated alpha version from PyPi:

.. code-block:: bash

   pip install pyhealth==2.0a2

or from github source for the latest version of PyHealth:

.. code-block:: bash

   git clone https://github.com/sunlabuiuc/PyHealth.git
   cd pyhealth
   pip install -e .


.. **Required Dependencies**\ :

.. .. code-block:: bash

..     python>=3.8
..     torch>=1.8.0
..     rdkit>=2022.03.4
..     scikit-learn>=0.24.2
..     networkx>=2.6.3
..     pandas>=1.3.2
..     tqdm

**Warning 1**\ :

PyHealth has multiple neural network based models, e.g., LSTM, which are
implemented in PyTorch. However, PyHealth does **NOT** install these DL libraries for you.
This reduces the risk of interfering with your local copies.
If you want to use neural-net based models, please make sure PyTorch is installed.
Similarly, models depending on **xgboost** would **NOT** enforce xgboost installation by default.




**CUDA Setting**\ :

To run PyHealth, you may also need CUDA and cudatoolkit that supports your GPU. `More info <https://developer.nvidia.com/cuda-gpus/>`_

For example, if you use NVIDIA RTX A6000 as your GPU for training, you should install a compatible cudatoolkit using:

.. code-block:: bash

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch.

----