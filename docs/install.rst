Installation
============

**Recommended Installation (Alpha Version)**

We recommend installing the latest alpha version from PyPi, which offers significant improvements in performance:

.. code-block:: bash

   pip install pyhealth==2.0a10

This version includes optimized implementations and enhanced features compared to the legacy version.

**Legacy Version**

The older stable version is still available for backward compatibility, though it may have performance limitations:

.. code-block:: bash

    pip install pyhealth

**Note:** The legacy version (1.x) should still work for most use cases, but we recommend upgrading to 2.0a10 for better performance.

**For Contributors and Developers**

If you are contributing to PyHealth or need the latest development features, install from GitHub source:

.. code-block:: bash

   git clone https://github.com/sunlabuiuc/PyHealth.git
   cd PyHealth
   pip install -e .

This approach is recommended for developers as it allows you to modify the code and immediately see changes without reinstalling.


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