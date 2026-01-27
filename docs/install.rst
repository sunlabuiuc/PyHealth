Installation
============

**Python Version Requirement**

PyHealth 2.0 requires **Python 3.12 or higher** (up to Python 3.13). This is a hard requirement due to dependencies on modern Python features for parallel processing and memory management.

.. code-block:: bash

   # Verify your Python version
   python --version  # Should be 3.12.x or 3.13.x

**Recommended Installation (Latest Release)**

Install the latest PyHealth 2.0 release from PyPI:

.. code-block:: bash

   pip install pyhealth

This version includes significant performance improvements, dynamic memory support, parallelized processing, multimodal dataloaders, and many new features.

**Legacy Version**

The older stable version (1.16) is still available for backward compatibility and supports Python 3.9+:

.. code-block:: bash

   pip install pyhealth==1.16

**Note:** The legacy version (1.16) should still work for most use cases, but we recommend using PyHealth 2.0 for better performance and new features.

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

**Platform-Specific Notes**

**Windows Subsystem for Linux (WSL)**

When using PyHealth on WSL, you **may need to** disable swap memory due to a bug in how Dask interacts with WSL's memory management when memory runs out. This prevents performance issues and potential crashes.

**Method 1: Using WSL Settings App (Windows 11)**

1. Open the WSL Settings app in Windows
2. Navigate to Memory and Processor settings
3. Set Swap size to 0 MB
4. Apply changes and restart WSL

**Method 2: Manual Configuration**

1. Open PowerShell as Administrator
2. Create or edit `%UserProfile%\.wslconfig` file
3. Add the following configuration:

.. code-block:: ini

    [wsl2]
    swap=0

4. Restart WSL by running in PowerShell: ``wsl --shutdown``

**Other Platforms**

PyHealth should work without additional configuration on:

- Linux (native)
- macOS
- Windows (with proper Python installation)

----