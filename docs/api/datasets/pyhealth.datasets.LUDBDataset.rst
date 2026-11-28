pyhealth.datasets.LUDBDataset
=============================

Overview
--------

The Lobachevsky University Electrocardiography Database (LUDB) has 200
ten-second, 12-lead ECGs sampled at 500 Hz, one per subject. For every lead,
cardiologists marked where each P wave, QRS complex and T wave starts and
ends, which makes LUDB the standard public dataset for ECG delineation.
``ludb.csv`` gives each subject's sex, age, rhythm and electric axis.

LUDB is open access on PhysioNet. Download the ZIP of version 1.0.1 (about
15 MB), unzip it and point ``root`` at the folder that holds ``ludb.csv``
and ``data/``. Reading the records needs the ``wfdb`` package
(``pip install wfdb``).

Each record is one patient. The dataset writes one metadata row per record
(the WFDB record path plus sex, age, rhythm and electric axis). The default
task, :class:`~pyhealth.tasks.ECGDelineationLUDB`, turns each lead into a
filtered signal and a point-wise P / QRS / T mask.

Source: `LUDB on PhysioNet <https://physionet.org/content/ludb/1.0.1/>`_
(Kalyakulina et al., IEEE Access 2020,
`doi:10.1109/ACCESS.2020.3029211 <https://doi.org/10.1109/ACCESS.2020.3029211>`_)

.. code-block:: python

    from pyhealth.datasets import LUDBDataset

    dataset = LUDBDataset(root="/path/to/ludb/1.0.1")
    dataset.stats()
    samples = dataset.set_task()  # ECGDelineationLUDB on all 12 leads

API Reference
-------------

.. autoclass:: pyhealth.datasets.LUDBDataset
    :members:
    :undoc-members:
    :show-inheritance:
