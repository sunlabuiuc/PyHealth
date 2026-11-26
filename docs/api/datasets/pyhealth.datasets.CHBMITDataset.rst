pyhealth.datasets.CHBMITDataset
===================================

Dataset is available at https://physionet.org/content/chbmit/1.0.0/

The CHB-MIT Scalp EEG Database is a collection of pediatric scalp EEG
recorded at Children's Hospital Boston. Each case is a folder ``chbXX``
with EDF recordings and a ``chbXX-summary.txt`` that lists the seizures.
Case ``chb21`` is the same person as ``chb01``, recorded about 1.5 years
later.

The dataset writes one metadata row per EDF. The default task,
``SeizureDetectionCHBMIT``, cuts 10-second windows on the 16 bipolar
channels used by CBraMod and TFM-Tokenizer. Choose the train, validation
and test subjects with ``SubjectSplit`` or
``CHBMITDataset.default_subject_split()`` (``chb01``-``chb20`` /
``chb21``-``chb22`` / ``chb23``-``chb24``).

.. autoclass:: pyhealth.datasets.CHBMITDataset
    :members:
    :undoc-members:
    :show-inheritance:
