pyhealth.datasets.LNDbDataset
=============================

LNDb lung nodule CT dataset. The loader indexes the merged radiologist
findings of the LNDb training set, one event per finding with its world
coordinate, size, nodule / non-nodule label and official cross-validation
fold, and reads the ``.mhd`` / ``.raw`` CTs with a built-in numpy reader
(SimpleITK is used when installed).

.. autoclass:: pyhealth.datasets.LNDbDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.lndb.read_mhd

.. autoclass:: pyhealth.datasets.lndb.MetaImage
    :members:
