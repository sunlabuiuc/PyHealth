pyhealth.datasets.MSDSegmentationDataset
========================================

Base loader for one Medical Segmentation Decathlon task. It builds metadata
from ``dataset.json`` and leaves pixel loading to ``NiftiProcessor``.

This is an implementation base class; most users should instantiate one of
:class:`~pyhealth.datasets.MSDHippocampusDataset`,
:class:`~pyhealth.datasets.MSDProstateDataset`, or
:class:`~pyhealth.datasets.MSDSpleenDataset`.

.. autoclass:: pyhealth.datasets.MSDSegmentationDataset
    :members:
    :undoc-members:
    :show-inheritance:
