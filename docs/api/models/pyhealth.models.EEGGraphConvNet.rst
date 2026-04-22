pyhealth.models.EEGGraphConvNet
================================

The EEGGraphConvNet model is based on the EEG-GCNN architecture introduced in:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://arxiv.org/abs/2011.10432

A shallow two-layer Graph Convolutional Network (GCN) for EEG-based
neurological disease classification. Node features are six PSD band-power
values per bipolar electrode channel. Edge weights encode a blended
geodesic distance / spectral coherence connectivity measure produced by
:class:`~pyhealth.datasets.EEGGCNNDataset`.

.. autoclass:: pyhealth.models.EEGGraphConvNet
    :members:
    :undoc-members:
    :show-inheritance:
