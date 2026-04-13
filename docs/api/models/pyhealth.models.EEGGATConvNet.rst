pyhealth.models.EEGGATConvNet
==============================

The EEGGATConvNet model adapts the EEG-GCNN architecture introduced in:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://arxiv.org/abs/2011.10432

A shallow two-layer Graph Attention Network (GAT) for EEG-based
neurological disease classification. It replaces the GCN convolutions of
:class:`~pyhealth.models.EEGGraphConvNet` with multi-head attention layers,
allowing the model to learn node-level attention weights over the electrode
graph. Edge weights from :class:`~pyhealth.datasets.EEGGCNNDataset` are
passed as edge features to bias the attention scores.

.. autoclass:: pyhealth.models.EEGGATConvNet
    :members:
    :undoc-members:
    :show-inheritance:
