pyhealth.datasets.EEGGCNNDataset
=================================

Dataset is available at https://figshare.com/articles/dataset/EEG-GCNN_Augmenting_Electroencephalogram-based_Neurological_Disease_Diagnosis_using_a_Domain-guided_Graph_Convolutional_Neural_Network/13251509

The EEG-GCNN dataset is derived from the pre-computed features introduced in:

    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    *Machine Learning for Health (ML4H) workshop, NeurIPS 2020*.
    https://arxiv.org/abs/2011.10432

Each EEG recording is segmented into fixed-length windows. Every window is
represented as a fully-connected graph over 8 bipolar EEG electrode pairs.
Node features are six PSD band-power values (delta, theta, alpha, beta,
low-gamma, high-gamma) per electrode. Edge weights encode a blended
connectivity measure combining geodesic distance between electrode positions
and spectral coherence between electrode signals, controlled by a mixing
parameter ``alpha``.

The dataset supports neurological disease classification (diseased vs.
healthy) and is designed for use with graph neural network models such as
:class:`~pyhealth.models.EEGGraphConvNet` and
:class:`~pyhealth.models.EEGGATConvNet`.

.. autoclass:: pyhealth.datasets.EEGGCNNDataset
    :members:
    :undoc-members:
    :show-inheritance:
