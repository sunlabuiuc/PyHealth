pyhealth.models.GNN
===================================

The GNN model (pyhealth trainer does not apply to GNN, refer to the example/ChestXray-image-generation-GAN.ipynb for examples of using GNN model).

.. note::
   ``pyhealth.models.gnn`` no longer seeds the global ``torch``/``numpy``
   random generators at import time. Previously ``torch.manual_seed(3)`` and
   ``np.random.seed(1)`` ran when ``pyhealth.models`` was imported,
   overwriting any seed set before the import. For reproducible GCN/GAT
   weights, seed right before constructing the model.

.. autoclass:: pyhealth.models.GAT
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.GCN
    :members:
    :undoc-members:
    :show-inheritance:
