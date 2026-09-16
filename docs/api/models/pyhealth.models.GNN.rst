pyhealth.models.GNN
===================================

The GNN model (pyhealth trainer does not apply to GNN, refer to the example/ChestXray-image-generation-GAN.ipynb for examples of using GNN model).

.. note::

   ``pyhealth.models.gnn`` no longer seeds the global ``torch``/``numpy``
   RNGs at import time. Previously, ``torch.manual_seed(3)`` and
   ``np.random.seed(1)`` ran as module-level statements, so simply
   importing ``pyhealth.models`` (which imports this module) would
   silently overwrite any seed the caller had already set, regardless of
   whether GCN/GAT were ever used. If you need reproducible GCN/GAT
   weight initialization, seed explicitly right before constructing the
   model instead.

.. autoclass:: pyhealth.models.GAT
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.GCN
    :members:
    :undoc-members:
    :show-inheritance:
