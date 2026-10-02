pyhealth.models.GAMENet
===================================

The separate callable GAMENetLayer and the complete GAMENet model.

GAMENet requires ``drugs_hist`` (nested per-visit drug history, with the
current/target visit already zeroed out, e.g. as produced by
:mod:`pyhealth.tasks.drug_recommendation`) in the dataset's ``input_schema``.
This is used to populate the paper's Dynamic Memory (Eq. 6): each previous
visit's actual administered drugs, retrieved via the query-key temporal
attention in Eq. 7.

.. autoclass:: pyhealth.models.GAMENetLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.GAMENet
    :members:
    :undoc-members:
    :show-inheritance:
