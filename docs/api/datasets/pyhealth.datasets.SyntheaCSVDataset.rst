pyhealth.datasets.SyntheaCSVDataset
===================================

Loads the CSV export of `Synthea <https://github.com/synthetichealth/synthea>`_,
a synthetic patient simulator, through the standard PyHealth dataset API. Point
``root`` at any Synthea CSV directory, or produce one with
:class:`~pyhealth.models.Synthea`.

Tables that Synthea omits because they have no rows are skipped when using the
default table list. Procedure timestamps from older exports (``date`` instead of
``start``) are normalized automatically.

.. autoclass:: pyhealth.datasets.SyntheaCSVDataset
    :members:
    :undoc-members:
    :show-inheritance:
