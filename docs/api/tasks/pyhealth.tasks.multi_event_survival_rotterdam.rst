Multi-event survival (Rotterdam)
================================

Overview
--------

:class:`~pyhealth.tasks.MultiEventSurvivalRotterdam` turns each patient of
:doc:`../datasets/pyhealth.datasets.RotterdamDataset` into one sample with a
dense feature vector and two events, recurrence then death:

- ``features``: ten values in this order: ``year``, ``age``, ``meno``,
  ``grade``, ``nodes``, ``pgr``, ``er``, ``hormon``, ``chemo`` and
  ``size_mm``, with tumour size coded as 10, 35 or 75 mm for ``"<=20"``,
  ``"20-50"`` and ``">50"``. With ``use_treatment=False`` the treatment
  indicators ``hormon`` and ``chemo`` are dropped, leaving eight features in
  the same order; the labels do not change.
- ``event_times``: ``[rtime, dtime]`` in days, exactly as recorded. A
  recurrence censored before a later death keeps its own ``rtime``, and tied
  times are kept.
- ``event_indicators``: ``[recur, death]``, 1 for an observed event and 0 for
  censoring at that time.

Every value is validated by
:func:`~pyhealth.datasets.rotterdam.parse_rotterdam_record`. Features are not
standardized here: fit any standardization on training patients only, after
splitting. Like PyHealth's other duration/event survival tasks, the labels use
the ``"tensor"`` processor, so a model consuming them sets ``mode=None`` and
supplies its own censored likelihood to the standard Trainer, as
:doc:`../models/pyhealth.models.MENSA` does.

Paper
-----

The features, size coding and event order follow the Rotterdam setup of
Lillelund et al., `MENSA: A Multi-Event Network for Survival Analysis with
Trajectory-based Likelihood Estimation`_ (ML4H 2025, PMLR 297), and its
`reference code`_.

Usage
-----

.. code-block:: python

    from pyhealth.datasets import RotterdamDataset
    from pyhealth.tasks import MultiEventSurvivalRotterdam

    dataset = RotterdamDataset(root="/path/to/folder")
    samples = dataset.set_task(MultiEventSurvivalRotterdam())  # ten features
    sample = samples[0]
    sample["features"].shape      # torch.Size([10])
    sample["event_times"]         # tensor([rtime, dtime]), days
    sample["event_indicators"]    # tensor([recur, death])

    # Without the treatment indicators (eight features):
    samples_8 = dataset.set_task(MultiEventSurvivalRotterdam(use_treatment=False))

API reference
-------------

.. autoclass:: pyhealth.tasks.MultiEventSurvivalRotterdam
    :members:
    :show-inheritance:

.. _MENSA\: A Multi-Event Network for Survival Analysis with Trajectory-based Likelihood Estimation: https://proceedings.mlr.press/v297/lillelund26a.html
.. _reference code: https://github.com/thecml/mensa
