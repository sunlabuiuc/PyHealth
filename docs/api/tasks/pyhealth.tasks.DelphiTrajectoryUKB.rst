pyhealth.tasks.DelphiTrajectoryUKB
====================================

Prefix-to-next-event samples for Delphi-2M from pre-extracted UK Biobank
events (Shmatko et al., `Nature 2025 <https://doi.org/10.1038/s41586-025-09529-3>`_,
`gerstung-lab/Delphi <https://github.com/gerstung-lab/Delphi>`_).
The task subclasses :class:`~pyhealth.tasks.DelphiTrajectoryMEDS` and only
changes the event type and birth codes. UK Biobank is not downloaded.

.. autoclass:: pyhealth.tasks.delphi_trajectory.DelphiTrajectoryUKB
    :members:
    :undoc-members:
    :show-inheritance:
