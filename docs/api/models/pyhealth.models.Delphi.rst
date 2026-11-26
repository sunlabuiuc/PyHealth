pyhealth.models.Delphi
=======================

Delphi-2M (Shmatko et al., `Nature 2025 <https://doi.org/10.1038/s41586-025-09529-3>`_)
is a generative causal transformer for longitudinal disease trajectories.
Continuous age replaces discrete positions, co-occurring timestamps can be
masked, and the loss joins next-event cross-entropy with an exponential
competing-risks model of the waiting time. Reference implementation:
`gerstung-lab/Delphi <https://github.com/gerstung-lab/Delphi>`_.

Train it on :class:`~pyhealth.tasks.DelphiTrajectoryMEDS` or
:class:`~pyhealth.tasks.DelphiTrajectoryMIMIC4`.

.. autoclass:: pyhealth.models.Delphi
    :members:
    :undoc-members:
    :show-inheritance:
