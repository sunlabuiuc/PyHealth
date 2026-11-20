pyhealth.datasets.PBCSeqDataset
===============================

Overview
--------

The Mayo Clinic primary biliary cholangitis (PBC) sequential dataset (``pbcseq``
in the R ``survival`` package) follows 312 trial patients over 1,945 visits, with
repeated liver labs and one time-to-event outcome per patient (death, liver
transplant, or censoring). It is a public substitute cohort for reproducing the
Model for Allograft Survival (MAS; Gao et al., MLHC 2024), whose registry data are
restricted. Use it with :class:`~pyhealth.tasks.LongitudinalLiverSurvival`.

.. autoclass:: pyhealth.datasets.PBCSeqDataset
    :members:
    :undoc-members:
    :show-inheritance:
