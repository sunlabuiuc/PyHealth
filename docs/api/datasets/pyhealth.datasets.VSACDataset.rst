pyhealth.datasets.VSACDataset
=============================

Clinical value sets from the NLM `Value Set Authority Center (VSAC) <https://vsac.nlm.nih.gov>`_, downloaded through its FHIR ``ValueSet/$expand`` API under a free UMLS license. Each value set is a patient keyed by OID, with a ``valuesets`` event (title, publisher, split) and one ``codes`` event per member code. The dataset reproduces the benchmark of "Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring" (`arXiv:2604.14616 <https://arxiv.org/abs/2604.14616>`_); download the content with the authors' `RASC scripts <https://github.com/mukhes3/RASC>`_.

.. autoclass:: pyhealth.datasets.VSACDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.ValueSetSplit
    :members:
    :show-inheritance:
