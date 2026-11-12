pyhealth.datasets.VSACDataset
=============================

Clinical value sets from the NLM `Value Set Authority Center (VSAC) <https://vsac.nlm.nih.gov>`_, downloaded through its FHIR ``ValueSet/$expand`` API under a free UMLS license. Each value set is a patient keyed by OID, with a ``valuesets`` event (title, publisher, split) and one ``codes`` event per member code. The dataset reproduces the data pipeline of the benchmark in "Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring" (`arXiv:2604.14616 <https://arxiv.org/abs/2604.14616>`_). Download value sets with :meth:`~pyhealth.datasets.VSACDataset.download_expansions`, which pages past VSAC's 1,000-code limit by default and also fetches each value set's ``purpose`` text. The authors' `RASC scripts <https://github.com/mukhes3/RASC>`_ produce the same layout but keep only the first 1,000 codes.

.. autoclass:: pyhealth.datasets.VSACDataset
    :members:
    :undoc-members:
    :show-inheritance:
