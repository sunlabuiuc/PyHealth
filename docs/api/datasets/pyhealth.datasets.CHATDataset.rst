pyhealth.datasets.CHATDataset
=============================

Overview
--------

The Childhood Adenotonsillectomy Trial (CHAT) recorded sleep studies
(polysomnography) of children aged 5 to 9.9 years with mild to moderate
obstructive sleep apnea, randomized to early adenotonsillectomy or watchful
waiting. Each child has a baseline study and, if randomized, a follow-up
study about 7 months later; a non-randomized group was also recorded. Each
study is an EDF recording of EEG, EOG, EMG, ECG and breathing channels, with
scored sleep stages and respiratory events in an NSRR XML file. Access is
through NSRR and needs an approved data request.

Point ``root`` at the downloaded ``chat`` folder, the one holding
``polysomnography/`` and ``datasets/``. The dataset writes one metadata row
per study, with the visit, the EDF and annotation paths, the sampling rate,
duration and channel labels of the recording, the number of apneas and
hypopneas, an apnea-hypopnea index counted from the annotations, and the
child's age, sex and race from the subject table of the visit. A child can
have a baseline and a follow-up study, so split by patient.

Source: `NSRR chat <https://sleepdata.org/datasets/chat>`_, described in
`Marcus et al., N Engl J Med 368:2366-2376, 2013
<https://doi.org/10.1056/NEJMoa1215881>`_

API Reference
-------------

.. autoclass:: pyhealth.datasets.CHATDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.chat.read_chat_annotations

.. autofunction:: pyhealth.datasets.chat.summarize_chat_annotations
