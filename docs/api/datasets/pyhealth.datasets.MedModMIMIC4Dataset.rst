pyhealth.datasets.MedModMIMIC4Dataset
=====================================

MIMIC-IV v1.0 and MIMIC-CXR-JPG 2.0.0 as used by MedMod (Elsharief et al.,
"MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
Health Records and Chest X-Ray Scans", CHIL 2025, PMLR 287;
`paper <https://proceedings.mlr.press/v287/elsharief25a.html>`__,
`code <https://github.com/nyuad-cai/MedMod>`__) for in-hospital mortality,
with MedMod's fixed train / val / test split. Use it with
:class:`~pyhealth.tasks.MedModInHospitalMortality`.

Access to `MIMIC-IV v1.0 <https://physionet.org/content/mimiciv/1.0/>`__
and `MIMIC-CXR-JPG 2.0.0 <https://physionet.org/content/mimic-cxr-jpg/2.0.0/>`__
needs PhysioNet credentialing.
Without it, ``pyhealth.datasets.medmod.synthetic.create_synthetic_samples``
builds synthetic samples with the same contract.

.. autoclass:: pyhealth.datasets.MedModMIMIC4Dataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.MedModSplit
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.split_by_medmod

MedSigLIP image embeddings
--------------------------

For the extension, :mod:`pyhealth.datasets.medmod.medsiglip` replaces the
ResNet-34 image features with those of a frozen ``google/medsiglip-448``
image encoder. Each image is embedded once and saved to one ``.npz`` file,
where the embeddings are looked up by image file name.

MedSigLIP (`google/medsiglip-448 <https://huggingface.co/google/medsiglip-448>`__)
is introduced in Sellergren et al., "MedGemma Technical Report", Nature
Medicine 32, 3695-3704 (2026), https://doi.org/10.1038/s41591-026-04626-w
(preprint arXiv:2507.05201). Its weights are downloaded from the Hugging Face Hub
under the `Health AI Developer Foundations terms of use
<https://developers.google.com/health-ai-developer-foundations/terms>`__.

.. automodule:: pyhealth.datasets.medmod.medsiglip
    :members:

ConVIRT pretraining samples
---------------------------

ConVIRT is pretrained on MedMod's phenotyping train and validation stays, each
with its whole ICU stay of EHR and its latest AP chest X-ray.
:mod:`pyhealth.datasets.medmod.pretraining` builds these samples from the
output of MedMod's extraction scripts and the MIMIC-CXR-JPG metadata
(``load_pretraining_samples``) and turns them into a PyHealth dataset
(``create_pretraining_dataset``).

.. automodule:: pyhealth.datasets.medmod.pretraining
    :members:
