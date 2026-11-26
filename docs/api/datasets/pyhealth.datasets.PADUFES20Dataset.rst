pyhealth.datasets.PADUFES20Dataset
==================================

Overview
--------

PAD-UFES-20: 2,298 smartphone (clinical, not dermoscopy) photos of 1,641 skin
lesions from 1,373 patients in Brazil, with six diagnoses (ACK, BCC, MEL, NEV,
SCC, SEK). Every column of ``metadata.csv`` is kept as an event attribute,
including all 21 clinical columns (age, gender, Fitzpatrick type, region,
diameters, symptom flags, smoking, drinking, pesticide exposure, family
background, cancer history, piped water and sewage). ``patient_id`` is the PAD
patient id, so patient-level splits keep all of a patient's lesions together.

Getting the data
----------------

- **Source:** `PAD-UFES-20 on Mendeley Data <https://data.mendeley.com/datasets/zr7vgbcyr2/1>`_
  (DOI `10.17632/zr7vgbcyr2.1 <https://doi.org/10.17632/zr7vgbcyr2.1>`_).
- **Paper:** Pacheco AGC, Lima GR, Salomao AS, et al. PAD-UFES-20: a skin
  lesion dataset composed of patient data and clinical images collected from
  smartphones. *Data in Brief* 32, 106221 (2020).
  https://doi.org/10.1016/j.dib.2020.106221
- **Access:** open. No account, registration or data-use agreement.
- **Licence:** CC BY 4.0 (commercial use and derivatives allowed, with
  attribution).
- **Files** (3.6 GB in total): ``metadata.csv`` (316 KB), ``imgs_part_1.zip``
  (1.2 GB), ``imgs_part_2.zip`` (1.1 GB) and ``imgs_part_3.zip`` (1.2 GB).
  Either click **Download All** on the Mendeley Data page (one zip of the same
  files; unzip it, then move ``metadata.csv`` into ``root`` and the three image
  archives into ``root/images/``), or fetch each file from its permanent
  public link:

  .. code-block:: bash

      BASE=https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files
      mkdir -p pad_ufes_20/images && cd pad_ufes_20
      wget -O metadata.csv $BASE/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded
      wget -O images/imgs_part_1.zip $BASE/1cc2f71f-20a2-412d-b746-a9b9bc20c966/file_downloaded
      wget -O images/imgs_part_2.zip $BASE/559a60ed-5504-475d-996c-6a8bc253b5e7/file_downloaded
      wget -O images/imgs_part_3.zip $BASE/34dcdf8e-e5f1-4b35-aa0b-5135051ff852/file_downloaded
      cd images && for f in imgs_part_*.zip; do unzip -q "$f"; done

- **Unpack:** each image archive extracts to an ``imgs_part_<n>/`` folder of
  PNGs inside ``images/``. The zips can then be deleted. There is no
  ``download=True``.
- **root layout** (PNGs may also sit directly in ``images/``):

  .. code-block:: text

      pad_ufes_20/              <- root
          metadata.csv
          images/
              imgs_part_1/PAT_100_393_595.png ...
              imgs_part_2/...
              imgs_part_3/...

The derived metadata CSV is written under the PyHealth cache, never into
``root``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.PADUFES20Dataset
    :members:
    :undoc-members:
    :show-inheritance:
