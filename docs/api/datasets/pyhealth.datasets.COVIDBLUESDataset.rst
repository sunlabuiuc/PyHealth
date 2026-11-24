pyhealth.datasets.COVIDBLUESDataset
===================================

COVID-BLUES bluepoint lung ultrasound dataset: 362 videos of 63 COVID-19
suspects (33 PCR-positive, 30 negative) at six BLUE points. The loader reads
pre-extracted video frames (or decodes the videos with an optional decoder)
and labels each frame with the patient's COVID-19 PCR result.

Source: `NinaWie/COVID-BLUES on GitHub <https://github.com/NinaWie/COVID-BLUES>`_
(mirror: `Hugging Face <https://huggingface.co/datasets/jannisborn/COVID-BLUES>`_).
Paper: Wiedemann et al., "COVID-BLUeS - A prospective study on the value of
AI in lung ultrasound analysis", IEEE Journal of Biomedical and Health
Informatics 29(9), 6301-6310 (2025), https://doi.org/10.1109/JBHI.2025.3543686 .

Getting the data
----------------

- **Access**: open, no account, registration or agreement.
- **Download** the whole repository (about 70 MB: 362 ``.mp4`` videos plus
  CSVs, no Git LFS):

  .. code-block:: bash

      git clone https://github.com/NinaWie/COVID-BLUES

  or download https://github.com/NinaWie/COVID-BLUES/archive/refs/heads/main.zip
  and unzip it (it unpacks to ``COVID-BLUES-main/``, which is then ``root``).
- **Frames**: pass ``decode_videos=True`` (needs ``pip install av`` or
  ``pip install opencv-python``), or extract them with ffmpeg, one folder per
  video stem:

  .. code-block:: bash

      cd COVID-BLUES
      for f in lus_videos/*.mp4; do
          s=$(basename "$f" .mp4); mkdir -p ../frames/$s
          ffmpeg -loglevel error -i "$f" ../frames/$s/%06d.png
      done

- **Layout**:

  .. code-block:: text

      COVID-BLUES/                     # root
          clinical_variables.csv       # patient_id, cov_test, ...
          severity.csv                 # optional; video_file, Severity Score
          lus_videos/
              patient_1_L1.mp4
              patient_15_L2_2.mp4      # _2: repeated recording
              ...
      frames/                          # frames_dir
          patient_1_L1/
              000001.png
              ...

  Frame folders must be named after the video stems. ``severity.csv``'s
  ``video_file`` values are matched to those stems (a trailing ``.mp4`` is
  ignored); if none match, a warning is logged.
- **Licence**: CC BY-NC-ND 4.0: non-commercial use only, with attribution,
  and adapted material (which may include extracted frames) must not be
  shared. Keep extracted frames local.

API Reference
-------------

.. autoclass:: pyhealth.datasets.COVIDBLUESDataset
    :members:
    :undoc-members:
    :show-inheritance:
