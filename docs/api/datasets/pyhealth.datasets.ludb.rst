pyhealth.datasets.ludb
======================

Native PyHealth adapter for externally prepared, frozen LUDB 1.0.1 windows.
Contributors: Thi Van Trinh Tran and Zexi Li.

Sources and scope
-----------------

Obtain the original 200-record, 12-lead dataset from
`PhysioNet LUDB 1.0.1 <https://physionet.org/content/ludb/1.0.1/>`_.
The data publication is Kalyakulina et al., *LUDB: A New Open-Access Validation
Tool for Electrocardiogram Delineation Algorithms*, IEEE Access (2020).
Follow the dataset's current access, attribution and license conditions.
Model motivation: `MedTsLLM <https://arxiv.org/abs/2408.07773>`_.

This contribution consumes frozen artifacts from the research preprocessing
pipeline. It does **not** implement raw WFDB annotation-to-mask preparation or
choose a mask/exclusion policy. The original preprocessing remains external.
Do not describe this adapter as an annotation generator or a full replacement
for that preprocessing workflow.

Required inputs
---------------

* ``records_root/<patient_id>.hea`` and ``.dat``: original LUDB signals,
  5000 samples by 12 leads at 500 Hz; no filtering or downsampling.
* ``mask_root/masks/<patient_id>.npz``: integer ``labels`` (12, 5000),
  binary ``ignore_mask`` (12, 5000), and ordered ``lead_names``.
* Window CSV: ``patient_id,lead,lead_index,split,window_index,start_sample,
  end_sample_exclusive,window_length,has_supervised_points``. Zero-based sample
  positions, 512-sample windows. Rows are not silently excluded or regenerated.
* Patient CSV: ``patient_id,split``; exactly one row per patient and explicit
  ``train``/``test`` membership. The research analysis used a fixed 80/20 split,
  not CV. The adapter also accepts tiny supplied splits for synthetic tests.
* Optional metadata CSV with unique numeric ``ID``. Every field, including
  identifiers and QC fields, is retained in the historical JSON context.

Membership must agree between the two CSVs. The adapter does not select which
mask artifact is authoritative. Preserve preparation summaries/checksums and
review label/ignore counts before scientifically comparing separate runs.
The previously completed research runs and this new port must not be treated
as numerically equivalent without a separately approved parity check.

.. code-block:: python

    from pyhealth.datasets import LUDBDataset

    raw = LUDBDataset(
        manifest_path="/project/window_manifest.csv",
        records_root="/project/ludb/data",
        mask_root="/project/prepared",
        split_manifest_path="/project/patient_split.csv",
        metadata_path="/project/metadata_prepared.csv",
        cache_dir="/project/controlled_cache",
    )
    train, test = raw.set_task(split=raw.manifest_split)

No synthetic clinical timestamps are invented. Source files are read-only;
PyHealth writes derived caches only under ``cache_dir``. Frozen mask/metadata
hashes and record size/mtime fingerprints enter task cache identity. Changing
signals while preserving size and mtime is not detected by the record
fingerprint: preserve external content hashes for a rigorous archive.

.. autoclass:: pyhealth.datasets.ludb.LUDBDataset
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.ludb.LUDBManifestSplit
    :members:
    :show-inheritance:
