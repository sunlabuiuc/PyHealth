pyhealth.processors.SequenceProcessorV2
=======================================

:class:`~pyhealth.processors.SequenceProcessorV2` (registered as
``"sequence_v2"``) processes sequential, hierarchical, and temporal clinical
records—categorical medical codes, continuous physiological measurements, and
timestamps at any nesting depth—into packed tensors without padding or
truncation.

Overview: Packed Records & Offsets at a Glance
----------------------------------------------

Instead of padding variable-length clinical encounters into rectangular arrays,
:class:`~pyhealth.processors.SequenceProcessorV2` packs all records at a level
contiguously into 1D/ND payload tensors and tracks group boundaries with a 1D
``offsets`` tensor (Compressed Sparse Row / jagged layout):

.. code-block:: text

    Raw visits:     visit 0: ["428.0", "401.9"]   visit 1: []   visit 2: ["250.00", "584.9"]
                                 │        │                        │         │
    Packed tokens:  tensor([     2,       3,                       4,        5     ])   # N_0 = 4 codes
    CSR offsets:    tensor([0,                  2,            2,                  4])   # N_1 + 1 = 4 bounds

    Slice visit j:  tokens[offsets[j] : offsets[j + 1]]
      j = 0 -> tokens[0:2] = tensor([2, 3])
      j = 1 -> tokens[2:2] = tensor([])        # empty visit: offsets[1] == offsets[2], no <pad> needed
      j = 2 -> tokens[2:4] = tensor([4, 5])

In a :class:`~pyhealth.tasks.BaseTask`, a feature's hierarchy is configured
from the innermost atomic level (level ``0``) outward to level ``L``:

.. code-block:: python

    input_schema = {
        "diagnoses": (
            "sequence_v2",
            {"levels": [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]},
        ),
        "vitals": (
            "sequence_v2",
            {"levels": [{"ordering": "series", "values": [2]}]},
        ),
    }


Glossary
--------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Term
     - Definition
   * - **Level** (``l = 0 .. L``) & **depth** (``L``)
     - Hierarchy tier within a sample, ordered innermost-first in ``levels``: level ``0`` holds atomic records (e.g., codes, lab readings, word tokens); level ``L`` (``proc.depth``) is the outermost grouping tier within a sample (e.g., visits or hospital admissions).
   * - :class:`~pyhealth.processors.Records`
     - Plain ``dict`` returned by a task for one level, holding columnar sequences: ``tokens``, ``values``, ``time``, and ``children`` (a list of child ``Records`` one level down; omitted at level ``0``).
   * - **Record count** (``N_l``)
     - Total number of records at level ``l`` packed across one sample (in :meth:`~pyhealth.processors.SequenceProcessorV2.process`) or across an entire batch (in :meth:`~pyhealth.processors.SequenceProcessorV2.collate`).
   * - **Group**
     - The contiguous slice of level-``l`` records belonging to a single parent record ``j`` at level ``l + 1``.
   * - **Role**
     - One tensor component emitted at level ``l``: ``"tokens"``, ``"values"``, ``"mask"``, ``"time"``, or ``"offsets"``.
   * - ``offsets``
     - 1D ``int64`` tensor of shape ``[N_{l+1} + 1]`` partitioning level ``l`` into level ``l + 1`` groups: parent ``j`` owns ``records_l[offsets[j] : offsets[j + 1]]``, with ``offsets[0] == 0`` and ``offsets[-1] == N_l``.
   * - **Token namespaces** (``K``)
     - Number of independent categorical vocabularies per record at a level (``<pad> = 0`` and ``<unk> = 1`` reserved in each). Shape is ``[N_l]`` when ``K == 1`` and ``[N_l, K]`` when ``K > 1``.
   * - **Value shape** (``S_l``)
     - Trailing dimensions of continuous ``float32`` measurements per record (``[N_l, *S_l]``). Scalar values use ``S_l = [1]`` and emit shape ``[N_l, 1]``.
   * - ``ordering``
     - Semantic order of records within each group: ``"set"`` (unordered; no ``time``), ``"sequence"`` (discrete step order; no ``time``), or ``"series"`` (explicit timestamps; ``time`` required).
   * - ``spatial``
     - One ``bool`` per axis of ``S_l``: ``False`` marks unordered channel/feature axes; ``True`` marks ordered continuous spatial or high-frequency waveform axes.
   * - ``schema()`` vs. ``batch_schema()``
     - Tuple of ``(level, role)`` keys matching the positional tensor tuple returned by :meth:`~pyhealth.processors.SequenceProcessorV2.process` vs. :meth:`~pyhealth.processors.SequenceProcessorV2.collate` (which appends batch boundaries ``(L, "offsets")`` of shape ``[B + 1]``).


Minimal Examples & Batching
---------------------------

Each example below introduces one concept and prints every output tensor in
full. (In end-to-end pipelines, ``dataset.set_task()`` and
``create_sample_dataset()`` call ``fit()``, ``process()``, and ``collate()``
automatically.)

1. Flat Code Sequence (1 Level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each token namespace reserves ``<pad> = 0`` and ``<unk> = 1``. Codes unseen
during ``fit()`` map to ``1`` (``<unk>``) without truncation:

.. code-block:: python

    from pyhealth.processors import SequenceProcessorV2

    flat_proc = SequenceProcessorV2(levels=[{"ordering": "sequence", "tokens": 1}])
    flat_proc.fit([{"diagnoses": {"tokens": ["428.0", "401.9"]}}], "diagnoses")

    flat_proc.vocab(0)
    # {'<pad>': 0, '<unk>': 1, '428.0': 2, '401.9': 3}
    flat_proc.schema()
    # ((0, 'tokens'),)

    # "250.00" was not seen in fit() -> maps to <unk> (1)
    (tokens,) = flat_proc.process({"tokens": ["428.0", "401.9", "428.0", "250.00"]})
    # tokens: tensor([2, 3, 2, 1])

2. Visits of Codes (2 Levels + ``offsets``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grouping unordered diagnosis codes (level ``0``) into an ordered sequence of
visits (level ``1``) adds ``(0, "offsets")``:

.. code-block:: python

    from pyhealth.processors import Records, SequenceProcessorV2

    visit_proc = SequenceProcessorV2(
        levels=[
            {"ordering": "set", "tokens": 1},  # Level 0: codes within a visit
            {"ordering": "sequence"},          # Level 1: sequence of visits
        ]
    )
    visit_proc.fit(
        [{"visits": {"children": [{"tokens": ["428.0", "401.9"]}, {"tokens": ["250.00"]}]}}],
        "visits",
    )
    visit_proc.schema()
    # ((0, 'tokens'), (0, 'offsets'))

    sample_visits: Records = {
        "children": [
            {"tokens": ["428.0", "401.9"]},  # Visit 0: 2 codes
            {"tokens": []},                  # Visit 1: empty follow-up
            {"tokens": ["250.00"]},          # Visit 2: 1 code
        ]
    }
    tokens, offsets = visit_proc.process(sample_visits)
    # tokens:  tensor([2, 3, 4])
    # offsets: tensor([0, 2, 2, 3])

    # Slice any visit j via tokens[offsets[j] : offsets[j + 1]]:
    # j = 0 -> tokens[0:2] == tensor([2, 3])
    # j = 1 -> tokens[2:2] == tensor([], dtype=torch.int64)
    # j = 2 -> tokens[2:3] == tensor([4])

3. Timestamped Numeric Values (1 Level + ``values`` & ``time``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``"series"`` level pairs each record with a finite, non-decreasing timestamp
in the task's chosen relative unit (e.g., hours since ICU admission):

.. code-block:: python

    from pyhealth.processors import SequenceProcessorV2

    ts_proc = SequenceProcessorV2(levels=[{"ordering": "series", "values": [2]}])
    ts_proc.fit([], "vitals")  # No tokens or dataset statistics -> nothing to learn
    ts_proc.schema()
    # ((0, 'values'), (0, 'time'))

    values, time = ts_proc.process(
        {
            "time": [0.0, 1.5, 4.0],
            "values": [[80.0, 75.0], [84.0, 78.0], [78.0, 72.0]],
        }
    )
    # values: tensor([[80., 75.],
    #                 [84., 78.],
    #                 [78., 72.]])
    # time:   tensor([0.0000, 1.5000, 4.0000])

4. Batching & Offset Shifting (``collate()``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When batching ``B`` patients, :meth:`~pyhealth.processors.SequenceProcessorV2.collate`
(called automatically by :func:`~pyhealth.datasets.get_dataloader`) inserts zero
padding. Instead, it:

1. Concatenates payload tensors (``tokens``, ``values``, ``mask``, ``time``) along dimension 0.
2. Shifts each patient's ``offsets_l[1:]`` by the cumulative ``N_l`` of preceding patients.
3. Appends outer ``batch_offsets`` ``(L, "offsets")`` of shape ``[B + 1]`` marking each patient's slice of level ``L``.

.. code-block:: python

    # Patient 0: 2 visits, 3 codes total -> offsets [0, 2, 3]
    p0 = visit_proc.process(
        {"children": [{"tokens": ["428.0", "401.9"]}, {"tokens": ["250.00"]}]}
    )
    # p0 == (tensor([2, 3, 4]), tensor([0, 2, 3]))

    # Patient 1: 2 visits, 2 codes total -> offsets [0, 1, 2]
    p1 = visit_proc.process(
        {"children": [{"tokens": ["401.9"]}, {"tokens": ["250.00"]}]}
    )
    # p1 == (tensor([3, 4]), tensor([0, 1, 2]))

    visit_proc.batch_schema()
    # ((0, 'tokens'), (0, 'offsets'), (1, 'offsets'))

    b_tokens, visit_offsets, batch_offsets = visit_proc.collate([p0, p1])
    # b_tokens:      tensor([2, 3, 4, 3, 4])  — 3 + 2 = 5 codes concatenated
    # visit_offsets: tensor([0, 2, 3, 4, 5])  — [0, 2, 3] ++ ([1, 2] + 3) across 4 total visits
    # batch_offsets: tensor([0, 2, 4])        — Patient 0 owns visits [0:2], Patient 1 owns visits [2:4]

    # Recovering patient b's visits and codes inside a model's forward():
    b = 1
    v_start, v_end = int(batch_offsets[b]), int(batch_offsets[b + 1])  # 2, 4
    c_start, c_end = int(visit_offsets[v_start]), int(visit_offsets[v_end])  # 3, 5
    patient_b_codes = b_tokens[c_start:c_end]  # tensor([3, 4])
    patient_b_visit_offsets = visit_offsets[v_start : v_end + 1] - c_start  # tensor([0, 1, 2])


Configuration, Output Roles & Preprocessing Reference
-----------------------------------------------------

``LevelConfig`` Settings Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every entry in ``levels`` is a :class:`~pyhealth.processors.LevelConfig` dictionary
validated at construction time:

.. list-table::
   :header-rows: 1
   :widths: 15 17 11 25 32

   * - Key
     - Type
     - Default
     - Effect on Output
     - Constraints
   * - ``ordering``
     - ``"set" | "sequence" | "series"``
     - ``"sequence"``
     - ``"series"`` emits ``(l, "time")`` (unless resampled); reported via ``ordering()``.
     - ``"series"`` requires finite, non-decreasing ``time`` in ``Records``; ``"set"`` and ``"sequence"`` omit ``time``. Level ``0`` must set ``tokens``, ``values``, or ``ordering="series"``.
   * - ``tokens``
     - ``int`` (``K >= 1``)
     - omitted
     - Emits ``(l, "tokens")`` (``[N_l]`` if ``K == 1``, ``[N_l, K]`` if ``K > 1``); ``vocab_size()[l]`` is a ``K``-tuple.
     - ``Records["tokens"]`` must hold ``str`` when ``K == 1`` and ``K``-tuples of ``str`` when ``K > 1`` (``None`` tokens raise).
   * - ``values``
     - ``Sequence[int]`` (``S_l``)
     - omitted
     - Emits ``(l, "values")`` of shape ``[N_l, *S_l]``; ``value_dim()[l]`` returns ``tuple(S_l)``.
     - Must be a non-empty list of positive integers (use ``[1]`` for scalars; flat ``[N_l]`` input lists are reshaped to ``[N_l, 1]``).
   * - ``spatial``
     - ``Sequence[bool] | None``
     - ``None`` (all ``False``)
     - Reported by ``spatial()[l]`` to distinguish channel axes (``False``) from spatial/waveform axes (``True``).
     - Requires ``values``; ``len(spatial)`` must equal ``len(values)``.
   * - ``code_mapping``
     - ``(src, dst)`` or ``Sequence[(src, dst) | None]``
     - ``None``
     - Maps codes via :class:`~pyhealth.medcode.CrossMap` before vocabulary indexing; 1-to-many targets expand records.
     - Requires ``tokens`` (1 pair, or ``K`` entries). One-to-many mappings require level ``0``, ``ordering="set"``, ``tokens=1``, and no ``values``.
   * - ``resample``
     - :class:`~pyhealth.processors.Resample` ``{"step", "method"}``
     - ``None``
     - Bins ``t >= 0`` into steps :math:`\lfloor t / \text{step} \rfloor`, drops ``(l, "time")``, and sets ``ordering()[l] == "sequence"``.
     - Requires ``ordering="series"`` and ``step > 0``. ``"min" | "max" | "avg"`` require level ``0`` with ``values`` only; ``"merge"`` requires ``children`` only with child ``ordering="set"``; resampled level ``0`` needs ``tokens`` or ``values``.
   * - ``missing_indicator``
     - ``bool``
     - ``False``
     - Emits ``bool`` ``(l, "mask")`` of shape ``[N_l, *S_l]`` right after ``values`` (``True`` where ``NaN`` before imputation).
     - Requires both ``values`` and ``impute``.
   * - ``impute``
     - :class:`~pyhealth.processors.Impute` or ``Sequence[Impute]``
     - ``None``
     - Fills ``NaN`` values (and resampled ``<pad>`` tokens for ``forward_fill`` / ``backward_fill`` without ``by_tokens``) in place.
     - ``"mean" | "min" | "max" | "linear"`` require ``values``. ``"forward_fill" | "backward_fill" | "linear"`` require ``"sequence"`` or ``"series"``. ``by_tokens`` must be non-empty indices ``< K``; ``by_dims`` distinct axes ``< len(S_l)``.
   * - ``normalize``
     - :class:`~pyhealth.processors.Normalize`
     - ``None``
     - Scales ``values`` in place via ``"zscore"`` or ``"minmax"`` (default ``by_dims=()``, per value position).
     - Requires ``values``; ``by_tokens`` and ``by_dims`` follow the same index bounds as ``impute``.

Output Roles & Positional Binding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schema metadata lives once on the processor and is never repeated inside cached
samples. :meth:`~pyhealth.processors.SequenceProcessorV2.process` returns a
``tuple[torch.Tensor, ...]`` ordered by **increasing level** (``0 .. L``), then
by **canonical role order**:

.. list-table::
   :header-rows: 1
   :widths: 14 12 26 48

   * - Role
     - ``dtype``
     - Shape
     - When Present
   * - ``"tokens"``
     - ``int64``
     - ``[N_l]`` if ``K == 1``, else ``[N_l, K]``
     - ``tokens: K`` is configured at level ``l``.
   * - ``"values"``
     - ``float32``
     - ``[N_l, *S_l]`` (scalars are ``[N_l, 1]``)
     - ``values: S_l`` is configured at level ``l``.
   * - ``"mask"``
     - ``bool``
     - ``[N_l, *S_l]``
     - ``missing_indicator: True`` (``True`` where ``NaN`` after resampling and before imputation).
   * - ``"time"``
     - ``float32``
     - ``[N_l]``
     - ``ordering == "series"`` and ``resample`` is ``None``.
   * - ``"offsets"``
     - ``int64``
     - ``[N_{l+1} + 1]`` (or ``[B + 1]`` at ``L``)
     - Every inner level ``l < L``; ``(L, "offsets")`` is appended in collated batches.

Bind tensors by ``(level, role)`` key when needed:

.. code-block:: python

    fields = dict(zip(visit_proc.schema(), p0))
    tokens_0 = fields[(0, "tokens")]

    batch_fields = dict(zip(visit_proc.batch_schema(), visit_proc.collate([p0, p1])))
    outer_batch_offsets = batch_fields[(visit_proc.depth, "offsets")]

Because ``len(proc.schema())`` is fixed by configuration,
:class:`~pyhealth.processors.SequenceProcessorV2` sets ``stores_tensor = True``
so :class:`~pyhealth.datasets.SampleDataset` writes every tensor natively via
``litdata`` instead of pickling per-sample objects. Undeclared extra keys in
:class:`~pyhealth.processors.Records` are ignored.

Preprocessing Order
~~~~~~~~~~~~~~~~~~~

Within each level, :meth:`~pyhealth.processors.SequenceProcessorV2.process`
runs configured transformations in fixed order:

1. ``code_mapping`` (before vocabulary lookup; unmapped codes are kept as-is).
2. ``resample`` (``min``, ``max``, and ``avg`` skip ``NaN`` readings; empty bins get ``<pad>`` tokens and ``NaN`` values).
3. ``missing_indicator`` (snapshots ``np.isnan(values)`` into ``mask`` before imputation).
4. ``impute`` (runs single step or chain in order; temporal fills never cross parent group boundaries).
5. ``normalize`` (scales imputed values; tokens lacking training statistics fall back per value position to the pooled statistic across all tokens).
6. **Finite check:** Any ``NaN`` or ``inf`` remaining in ``values`` raises ``ValueError``.

.. note::

   Dataset statistics for ``impute`` (``mean``, ``min``, ``max``) and
   ``normalize`` (``zscore``, ``minmax``) are streamed in one pass over **raw**
   training values (before resampling) during ``fit()``. When
   ``proc.learns_statistics`` is ``True``, pass
   ``split=PatientSplit(...)`` to ``dataset.set_task()`` so validation and test
   patients do not leak into preprocessing statistics.


Mapping Common Clinical Data Structures
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 31 22 22

   * - Clinical Data Structure
     - ``levels`` Configuration
     - Output ``schema()``
     - Ordering & Notes
   * - **Ordered flat code sequence** (chronological diagnoses or procedures)
     - ``[{"ordering": "sequence", "tokens": 1}]``
     - ``((0, "tokens"),)``
     - ``{0: "sequence"}``; 1D token IDs ``[N_0]``.
   * - **Unordered patient code set** (bag-of-codes)
     - ``[{"ordering": "set", "tokens": 1}]``
     - ``((0, "tokens"),)``
     - ``{0: "set"}``; pool via ``EmbeddingBag`` or scatter to multi-hot.
   * - **Visits of medical codes** (nested visits / per-visit bag-of-codes)
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]``
     - ``((0, "tokens"), (0, "offsets"))``
     - ``{0: "set", 1: "sequence"}``; ``(0, "offsets")`` slices codes per visit.
   * - **Admissions -> Visits -> Codes** (3-level longitudinal history)
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}, {"ordering": "sequence"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "offsets"))``
     - ``(0, "offsets")`` groups codes into visits; ``(1, "offsets")`` groups visits into admissions.
   * - **Timestamped visits of codes** (irregular encounter intervals)
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "series"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "time"))``
     - ``{0: "set", 1: "series"}``; visit timestamps at level ``1``.
   * - **Nested numeric measurements per visit**
     - ``[{"ordering": "sequence", "values": [F]}, {"ordering": "sequence"}]``
     - ``((0, "values"), (0, "offsets"))``
     - Packed ``float32`` ``[N_0, F]`` sliced per visit via ``(0, "offsets")``.
   * - **Timestamped visits of numeric vectors**
     - ``[{"ordering": "sequence", "values": [F]}, {"ordering": "series"}]``
     - ``((0, "values"), (0, "offsets"), (1, "time"))``
     - Level-``0`` numeric vectors paired with visit-level ``(1, "time")``.
   * - **Regularly gridded multichannel time series**
     - ``[{"ordering": "sequence", "values": [F]}]`` (or ``"series"`` + ``resample``)
     - ``((0, "values"),)`` (+ ``(0, "mask")``)
     - ``{0: "sequence"}``; uniform grid steps ``[N_0, F]``.
   * - **Irregular timestamped multichannel time series**
     - ``[{"ordering": "series", "values": [F]}]``
     - ``((0, "values"), (0, "time"))``
     - ``{0: "series"}``; aligned ``values`` ``[N_0, F]`` and ``time`` ``[N_0]``.
   * - **Long-format coded quantitative events** (e.g., ``(test, unit)`` + value + time)
     - ``[{"ordering": "series", "tokens": K, "values": [1]}]``
     - ``((0, "tokens"), (0, "values"), (0, "time"))``
     - ``{0: "series"}``; co-indexed ontology tuples, values, and timestamps.
   * - **Single pre-tokenized clinical note**
     - ``[{"ordering": "sequence", "tokens": 1}]``
     - ``((0, "tokens"),)``
     - ``{0: "sequence"}``; text arrives pre-split into tokens.
   * - **Timestamped sequence of clinical notes**
     - ``[{"ordering": "sequence", "tokens": 1}, {"ordering": "series"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "time"))``
     - ``{0: "sequence", 1: "series"}``; ``(0, "offsets")`` slices tokens per note.
   * - **Waveforms & serial medical images** (ECG strips, Chest X-ray series)
     - ``[{"ordering": "series", "values": [C, H, W], "spatial": [False, True, True]}]``
     - ``((0, "values"), (0, "time"))``
     - Pre-loaded arrays with ``spatial`` marking channel vs. continuous axes.

Constructing ``Records`` in a Task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside :meth:`BaseTask.__call__ <pyhealth.tasks.BaseTask.__call__>`, convert
standard Python collections and EHR query rows into
:class:`~pyhealth.processors.Records` dictionaries as follows:

.. code-block:: python

    from pyhealth.processors import Records

    # 1. Flat list of codes: list[str]
    rec_flat: Records = {"tokens": ["428.0", "401.9", "250.00"]}

    # 2. Nested list of visits: list[list[str]]
    visits = [["428.0", "401.9"], [], ["250.00"]]
    rec_visits: Records = {"children": [{"tokens": v} for v in visits]}

    # 3. 3-level admissions -> visits -> codes: list[list[list[str]]]
    admissions = [[["486", "786.05"], ["518.81"]], [["428.0"]]]
    rec_admissions: Records = {
        "children": [
            {"children": [{"tokens": v} for v in adm]}
            for adm in admissions
        ]
    }

    # 4. Timestamped visits of codes: list[tuple[float, list[str]]]
    timed_visits = [(0.0, ["428.0", "401.9"]), (30.0, ["584.9"])]
    rec_timed_visits: Records = {
        "time": [t for t, _ in timed_visits],
        "children": [{"tokens": codes} for _, codes in timed_visits],
    }

    # 5. Irregular multichannel time series: (timestamps, values_2d)
    rec_vitals: Records = {
        "time": [0.0, 1.5, 4.0],
        "values": [[80.0, 75.0], [84.0, 78.0], [78.0, 74.0]],
    }

    # 6. Timestamped pre-tokenized clinical notes: list[tuple[float, list[str]]]
    timed_notes = [(0.0, ["admit", "dyspnea"]), (12.0, ["stable", "on", "room", "air"])]
    rec_notes: Records = {
        "time": [t for t, _ in timed_notes],
        "children": [{"tokens": words} for _, words in timed_notes],
    }

    # 7. Long-format lab rows: list[tuple[float, str, str, float]]
    lab_rows = [(0.0, "Glucose", "mg/dL", 108.0), (2.5, "Potassium", "mEq/L", 4.1)]
    rec_labs: Records = {
        "time": [t for t, _, _, _ in lab_rows],
        "tokens": [(test, unit) for _, test, unit, _ in lab_rows],
        "values": [[val] for _, _, _, val in lab_rows],
    }


Clinical Recipes & Subclass Presets
-----------------------------------

1. 3-Level Admissions History with ICD-to-CCS Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple hospital admissions (level ``2``), encounters within each admission
(level ``1``), and unordered ICD-9-CM diagnosis codes mapped to Clinical
Classifications Software (``CCSCM``) categories (level ``0``):

.. code-block:: python

    from pyhealth.processors import Records, SequenceProcessorV2

    adm_proc = SequenceProcessorV2(
        levels=[
            {"ordering": "set", "tokens": 1, "code_mapping": ("ICD9CM", "CCSCM")},
            {"ordering": "sequence"},  # Encounters within an admission
            {"ordering": "sequence"},  # Hospital admissions
        ]
    )

    patient_admissions: Records = {
        "children": [
            {  # Admission 0: ED intake (pneumonia, dyspnea) -> ICU transfer (resp. failure, severe sepsis)
                "children": [
                    {"tokens": ["486", "786.05"]},
                    {"tokens": ["518.81", "995.92"]},
                ]
            },
            {  # Admission 1: Routine check (no new diagnoses) -> Cardiology consult (CHF, hypertension)
                "children": [
                    {"tokens": []},
                    {"tokens": ["428.0", "401.9"]},
                ]
            },
        ]
    }

    adm_proc.fit([{"admissions": patient_admissions}], "admissions")
    adm_proc.vocab(0)
    # {'<pad>': 0, '<unk>': 1, '122': 2, '133': 3, '131': 4, '2': 5, '108': 6, '98': 7}

    tokens, enc_offsets, adm_offsets = adm_proc.process(patient_admissions)
    # tokens:      tensor([2, 3, 4, 5, 6, 7])
    # enc_offsets: tensor([0, 2, 4, 4, 6])  — 4 encounters (encounter 2 is empty: [4:4])
    # adm_offsets: tensor([0, 2, 4])        — 2 admissions (encounters [0:2] and [2:4])

2. Multi-Namespace Quantitative Labs ``(Analyte, Unit)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long-format serum labs with ``tokens: 2`` (independent vocabularies for analyte
and measurement unit), per-analyte imputation (``forward_fill`` -> ``mean`` ->
``zero``), ``missing_indicator: True``, and per-``(analyte, unit)`` z-score
normalization:

.. code-block:: python

    from pyhealth.processors import Records, SequenceProcessorV2

    lab_proc = SequenceProcessorV2(
        levels=[
            {
                "ordering": "series",
                "tokens": 2,
                "values": [1],
                "impute": [
                    {"method": "forward_fill", "by_tokens": [0]},
                    {"method": "mean", "by_tokens": [0]},
                    {"method": "zero"},
                ],
                "missing_indicator": True,
                "normalize": {"method": "zscore", "by_tokens": [0, 1]},
            }
        ]
    )

    # Training stats: Glucose [90, 110] -> mean=100, std=10; Potassium [3.8, 4.2] -> mean=4.0, std=0.2
    train_labs = [
        {
            "labs": {
                "time": [0.0, 1.0, 2.0, 3.0],
                "tokens": [
                    ("Glucose", "mg/dL"),
                    ("Potassium", "mEq/L"),
                    ("Glucose", "mg/dL"),
                    ("Potassium", "mEq/L"),
                ],
                "values": [[90.0], [3.8], [110.0], [4.2]],
            }
        }
    ]
    lab_proc.fit(train_labs, "labs")
    lab_proc.vocab_size()
    # {0: (4, 4)}  — namespace 0 (analyte) and namespace 1 (unit)

    nan = float("nan")
    sample_labs: Records = {
        "time": [0.5, 3.0, 5.5],
        "tokens": [
            ("Glucose", "mg/dL"),
            ("Glucose", "mg/dL"),      # Hemolyzed / missing -> forward-filled to 105.0
            ("Potassium", "mEq/L"),
        ],
        "values": [[105.0], [nan], [4.3]],
    }
    l_tokens, l_values, l_mask, l_time = lab_proc.process(sample_labs)
    # l_tokens: tensor([[2, 2],
    #                   [2, 2],
    #                   [3, 3]])
    # l_values: tensor([[0.5000],      # (105.0 - 100.0) / 10.0
    #                   [0.5000],      # carried forward from t=0.5, then scaled
    #                   [1.5000]])     # (4.3 - 4.0) / 0.2
    # l_mask:   tensor([[False],
    #                   [ True],
    #                   [False]])
    # l_time:   tensor([0.5000, 3.0000, 5.5000])

3. Irregular Multichannel ICU Vitals Resampled Hourly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bedside Heart Rate (bpm) and Mean Arterial Pressure (mmHg) binned into 1-hour
averages (``resample={"step": 1.0, "method": "avg"}``), with unobserved hours
flagged in ``mask``, forward-filled, and z-score normalized per channel:

.. code-block:: python

    from pyhealth.processors import SequenceProcessorV2

    vitals_proc = SequenceProcessorV2(
        levels=[
            {
                "ordering": "series",
                "values": [2],
                "resample": {"step": 1.0, "method": "avg"},
                "impute": [
                    {"method": "forward_fill"},
                    {"method": "mean", "by_dims": ()},
                    {"method": "zero"},
                ],
                "missing_indicator": True,
                "normalize": {"method": "zscore"},
            }
        ]
    )

    # Raw training stats: HR [80, 100] -> mean=90, std=10; MAP [70, 80] -> mean=75, std=5
    vitals_proc.fit(
        [{"vitals": {"time": [0.0, 1.0], "values": [[80.0, 70.0], [100.0, 80.0]]}}],
        "vitals",
    )
    vitals_proc.schema()    # ((0, 'values'), (0, 'mask')) — time is replaced by 1-hour steps
    vitals_proc.ordering()  # {0: 'sequence'}

    # Hour 0 [0, 1): avg([82, 70], [88, 75]) = [85.0, 72.5]; Hour 1 [1, 2): empty; Hour 2 [2, 3): [100.0, 80.0]
    v_values, v_mask = vitals_proc.process(
        {
            "time": [0.2, 0.8, 2.4],
            "values": [[82.0, 70.0], [88.0, 75.0], [100.0, 80.0]],
        }
    )
    # v_values: tensor([[-0.5000, -0.5000],
    #                   [-0.5000, -0.5000],
    #                   [ 1.0000,  1.0000]])
    # v_mask:   tensor([[False, False],
    #                   [ True,  True],
    #                   [False, False]])

4. Timestamped Clinical Notes & ``spatial`` Signal/Image Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-tokenized clinical notes over an ICU stay combine a level-``0`` token
``"sequence"`` with a level-``1`` timestamped ``"series"``:

.. code-block:: python

    import numpy as np
    from pyhealth.processors import Records, SequenceProcessorV2

    notes_proc = SequenceProcessorV2(
        levels=[
            {"ordering": "sequence", "tokens": 1},  # Level 0: word tokens in a note
            {"ordering": "series"},                 # Level 1: timestamped notes
        ]
    )
    icu_notes: Records = {
        "time": [0.0, 8.5],
        "children": [
            {"tokens": ["acute", "dyspnea"]},
            {"tokens": ["room", "air", "stable"]},
        ],
    }
    notes_proc.fit([{"notes": icu_notes}], "notes")
    note_tokens, note_offsets, note_time = notes_proc.process(icu_notes)
    # note_tokens:  tensor([2, 3, 4, 5, 6])
    # note_offsets: tensor([0, 2, 5])
    # note_time:    tensor([0.0000, 8.5000])

    # 12-lead ECG windows (12 leads x 1000 samples at 500 Hz)
    ecg_proc = SequenceProcessorV2(
        levels=[{"ordering": "series", "values": [12, 1000], "spatial": [False, True]}]
    )
    ecg_proc.fit([], "ecg")
    ecg_values, ecg_time = ecg_proc.process(
        {"time": [0.0, 6.0], "values": np.zeros((2, 12, 1000), dtype=np.float32)}
    )
    # ecg_values.shape == torch.Size([2, 12, 1000]), ecg_proc.spatial() == {0: (False, True)}

Common ``values`` and ``spatial`` combinations across clinical modalities:

.. list-table::
   :header-rows: 1
   :widths: 30 16 20 34

   * - Clinical Modality
     - ``values``
     - ``spatial``
     - Downstream Layer Semantics
   * - Scalar lab reading (e.g., Potassium)
     - ``[1]``
     - ``[False]``
     - 1D scalar projection (``nn.Linear(1, D)``).
   * - 8-channel ICU vital sign panel
     - ``[8]``
     - ``[False]``
     - Unordered heterogeneous features (``nn.Linear(8, D)``).
   * - 12-lead ECG strip (2 s at 500 Hz)
     - ``[12, 1000]``
     - ``[False, True]``
     - 12 lead channels, 1D temporal axis (``nn.Conv1d``).
   * - Serial bedside Chest X-rays
     - ``[1, 224, 224]``
     - ``[False, True, True]``
     - 1 grayscale channel, 2D pixel grid (``nn.Conv2d`` / ViT).

5. Defining Reusable Configuration Subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclasses of :class:`~pyhealth.processors.SequenceProcessorV2` are pure
configuration presets: their ``__init__`` passes a ``levels`` list to
``super().__init__(levels=...)`` without overriding ``fit()``, ``process()``,
or ``collate()``. Registering them via
:func:`~pyhealth.processors.register_processor` makes them usable by name or
``(name, kwargs)`` tuple in any task schema:

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.processors import SequenceProcessorV2, register_processor


    @register_processor("visit_codes")
    class VisitCodesProcessor(SequenceProcessorV2):
        """Unordered medical codes grouped into a sequence of clinical visits."""

        def __init__(self, code_mapping: tuple[str, str] | None = None):
            super().__init__(
                levels=[
                    {"ordering": "set", "tokens": 1, "code_mapping": code_mapping},
                    {"ordering": "sequence"},
                ]
            )


    @register_processor("timed_visit_codes")
    class TimedVisitCodesProcessor(SequenceProcessorV2):
        """Unordered medical codes grouped into timestamped clinical visits."""

        def __init__(self):
            super().__init__(
                levels=[{"ordering": "set", "tokens": 1}, {"ordering": "series"}]
            )


    @register_processor("hourly_vitals")
    class HourlyVitalsProcessor(SequenceProcessorV2):
        """Irregular multichannel vitals resampled hourly, imputed, and scaled."""

        def __init__(self, num_channels: int, step_hours: float = 1.0):
            super().__init__(
                levels=[
                    {
                        "ordering": "series",
                        "values": [num_channels],
                        "resample": {"step": step_hours, "method": "avg"},
                        "impute": [
                            {"method": "forward_fill"},
                            {"method": "mean", "by_dims": ()},
                            {"method": "zero"},
                        ],
                        "missing_indicator": True,
                        "normalize": {"method": "zscore"},
                    }
                ]
            )


    samples = [
        {
            "patient_id": "p0",
            "diagnoses": {"children": [{"tokens": ["428.0", "401.9"]}, {"tokens": ["584.9"]}]},
            "vitals": {"time": [0.2, 1.4], "values": [[82.0, 76.0], [88.0, 72.0]]},
            "readmission": 1,
        },
        {
            "patient_id": "p1",
            "diagnoses": {"children": [{"tokens": ["250.00"]}]},
            "vitals": {"time": [0.5], "values": [[74.0, 82.0]]},
            "readmission": 0,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "diagnoses": ("visit_codes", {"code_mapping": ("ICD9CM", "CCSCM")}),
            "vitals": ("hourly_vitals", {"num_channels": 2}),
        },
        output_schema={"readmission": "binary"},
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    # batch["diagnoses"] == (tensor([2, 3, 4, 5]), tensor([0, 2, 3, 4]), tensor([0, 2, 3]))


Using Packed Outputs in a Model
-------------------------------

.. note::

   Built-in adapters bridging packed records to dense padded inputs and
   built-in PyHealth models native to :class:`~pyhealth.processors.SequenceProcessorV2`
   are currently work in progress.

When implementing a custom model over a collated ``"sequence_v2"`` batch tuple
today, standard PyTorch operations consume ``(tokens, offsets)`` directly:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pad_sequence

    b_tokens, visit_offsets, batch_offsets = visit_proc.collate([p0, p1])
    vocab_len = visit_proc.vocab_size()[0][0]

    # Path A: Zero-padding PyTorch 2.x jagged NestedTensor (for SDPA / FlashAttention)
    jagged_visits = torch.nested.nested_tensor_from_jagged(b_tokens, visit_offsets)

    # Path B: Direct set pooling per visit via nn.EmbeddingBag (zero padding, handles empty visits)
    bag = nn.EmbeddingBag(
        vocab_len, 8, mode="mean", padding_idx=0, include_last_offset=True
    )
    visit_embeddings = bag(b_tokens, visit_offsets)  # shape [4, 8] (4 total visits across batch)

    # Path C: Rectangular padded tensor [total_visits, max_codes_in_batch]
    padded_visits = pad_sequence(jagged_visits.unbind(), batch_first=True, padding_value=0)
    # tensor([[2, 3],
    #         [4, 0],
    #         [3, 0],
    #         [4, 0]])

    # Path D: On-the-fly per-visit multi-hot matrix [total_visits, vocab_len]
    num_visits = visit_offsets.numel() - 1
    visit_ids = torch.repeat_interleave(torch.arange(num_visits), visit_offsets.diff())
    multihot_visits = torch.zeros(num_visits, vocab_len, dtype=torch.float32)
    multihot_visits[visit_ids, b_tokens] = 1.0
    # tensor([[0., 0., 1., 1., 0.],
    #         [0., 0., 0., 0., 1.],
    #         [0., 0., 0., 1., 0.],
    #         [0., 0., 0., 0., 1.]])


Design Rationale
----------------

- **Parameterized hierarchy without pre-padding or truncation:** Clinical EHR
  features span flat event streams (1 level), visits or notes (2 levels), and
  multi-admission trajectories (3 levels) with long-tailed group sizes.
  Representing nesting via ``levels`` and 1D CSR ``offsets`` eliminates
  ``<pad>`` storage in :class:`~pyhealth.datasets.SampleDataset` caches and
  never truncates validation or test encounters that exceed training lengths.
- **One canonical representation per clinical semantic:** Unordered code sets
  and token sequences within visits describe the same underlying clinical
  events, even when different models consume them as embeddings, pooled bags,
  padded grids, or multi-hot vectors. Storing lossless packed payloads once
  preserves repeat counts, timestamps, and paired numeric values for every
  downstream architecture.
- **Explicit semantic metadata on the processor:**
  - ``ordering()`` distinguishes unordered ``"set"`` levels (permutation-invariant
    pooling / self-attention without positional encoding), discrete
    ``"sequence"`` steps (index-based positional encoding), and timestamped
    ``"series"`` observations (continuous-time encoding).
  - Multi-namespace ``tokens: K`` binds co-occurring ontology attributes at the
    same event—such as ``(analyte, unit)`` lab results, ``(drug, route)``
    medication administrations, ``(specimen_site, organism)`` microbiology
    cultures, or ``(diagnosis_code, poa_status)`` inpatient diagnoses—with
    separate vocabularies rather than combinatorially concatenated strings.
  - ``spatial()`` distinguishes unordered feature channels (``False`` -> dense
    ``nn.Linear`` layers) from continuous waveform or image dimensions
    (``True`` -> ``nn.Conv1d`` / ``nn.Conv2d`` or patch embeddings).


API Reference
-------------

.. autoclass:: pyhealth.processors.SequenceProcessorV2
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.processors.Records
    :members:

.. autoclass:: pyhealth.processors.LevelConfig
    :members:

.. autoclass:: pyhealth.processors.Resample
    :members:

.. autoclass:: pyhealth.processors.Impute
    :members:

.. autoclass:: pyhealth.processors.Normalize
    :members:
