pyhealth.processors.SequenceProcessorV2
=======================================

:class:`~pyhealth.processors.SequenceProcessorV2` (registered as
``"sequence_v2"``) is PyHealth's unified feature processor for sequential,
hierarchical, and temporal clinical records. It represents categorical medical
codes, continuous physiological measurements, and timestamps at any depth of
nesting in a single lossless, padding-free Compressed Sparse Row (CSR / jagged)
layout.

Quick Start
-----------

A task supplies each feature as a :class:`~pyhealth.processors.Records`
dictionary of columns (``tokens``, ``values``, ``time``, and ``children``), and
configures the hierarchy from the innermost atomic level (level ``0``) outward
to level ``L``:

.. code-block:: python

    from pyhealth.processors import Records, SequenceProcessorV2

    # Level 0: unordered ICD-9 diagnosis codes within an encounter
    # Level 1: chronological sequence of encounters for a patient
    proc = SequenceProcessorV2(
        levels=[
            {"ordering": "set", "tokens": 1},
            {"ordering": "sequence"},
        ]
    )

    train_samples = [
        {
            "diagnoses": {
                "children": [
                    {"tokens": ["428.0", "401.9"]},  # Heart failure, hypertension
                    {"tokens": ["250.00"]},          # Type 2 diabetes
                ]
            }
        }
    ]
    proc.fit(train_samples, "diagnoses")
    proc.schema()
    # ((0, 'tokens'), (0, 'offsets'))

    # Encounter 1 has no new diagnoses; "584.9" (acute kidney injury) was unseen in fit -> <unk> (1)
    sample: Records = {
        "children": [
            {"tokens": ["428.0", "401.9"]},
            {"tokens": []},
            {"tokens": ["250.00", "584.9"]},
        ]
    }
    tokens, offsets = proc.process(sample)
    # tokens:  tensor([2, 3, 4, 1])      — all 4 codes packed contiguously
    # offsets: tensor([0, 2, 2, 4])      — encounter slices [0:2], [2:2] (empty), [2:4]

In a :class:`~pyhealth.tasks.BaseTask`, configurations are plain data passed
via ``("sequence_v2", {"levels": [...]})`` or encapsulated in reusable
configuration subclasses:

.. code-block:: python

    input_schema = {
        "diagnoses": (
            "sequence_v2",
            {"levels": [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]},
        ),
        "vitals": (
            "sequence_v2",
            {
                "levels": [
                    {
                        "ordering": "series",
                        "values": [4],
                        "resample": {"step": 1.0, "method": "avg"},
                        "impute": [{"method": "forward_fill"}, {"method": "zero"}],
                        "missing_indicator": True,
                        "normalize": {"method": "zscore"},
                    }
                ]
            },
        ),
    }


Why a Unified Sequence Representation?
--------------------------------------

Clinical Data Is Naturally Hierarchical and Multimodal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Electronic health record (EHR) data spans multiple levels of clinical
granularity and combines discrete medical ontologies with continuous signals:

- **1 level (flat events or signals):** A patient's chronological medication
  administrations, a set of chronic conditions, or a stream of bedside ICU
  vital signs.
- **2 levels (encounters or documents):** Diagnosis and procedure codes grouped
  into outpatient visits, quantitative lab panels grouped by clinical
  encounter, or word tokens grouped into timestamped progress notes.
- **3 levels (longitudinal hospital stays):** Medical codes grouped into
  clinical encounters, and encounters grouped into distinct hospital
  admissions across a patient's multi-year trajectory.

Rather than treating each combination of nesting depth, timestamp presence, and
payload modality as an isolated preprocessing pipeline,
:class:`~pyhealth.processors.SequenceProcessorV2` formalizes hierarchy
directly through ``levels``. Adding timestamps to clinical visits or
introducing an outer hospital-admission grouping tier is a single declarative
change in ``levels`` rather than a rewrite of the dataset schema.

One Clinical Semantic, One Canonical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different model architectures often consume the exact same clinical data
through different tensor views. Consider a patient's longitudinal visit
history, where each visit records an unordered set of diagnosis codes:

- Sequence and attention architectures often embed individual token indices,
  either in native jagged layout or padded to ``[num_visits, max_codes]``.
- Bag-of-codes and generative architectures often represent each visit as a
  binary multi-hot vector of shape ``[num_visits, vocab_size]``.

Those two views describe the **exact same clinical phenomenon**—a sequence of
visits, each containing an unordered set of codes. Materializing model-specific
views during preprocessing fragments datasets into incompatible cached formats
and causes **premature information loss**: collapsing codes into a binary
multi-hot matrix during preprocessing irreversibly discards repeat code
occurrences, intra-group ordering, and code-level quantitative attributes (such
as medication dosages or lab values), preventing embedding-based models from
using the same processed dataset.

:class:`~pyhealth.processors.SequenceProcessorV2` enforces the principle that
**identical clinical semantics always yield the same canonical, lossless
representation**. Raw events, values, timestamps, and group boundaries are
stored once; dense padded matrices, jagged nested tensors, or multi-hot
projections are derived on the model side.

Zero Padding Waste and Zero Silent Truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clinical encounter sizes follow heavy-tailed distributions: routine outpatient
visits typically record 2–5 diagnosis codes, whereas complex critical-care
stays can record 80 or more. Enforcing fixed rectangular tensor shapes during
preprocessing introduces two severe flaws:

1. **Silent test-time truncation:** Fixing ``max_codes_per_visit`` from the
   training split silently truncates validation or test encounters whenever a
   patient presents with more codes than the longest training visit.
2. **70–90% memory and cache waste:** Padding every routine visit up to the
   dataset maximum fills disk caches, RAM, and host-to-device transfers
   predominantly with ``<pad>`` tokens (ID ``0``).

By packing records contiguously and tracking group boundaries via 1D CSR index
``offsets``, :class:`~pyhealth.processors.SequenceProcessorV2` achieves **zero
padding and zero truncation** across preprocessing, native binary tensor disk
caching in :class:`~pyhealth.datasets.SampleDataset`, and batch collation in
:func:`~pyhealth.datasets.get_dataloader`.


Format Design Decisions
-----------------------

Hierarchical ``levels`` and CSR ``offsets``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data within a sample is organized into levels indexed from ``l = 0``
(innermost atomic records, such as individual codes, lab measurements, vital
sign readings, or text tokens) to ``l = L`` (outermost grouping tier within a
sample, such as visits or hospital admissions). The ``levels`` sequence is
configured **innermost-first**: ``[level_0, level_1, ..., level_L]``.

Whenever ``L > 0``, boundaries partitioning the ``N_l`` records at level ``l``
into the ``N_{l+1}`` parent groups at level ``l + 1`` are stored as a 1D
``int64`` ``offsets`` tensor of shape ``[N_{l+1} + 1]`` obeying standard CSR
invariants:

- **Contiguous slicing:** Group ``j`` at level ``l + 1`` owns the slice
  ``records_l[offsets_l[j] : offsets_l[j + 1]]``.
- **Anchored endpoints:** ``offsets_l[0] == 0`` and ``offsets_l[-1] == N_l``.
- **Monotonicity and empty groups:** ``offsets_l[j] <= offsets_l[j + 1]``.
  Equal adjacent offsets (``offsets_l[j] == offsets_l[j + 1]``) represent an
  empty group—such as a clinical follow-up visit with zero new diagnosis
  codes—without inserting dummy placeholder tokens.
- **Batch boundaries:** Within a single sample, the outermost level ``L`` has
  no internal offsets. When batching ``B`` samples, :meth:`~pyhealth.processors.SequenceProcessorV2.collate`
  concatenates all levels without padding, shifts each sample's internal
  offsets by the cumulative record counts of preceding samples, and appends
  outer batch offsets ``(L, "offsets")`` of shape ``[B + 1]``.

Columnar Input (``Records``) and Positional Tensor Tuple Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Declared Columnar Input.** A task provides each feature as a
:class:`~pyhealth.processors.Records` dictionary holding columnar sequences at
one level (``tokens``, ``values``, ``time``, and ``children`` holding sub-records
one level down). Active roles are **declared in**
:class:`~pyhealth.processors.LevelConfig` **rather than inferred from data**:

- Every sample—including empty patient histories (``N_l = 0``)—produces the
  exact same number and dtypes of tensors.
- Missing declared keys or mismatched column lengths raise ``ValueError``
  immediately during :meth:`~pyhealth.processors.SequenceProcessorV2.process`.
- Undeclared extra keys in :class:`~pyhealth.processors.Records` are ignored,
  making it inexpensive to iterate on ``input_schema`` without rewriting task
  extraction logic.
- Timestamps are supplied as plain numbers in the task's chosen unit and
  origin. Relative time (e.g., hours since ICU admission or days since index
  diagnosis) is recommended because ``float32`` cannot represent absolute Unix
  epoch seconds at sub-minute precision.

**Positional Tensor Tuple and Processor Metadata.** Schema metadata lives once
on the processor (:meth:`~pyhealth.processors.SequenceProcessorV2.schema`,
:meth:`~pyhealth.processors.SequenceProcessorV2.batch_schema`,
:meth:`~pyhealth.processors.SequenceProcessorV2.ordering`,
:meth:`~pyhealth.processors.SequenceProcessorV2.vocab_size`,
:meth:`~pyhealth.processors.SequenceProcessorV2.value_dim`,
:meth:`~pyhealth.processors.SequenceProcessorV2.spatial`) and is never
duplicated inside cached samples.
:meth:`~pyhealth.processors.SequenceProcessorV2.process` returns a plain
``tuple[torch.Tensor, ...]`` ordered by **increasing level** (``0 .. L``), then
by **canonical role order**:

.. list-table::
   :header-rows: 1
   :widths: 14 12 24 50

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
     - ``missing_indicator: True`` (``True`` where ``NaN`` before imputation).
   * - ``"time"``
     - ``float32``
     - ``[N_l]``
     - ``ordering == "series"`` and ``resample`` is ``None``.
   * - ``"offsets"``
     - ``int64``
     - ``[N_{l+1} + 1]`` (or ``[B + 1]`` at ``L``)
     - Every inner level ``l < L``; ``(L, "offsets")`` is appended in batches.

Consumers can unpack the tuple positionally or bind tensors to ``(level, role)``
keys via :meth:`~pyhealth.processors.SequenceProcessorV2.schema` (single
sample) or :meth:`~pyhealth.processors.SequenceProcessorV2.batch_schema`
(collated batch):

.. code-block:: python

    fields = dict(zip(proc.schema(), proc.process(sample)))
    tokens_0 = fields[(0, "tokens")]
    offsets_0 = fields[(0, "offsets")]

    batched = proc.collate([proc.process(sample)])
    batch_fields = dict(zip(proc.batch_schema(), batched))
    batch_offsets = batch_fields[(proc.depth, "offsets")]  # shape [B + 1]

Because every sample emits a tuple of fixed length ``len(proc.schema())``,
:class:`~pyhealth.processors.SequenceProcessorV2` declares
``stores_tensor = True``: :class:`~pyhealth.datasets.SampleDataset` serializes
all tensors natively through ``litdata`` without falling back to per-sample
Python ``pickle``, while allowing every tensor's leading record count ``N_l``
to vary freely across patients.

Record ``ordering``: ``"set"`` vs. ``"sequence"`` vs. ``"series"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each level declares an ``ordering`` that captures the clinical ordering
semantics of its records and informs downstream models how to encode position:

- **``"set"`` (unordered collection; no ``time`` tensor):** Record order carries
  no clinical meaning—for example, ICD diagnosis codes or CPT procedure codes
  assigned on a single hospital discharge summary (``{"428.0", "401.9"}`` is
  clinically identical to ``{"401.9", "428.0"}``). Downstream models process
  ``"set"`` levels with permutation-invariant pooling or self-attention
  **without positional encodings**.
- **``"sequence"`` (discrete step order; no ``time`` tensor):** Discrete
  step-by-step progression where order matters, either because inter-step
  elapsed time is unmeasured (e.g., visit 1 -> visit 2 -> visit 3, or subword
  tokens in a clinical note) or because a ``"series"`` level was placed onto a
  regular time grid via ``resample``. Downstream models apply **discrete
  index-based positional encodings** (indices ``0, 1, ..., n - 1``, such as
  sinusoidal encodings, learned position embeddings, or RoPE).
- **``"series"`` (timestamped observations; ``time`` tensor required):**
  Continuous or irregular time series where each record occurs at an explicit,
  finite, non-decreasing timestamp (e.g., bedside ICU vital signs or serum lab
  draws at ``t = 0.0, 1.5, 4.2`` hours). Downstream models apply
  **continuous-time positional encodings** driven directly by ``time`` (such as
  continuous sinusoidal embeddings, Time2Vec, or inter-event time-decay
  :math:`\Delta t`).

Categorical ``tokens`` and Multi-Namespace Ontologies (``K >= 1``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring ``tokens: K`` declares ``K`` distinct categorical token namespaces
at that level. During :meth:`~pyhealth.processors.SequenceProcessorV2.fit`,
each namespace ``k in {0, ..., K - 1}`` builds an independent vocabulary
reserving ``<pad> = 0`` and ``<unk> = 1`` (any code unseen during training maps
to ``<unk>`` at evaluation time, with no frequency cutoff).

In clinical data, a single observation is frequently characterized by a tuple
of categorical attributes from different medical ontologies that must remain
bound to the same event. Concatenating attributes into a composite string (such
as ``"Glucose_mg/dL"``) causes combinatorial vocabulary explosion and prevents
models from learning shared representations across common tests, drugs, or
units. Setting ``tokens: K`` keeps independent vocabularies
(:meth:`~pyhealth.processors.SequenceProcessorV2.vocab_size` returns
``{level: (V_0, ..., V_{K-1})}``) so models can embed each coordinate
separately:

1. **Quantitative Lab Measurements ``(test_code, unit)`` (``tokens: 2``):**
   ``("Glucose", "mg/dL")`` vs. ``("Glucose", "mmol/L")``. Multi-center EHRs
   often record the same analyte in different measurement units; separating
   ``unit`` lets models calibrate the paired numeric ``values`` without
   duplicating the lab test ontology.
2. **Medication Administrations ``(drug_code, route)`` (``tokens: 2``):**
   ``("Insulin", "IV")`` vs. ``("Insulin", "SC")``. Administration route
   fundamentally alters pharmacokinetics (onset, peak, duration) and clinical
   acuity.
3. **Microbiology Cultures ``(specimen_site, organism)`` (``tokens: 2``):**
   ``("Blood", "Staphylococcus aureus")`` vs.
   ``("Sputum", "Pseudomonas aeruginosa")``. Isolating bacteria from sterile
   blood versus non-sterile sputum carries distinct diagnostic implications for
   bacteremia, sepsis, and antibiotic selection.
4. **Inpatient Diagnoses with Admission Status ``(diagnosis_code, poa_status)``
   (``tokens: 2``):** ``("J18.9", "POA_Y")`` vs. ``("J18.9", "POA_N")``.
   "Present On Admission" (POA) flags separate pre-existing community-acquired
   conditions from hospital-acquired complications.

Continuous ``values`` and ``spatial`` Axis Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring ``values: S`` (where ``S = [S_0, S_1, ...]``) declares a continuous
``float32`` payload of shape ``[N_l, *S]`` at level ``l``. Scalar measurements
always use ``values: [1]`` and emit shape ``[N_l, 1]`` (a flat 1D input list of
length ``N_l`` is also accepted for convenience).

Because multi-dimensional clinical measurements mix unordered feature channels
with ordered continuous axes, ``spatial`` provides one boolean flag per axis of
``S`` (defaulting to ``(False,) * len(S)``):

- **``False`` (unordered feature / channel axis):** Permuting channel order
  does not alter the physical signal; downstream models apply dense projections
  (``nn.Linear``).
- **``True`` (ordered spatial / high-frequency temporal axis):** Adjacent
  elements share geometric or temporal neighborhood continuity; downstream
  models apply convolutions (``nn.Conv1d``, ``nn.Conv2d``) or spatial patch
  embeddings.

.. list-table::
   :header-rows: 1
   :widths: 24 16 20 40

   * - Clinical Modality
     - ``values`` Shape
     - ``spatial`` Flags
     - Axis Semantics
   * - Scalar lab measurement (e.g., Serum Potassium, 4.2 mEq/L)
     - ``[1]``
     - ``[False]``
     - Single quantitative scalar attached to each lab observation.
   * - Multichannel ICU vital panel (HR, SBP, DBP, MAP, RR, SpO2, Temp, GCS)
     - ``[8]``
     - ``[False]``
     - 8 heterogeneous physiological variables; cross-channel interactions use ``nn.Linear`` rather than 1D convolutions across feature indices.
   * - 12-lead ECG waveform strip (2 s at 500 Hz across leads I–III, aVR/L/F, V1–V6)
     - ``[12, 1000]``
     - ``[False, True]``
     - Axis 0 (``12``, ``False``) indexes distinct anatomical leads; axis 1 (``1000``, ``True``) is a continuous waveform axis for ``nn.Conv1d`` filters.
   * - Serial bedside Chest X-rays over consecutive ICU days
     - ``[1, 224, 224]``
     - ``[False, True, True]``
     - Axis 0 (``1``, ``False``) is grayscale channel; axes 1–2 (``224 x 224``, ``True, True``) form a 2D pixel grid for ``nn.Conv2d`` or ViT patches.

Built-In Clinical Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each level supports optional, declarative clinical preprocessing steps executed
in a strict, reproducible order during
:meth:`~pyhealth.processors.SequenceProcessorV2.process`:

1. **Ontology translation (``code_mapping``):** Maps token namespaces through
   :class:`~pyhealth.medcode.CrossMap` (e.g., ``("ICD9CM", "CCSCM")``) before
   vocabulary lookup. Unmapped codes are preserved as-is. One-to-many ontology
   mappings expand one input code into multiple records and are restricted to
   bottom-level ``"set"`` configurations with ``tokens: 1`` and no ``values``.
2. **Temporal gridding (``resample``):** Bins a ``"series"`` level onto uniform
   grid steps :math:`\lfloor t / \text{step} \rfloor` starting at :math:`t = 0`
   using ``"first"``, ``"last"``, ``"min"``, ``"max"``, ``"avg"`` (skipping
   ``NaN`` readings within a bin), or ``"merge"`` (taking the union of child
   ``"set"`` records). Unobserved bins receive ``<pad>`` tokens and ``NaN``
   values for downstream imputation. Output
   :meth:`~pyhealth.processors.SequenceProcessorV2.ordering` becomes
   ``"sequence"`` and ``time`` is dropped.
3. **Informative missingness tracking (``missing_indicator``):** When ``True``,
   records a ``bool`` ``mask`` tensor (``[N_l, *S_l]``) immediately after
   resampling and before imputation, marking every position that was originally
   ``NaN`` or belonged to an unobserved resampled step.
4. **Chained missing-value imputation (``impute``):** Applies one
   :class:`~pyhealth.processors.Impute` step or a list of steps in sequence
   (``"zero"``, ``"mean"``, ``"min"``, ``"max"``, ``"forward_fill"``,
   ``"backward_fill"``, ``"linear"``), optionally scoped per token key
   (``by_tokens``) or reduced across selected value axes (``by_dims``).
   Temporal fills operate strictly within each parent record's group and never
   leak across visits or admissions.
5. **Feature scaling (``normalize``):** Applies :class:`~pyhealth.processors.Normalize`
   (``"zscore"`` or ``"minmax"``) after imputation, scoped by ``by_tokens`` and
   ``by_dims`` (default ``by_dims=()``, one scale per channel/value position).
   Where a token lacks training statistics (``<unk>``, ``<pad>``, or all-``NaN``
   in training), each value position falls back to the pooled population
   statistic rather than collapsing the measurement to zero. Any ``NaN`` or
   ``inf`` remaining after the pipeline raises ``ValueError``.

.. note::

   Dataset statistics for ``impute`` (``mean``, ``min``, ``max``) and
   ``normalize`` (``zscore``, ``minmax``) are learned in a single streaming
   pass over **raw** training values (before resampling) during
   :meth:`~pyhealth.processors.SequenceProcessorV2.fit`. Whenever statistics
   are enabled (``proc.learns_statistics == True``), fit on the training split
   via ``dataset.set_task(task, split=PatientSplit(...))`` so validation and
   test patients do not influence preprocessing statistics.


Clinical Use Cases
------------------

1. Multi-Admission Longitudinal History with ICD-to-CCS Mapping (3 Levels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A patient's longitudinal hospital trajectory spanning multiple admissions
(level ``2``), clinical encounters within each admission (level ``1``), and
unordered ICD-9-CM diagnosis codes mapped automatically to Clinical
Classifications Software (``CCSCM``) categories (level ``0``):

.. code-block:: python

    from pyhealth.processors import Records, SequenceProcessorV2

    adm_proc = SequenceProcessorV2(
        levels=[
            {
                "ordering": "set",
                "tokens": 1,
                "code_mapping": ("ICD9CM", "CCSCM"),
            },
            {"ordering": "sequence"},  # Level 1: encounters within an admission
            {"ordering": "sequence"},  # Level 2: hospital admissions
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

    adm_proc.schema()
    # ((0, 'tokens'), (0, 'offsets'), (1, 'offsets'))

    tokens, enc_offsets, adm_offsets = adm_proc.process(patient_admissions)
    # tokens:      tensor([2, 3, 4, 5, 6, 7])  — 6 mapped CCS codes at level 0
    # enc_offsets: tensor([0, 2, 4, 4, 6])     — partitions 6 codes into 4 encounters (encounter 2 is empty)
    # adm_offsets: tensor([0, 2, 4])           — partitions 4 encounters into 2 admissions

2. Long-Format Quantitative Labs with ``(Test, Unit)`` Namespaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Irregularly sampled serum lab results in long format, binding ``(analyte, unit)``
token pairs (``tokens: 2``) to quantitative scalar readings (``values: [1]``)
and relative hours since emergency department arrival. Missing numeric readings
are flagged in ``mask``, imputed per analyte (carry forward prior reading of
the same analyte -> analyte training mean -> ``0.0``), and z-score normalized
per ``(analyte, unit)`` pair:

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

    train_labs = [
        {
            "labs": {
                "time": [0.0, 2.0, 4.0, 6.0],
                "tokens": [
                    ("Glucose", "mg/dL"),
                    ("Potassium", "mEq/L"),
                    ("Glucose", "mg/dL"),
                    ("Potassium", "mEq/L"),
                ],
                "values": [[100.0], [4.0], [120.0], [4.4]],
            }
        }
    ]
    lab_proc.fit(train_labs, "labs")
    lab_proc.schema()
    # ((0, 'tokens'), (0, 'values'), (0, 'mask'), (0, 'time'))
    lab_proc.vocab_size()
    # {0: (4, 4)}  — independent vocabularies for analyte (ns 0) and unit (ns 1)

    nan = float("nan")
    sample_labs: Records = {
        "time": [0.5, 3.0, 5.5],
        "tokens": [
            ("Glucose", "mg/dL"),
            ("Glucose", "mg/dL"),      # Pending / hemolyzed value -> carried forward from t=0.5
            ("Potassium", "mEq/L"),
        ],
        "values": [[110.0], [nan], [4.2]],
    }
    tokens, values, mask, time = lab_proc.process(sample_labs)
    # tokens: shape [3, 2], int64
    # values: tensor([[0.], [0.], [0.]])       — normalized per (analyte, unit)
    # mask:   tensor([[False], [True], [False]])
    # time:   tensor([0.5000, 3.0000, 5.5000])

3. Irregular Multichannel ICU Vital Signs Resampled Hourly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bedside Heart Rate (bpm) and Mean Arterial Pressure (mmHg) recorded at
irregular timestamps, aggregated into 1-hour averages
(``resample={"step": 1.0, "method": "avg"}``). Unobserved hours are marked in
``mask``, forward-filled from prior hours (with channel means as a leading-gap
fallback), and z-score scaled per vital channel:

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

    train_vitals = [
        {
            "vitals": {
                "time": [0.1, 0.6, 1.2, 2.5],
                "values": [[80.0, 75.0], [84.0, 79.0], [90.0, 70.0], [100.0, 65.0]],
            }
        }
    ]
    vitals_proc.fit(train_vitals, "vitals")
    vitals_proc.schema()    # ((0, 'values'), (0, 'mask')) — time is replaced by 1-hour grid steps
    vitals_proc.ordering()  # {0: 'sequence'}

    # Two readings in hour 0 [0.0, 1.0), no readings in hour 1 [1.0, 2.0), one in hour 2 [2.0, 3.0)
    values, mask = vitals_proc.process(
        {
            "time": [0.2, 0.8, 2.4],
            "values": [[82.0, 76.0], [86.0, 78.0], [96.0, 68.0]],
        }
    )
    # values: shape [3, 2], float32 (hour 1 forward-filled from hour 0 average [84.0, 77.0], then scaled)
    # mask:   tensor([[False, False], [True, True], [False, False]])

4. Timestamped Clinical Progress Notes and Multi-Lead Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-tokenized clinical notes recorded across an ICU stay use 2 levels—word or
subword tokens as a ``"sequence"`` at level ``0``, grouped into timestamped
notes as a ``"series"`` at level ``1``—while multi-lead physiological waveforms
(such as 12-lead ECG segments sampled at 500 Hz for 2 seconds) use ``spatial``
flags to distinguish anatomical leads from the temporal waveform axis:

.. code-block:: python

    import numpy as np
    from pyhealth.processors import Records, SequenceProcessorV2

    # Pre-tokenized nursing/physician notes over elapsed ICU hours
    notes_proc = SequenceProcessorV2(
        levels=[
            {"ordering": "sequence", "tokens": 1},  # Level 0: tokens within a note
            {"ordering": "series"},                 # Level 1: timestamped notes
        ]
    )
    icu_notes: Records = {
        "time": [0.0, 8.5],
        "children": [
            {"tokens": ["chief", "complaint", "acute", "dyspnea"]},
            {"tokens": ["oxygen", "saturation", "improved", "on", "nasal", "cannula"]},
        ],
    }
    notes_proc.fit([{"notes": icu_notes}], "notes")
    note_tokens, note_offsets, note_time = notes_proc.process(icu_notes)
    # note_tokens:  shape [10], int64
    # note_offsets: tensor([0, 4, 10])
    # note_time:    tensor([0.0000, 8.5000])

    # Serial 12-lead ECG strips (12 leads x 1000 samples per 2-second window)
    ecg_proc = SequenceProcessorV2(
        levels=[
            {
                "ordering": "series",
                "values": [12, 1000],
                "spatial": [False, True],  # 12 leads (channels), 1000 time ticks (spatial/temporal)
            }
        ]
    )
    ecg_strips: Records = {
        "time": [0.0, 6.0],
        "values": np.zeros((2, 12, 1000), dtype=np.float32),
    }
    ecg_proc.fit([{"ecg": ecg_strips}], "ecg")
    ecg_values, ecg_time = ecg_proc.process(ecg_strips)
    # ecg_values: shape [2, 12, 1000], ecg_proc.spatial() == {0: (False, True)}

5. Consuming Packed Batches in Downstream Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a single packed batch produced by
:meth:`~pyhealth.processors.SequenceProcessorV2.collate` (used automatically by
:func:`~pyhealth.datasets.get_dataloader`), downstream models can execute
natively on PyTorch 2.x jagged tensors without padding, materialize dense
rectangular tensors, or project unordered code sets into multi-hot vectors on
the fly:

.. code-block:: python

    import torch
    from torch.nn.utils.rnn import pad_sequence

    # Collate a batch of 2 patients (2 visits for patient 0, 1 visit for patient 1 -> 3 visits total)
    p0 = proc.process({"children": [{"tokens": ["428.0", "401.9"]}, {"tokens": ["250.00"]}]})
    p1 = proc.process({"children": [{"tokens": ["428.0"]}]})
    tokens, visit_offsets, batch_offsets = proc.collate([p0, p1])

    # Path A: Zero-padding PyTorch 2.x jagged NestedTensor (compatible with SDPA / FlashAttention)
    jagged_visits = torch.nested.nested_tensor_from_jagged(tokens, visit_offsets)

    # Path B: Dense rectangular padded tensor [total_visits, max_codes_in_batch]
    padded_visits = pad_sequence(jagged_visits.unbind(), batch_first=True, padding_value=0)
    # tensor([[2, 3],
    #         [4, 0],
    #         [2, 0]])

    # Path C: On-the-fly per-visit multi-hot matrix [total_visits, vocab_size]
    num_visits = visit_offsets.numel() - 1
    vocab_len = proc.vocab_size()[0][0]
    visit_ids = torch.repeat_interleave(torch.arange(num_visits), visit_offsets.diff())
    multihot_visits = torch.zeros(num_visits, vocab_len, dtype=torch.float32)
    multihot_visits[visit_ids, tokens] = 1.0


Mapping Common Clinical Data Structures
---------------------------------------

The table below maps common clinical data structures and modeling views to
their canonical :class:`~pyhealth.processors.Records` input, ``levels``
configuration, output :meth:`~pyhealth.processors.SequenceProcessorV2.schema`,
and model-side consumption pattern:

.. list-table::
   :header-rows: 1
   :widths: 20 24 24 16 16

   * - Clinical Data Structure
     - Raw ``Records`` Layout
     - ``levels`` Configuration
     - ``schema()`` & ``ordering()``
     - Model-Side View
   * - **Ordered flat code sequence** (chronological diagnoses or procedures)
     - ``{"tokens": ["428.0", "401.9"]}``
     - ``[{"ordering": "sequence", "tokens": 1}]``
     - ``((0, "tokens"),)``; ``{0: "sequence"}``
     - 1D ``int64`` token IDs ``[N_0]``; batch splits via ``(0, "offsets")``.
   * - **Unordered patient code set** (bag-of-codes / flat multi-hot)
     - ``{"tokens": ["428.0", "401.9"]}``
     - ``[{"ordering": "set", "tokens": 1}]``
     - ``((0, "tokens"),)``; ``{0: "set"}``
     - Embed as permutation-invariant set, or scatter ``tokens`` into ``[B, V]`` multi-hot.
   * - **Visits of medical codes** (nested code sequences & per-visit multi-hot)
     - ``{"children": [{"tokens": ["428.0", "401.9"]}, {"tokens": ["250.00"]}]}``
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]``
     - ``((0, "tokens"), (0, "offsets"))``; ``{0: "set", 1: "sequence"}``
     - Jagged ``(tokens, offsets)``, padded ``[V_tot, max_C]``, or scattered ``[V_tot, V]`` multi-hot.
   * - **Admissions -> Visits -> Codes** (3-level longitudinal history)
     - ``{"children": [{"children": [{"tokens": [...]}, ...]}, ...]}``
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}, {"ordering": "sequence"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "offsets"))``; ``{0: "set", 1: "sequence", 2: "sequence"}``
     - ``(0, "offsets")`` groups codes into visits; ``(1, "offsets")`` groups visits into admissions.
   * - **Timestamped visits of codes** (disease progression across irregular visits)
     - ``{"time": [0.0, 30.0], "children": [{"tokens": [...]}, {"tokens": [...]}]}``
     - ``[{"ordering": "set", "tokens": 1}, {"ordering": "series"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "time"))``; ``{0: "set", 1: "series"}``
     - Codes at level 0 with visit boundaries ``(0, "offsets")`` and visit times ``(1, "time")``.
   * - **Nested numeric measurements per visit**
     - ``{"children": [{"values": [[1.2], [1.5]]}, {"values": [[0.9]]}]}``
     - ``[{"ordering": "sequence", "values": [F]}, {"ordering": "sequence"}]``
     - ``((0, "values"), (0, "offsets"))``; ``{0: "sequence", 1: "sequence"}``
     - Packed ``float32`` ``[N_0, F]`` partitioned into visits via ``(0, "offsets")``.
   * - **Timestamped visits of numeric vectors**
     - ``{"time": [0.0, 14.0], "children": [{"values": [[...]]}, {"values": [[...]]}]}``
     - ``[{"ordering": "sequence", "values": [F]}, {"ordering": "series"}]``
     - ``((0, "values"), (0, "offsets"), (1, "time"))``; ``{0: "sequence", 1: "series"}``
     - Numeric features at level 0 paired with visit-level timestamps ``(1, "time")``.
   * - **Regularly gridded multichannel time series**
     - ``{"values": [[80.0, 75.0], [82.0, 78.0]]}`` (or ``"series"`` + ``resample``)
     - ``[{"ordering": "sequence", "values": [F]}]`` (or ``"series"`` with ``resample``)
     - ``((0, "values"),)`` (+ ``(0, "mask")``); ``{0: "sequence"}``
     - Dense step sequence ``[N_0, F]`` per sample; batch boundaries in ``(0, "offsets")``.
   * - **Irregular timestamped multichannel time series**
     - ``{"time": [0.0, 1.5, 4.0], "values": [[80.0, 75.0], ...]}``
     - ``[{"ordering": "series", "values": [F]}]``
     - ``((0, "values"), (0, "time"))``; ``{0: "series"}``
     - Aligned ``float32`` ``values`` ``[N_0, F]`` and ``time`` ``[N_0]``.
   * - **Long-format clinical events** (code / ``(code, unit)`` + value + timestamp)
     - ``{"time": [...], "tokens": [("Glucose", "mg/dL"), ...], "values": [[105.0], ...]}``
     - ``[{"ordering": "series", "tokens": K, "values": [1]}]``
     - ``((0, "tokens"), (0, "values"), (0, "time"))``; ``{0: "series"}``
     - Co-indexed ontology IDs, quantitative values, and observation timestamps.
   * - **Single tokenized clinical note**
     - ``{"tokens": ["acute", "hypoxic", "respiratory", "failure"]}``
     - ``[{"ordering": "sequence", "tokens": 1}]``
     - ``((0, "tokens"),)``; ``{0: "sequence"}``
     - 1D ``int64`` token sequence (text arrives pre-split into tokens).
   * - **Timestamped sequence of clinical notes**
     - ``{"time": [0.0, 8.5], "children": [{"tokens": [...]}, {"tokens": [...]}]}``
     - ``[{"ordering": "sequence", "tokens": 1}, {"ordering": "series"}]``
     - ``((0, "tokens"), (0, "offsets"), (1, "time"))``; ``{0: "sequence", 1: "series"}``
     - Tokens partitioned into notes by ``(0, "offsets")``, aligned with ``(1, "time")``.
   * - **Waveforms & serial medical images** (ECG strips, Chest X-ray series)
     - ``{"time": [0.0, 24.0], "values": array}`` (e.g., ``[N_0, C, H, W]``)
     - ``[{"ordering": "series", "values": [C, H, W], "spatial": [False, True, True]}]``
     - ``((0, "values"), (0, "time"))``; ``{0: "series"}``
     - Pre-loaded arrays with ``spatial()`` flagging channel vs. topological axes.

Constructing ``Records`` from Common Python Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside :meth:`BaseTask.__call__ <pyhealth.tasks.BaseTask.__call__>`, common
Python collections and tabular EHR query outputs map directly into
:class:`~pyhealth.processors.Records` dictionaries:

.. code-block:: python

    from pyhealth.processors import Records

    # 1. Flat list of codes: list[str]
    flat_codes = ["428.0", "401.9", "250.00"]
    rec_flat: Records = {"tokens": flat_codes}

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
    times = [0.0, 1.5, 4.0]
    vitals_matrix = [[80.0, 75.0], [84.0, 78.0], [78.0, 74.0]]
    rec_vitals: Records = {"time": times, "values": vitals_matrix}

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


Defining Configuration Subclasses
---------------------------------

Users do not need to repeat raw ``levels=[...]`` dictionaries across task
schemas. :class:`~pyhealth.processors.SequenceProcessorV2` implements all
parsing, validation, vocabulary and statistics fitting, native disk storage,
and packed collation in one engine; **subclasses are pure configuration
presets** whose ``__init__`` passes a fixed (or parameterized) ``levels`` list
to ``super().__init__(levels=...)`` without overriding ``fit()``, ``process()``,
or ``collate()``.

Registering configuration subclasses with
:func:`~pyhealth.processors.register_processor` makes them reusable by string
key or ``(key, kwargs)`` tuple in any task's ``input_schema``:

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


    @register_processor("admission_visit_codes")
    class AdmissionVisitCodesProcessor(SequenceProcessorV2):
        """Unordered codes grouped into visits, grouped into hospital admissions."""

        def __init__(self, code_mapping: tuple[str, str] | None = None):
            super().__init__(
                levels=[
                    {"ordering": "set", "tokens": 1, "code_mapping": code_mapping},
                    {"ordering": "sequence"},
                    {"ordering": "sequence"},
                ]
            )


    @register_processor("timed_visit_codes")
    class TimedVisitCodesProcessor(SequenceProcessorV2):
        """Unordered medical codes grouped into timestamped clinical visits."""

        def __init__(self):
            super().__init__(
                levels=[
                    {"ordering": "set", "tokens": 1},
                    {"ordering": "series"},
                ]
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


    # Use the registered configuration subclasses directly in a task schema:
    samples = [
        {
            "patient_id": "p0",
            "diagnoses": {
                "children": [
                    {"tokens": ["428.0", "401.9"]},
                    {"tokens": ["584.9"]},
                ]
            },
            "vitals": {
                "time": [0.2, 1.4],
                "values": [[82.0, 76.0, 98.0, 18.0], [88.0, 72.0, 97.0, 20.0]],
            },
            "readmission": 1,
        },
        {
            "patient_id": "p1",
            "diagnoses": {"children": [{"tokens": ["250.00"]}]},
            "vitals": {
                "time": [0.5],
                "values": [[74.0, 82.0, 99.0, 16.0]],
            },
            "readmission": 0,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "diagnoses": ("visit_codes", {"code_mapping": ("ICD9CM", "CCSCM")}),
            "vitals": ("hourly_vitals", {"num_channels": 4}),
        },
        output_schema={"readmission": "binary"},
    )
    batch = next(iter(get_dataloader(dataset, batch_size=2)))
    # batch["diagnoses"] -> (tokens, visit_offsets, batch_offsets)
    # batch["vitals"]    -> (values, mask, batch_offsets)


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
