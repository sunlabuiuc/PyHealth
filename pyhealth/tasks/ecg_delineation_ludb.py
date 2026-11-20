"""ECG delineation on LUDB: label each point as P wave, QRS complex or T wave.

Authors: Shray Srivastava (ssriv5@illinois.edu), Sid Wanjara (swanj2@illinois.edu)

Paper: J. Park et al. "Benchmarking ECG Delineation using Deep Neural
Network-based Semantic Segmentation Models." CHIL 2025, PMLR 287:63-88.
https://proceedings.mlr.press/v287/park25a.html

Description: The paper treats ECG delineation as 1D segmentation. Each lead
of each record is one sample and every point of the signal gets a label:
background, P wave, QRS complex or T wave. Signals are filtered with a
Butterworth band-pass (0.05 Hz high-pass, 150 Hz low-pass) first.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# The 12 leads of every LUDB record. Each lead's annotation file uses the
# lead name as its extension, e.g. "1.ii".
LUDB_LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")

# Mask value for points the cardiologists did not annotate (before the first
# wave and after the last one). Models and metrics should ignore it.
IGNORE_INDEX = 255

# Mask value of each wave, keyed by the WFDB peak symbol LUDB uses.
WAVE_LABELS = {"p": 1, "N": 2, "t": 3}

# Butterworth band-pass from the paper (Section 3.2), in Hz.
FILTER_BAND_HZ = (0.05, 150.0)


def annotations_to_mask(
    symbols: list[str], samples: list[int], n_samples: int
) -> np.ndarray:
    """Turn the annotations of one LUDB lead into a label for every point.

    LUDB marks each wave as "(" (onset), a peak symbol ("p", "N" or "t") and
    ")" (offset). A few waves are missing their onset or offset, so for each
    peak we take the closest "(" before it and ")" after it, without going
    past the neighbouring peaks. If one is missing we use the peak itself.

    Labels: 0 background, 1 P wave, 2 QRS complex, 3 T wave. Points before
    the first wave and after the last wave were not annotated, so they get
    IGNORE_INDEX (255).

    Args:
        symbols: Annotation symbols in time order.
        samples: Sample index of each symbol.
        n_samples: Length of the signal.

    Returns:
        An int64 array of length n_samples.

    Raises:
        ValueError: If symbols and samples have different lengths.

    Notes:
        How to write it:

        1. Check the two lists are the same length (else ValueError).
        2. Start with ``mask = np.full(n_samples, IGNORE_INDEX)`` (int64).
        3. Find the positions in ``symbols`` that are peaks (keys of
           ``WAVE_LABELS``). If there are none, return the mask as is.
        4. For each peak, search backwards for the nearest "(" and forwards
           for the nearest ")". Stop each search at the previous / next
           peak, so a wave never borrows its neighbour's boundary. If
           nothing is found, use the peak's own sample. Keep
           ``(label, onset, offset)``.
        5. Set everything from the first onset to the last offset to 0
           (background), then paint each wave's label from onset to offset.
           Both ends are included, so slice to ``offset + 1``.

        Why not just read three symbols at a time? About 300 of LUDB's
        175,000 marks break the "(" peak ")" pattern, mostly QRS complexes
        written as "N )" with no onset. With the rule above the masks match
        the paper authors' published annotations on all but 1,168 of about
        8.8 million labelled points. The ``>>>`` example below is a good
        first check for your version.

    Examples:
        >>> from pyhealth.tasks.ecg_delineation_ludb import annotations_to_mask
        >>> symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
        >>> samples = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        >>> annotations_to_mask(symbols, samples, 14).tolist()
        [255, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 255, 255]
    """
    if len(symbols) != len(samples):
        raise ValueError(f"Got {len(symbols)} symbols but {len(samples)} samples.")

    mask = np.full(n_samples, IGNORE_INDEX, dtype=np.int64)
    peaks = [i for i, symbol in enumerate(symbols) if symbol in WAVE_LABELS]
    if len(peaks) == 0:
        return mask

    waves = []
    for k, peak in enumerate(peaks):
        # Only search between this peak and the peaks next to it.
        prev_peak = peaks[k - 1] if k > 0 else -1
        next_peak = peaks[k + 1] if k < len(peaks) - 1 else len(symbols)

        onset = samples[peak]
        for i in range(peak - 1, prev_peak, -1):
            if symbols[i] == "(":
                onset = samples[i]
                break

        offset = samples[peak]
        for i in range(peak + 1, next_peak):
            if symbols[i] == ")":
                offset = samples[i]
                break

        waves.append((WAVE_LABELS[symbols[peak]], onset, offset))

    # Everything between the first and last wave is background to start with.
    first = max(min(onset for _, onset, _ in waves), 0)
    last = max(offset for _, _, offset in waves)
    mask[first : last + 1] = 0
    for label, onset, offset in waves:
        mask[max(onset, 0) : offset + 1] = label
    return mask


class ECGDelineationLUDB(BaseTask):
    """Point-wise P / QRS / T segmentation of LUDB leads.

    Runs on :class:`~pyhealth.datasets.LUDBDataset`. Each lead of each record
    becomes one sample, as in Park et al. (2025). For each lead the task
    reads the signal, optionally adds noise (for the robustness ablation),
    applies the paper's 0.05-150 Hz Butterworth band-pass, and builds the
    label mask with :func:`annotations_to_mask`.

    Each sample has:

    - ``signal``: float32, shape ``(1, 1, n_samples)``, in mV. The extra
      dimension makes it a one-row image so PyHealth's ``UNet2D`` can use it.
    - ``mask``: int64, shape ``(1, n_samples)``, values 0 background, 1 P,
      2 QRS, 3 T and 255 for unannotated points (use ``ignore_index=255``).
    - ``patient_id`` (the record number) and ``lead``.

    The paper predicts three sigmoid channels with binary cross-entropy. We
    use one class per point instead, which holds the same information
    because LUDB's waves never overlap, and works with PyHealth's
    segmentation models and metrics.

    Args:
        leads: Leads to use, e.g. ``["ii"]`` (any case). Defaults to all 12.
        apply_filter: Whether to apply the band-pass filter. Defaults to True.
        noise_snr_db: If set, add white Gaussian noise to each lead at this
            signal-to-noise ratio (in dB) before filtering. Defaults to None.
        noise_seed: Seed for the noise, so every run gets the same noise.

    Raises:
        ValueError: If ``leads`` is empty, has duplicates or unknown names.

    Examples:
        >>> from pyhealth.tasks import ECGDelineationLUDB
        >>> task = ECGDelineationLUDB(leads=["II"], noise_snr_db=10.0)
        >>> task.leads
        ['ii']
        >>> from pyhealth.datasets import LUDBDataset  # doctest: +SKIP
        >>> dataset = LUDBDataset(root="/path/to/ludb")  # doctest: +SKIP
        >>> samples = dataset.set_task(ECGDelineationLUDB())  # doctest: +SKIP
        >>> samples[0]["signal"].shape, samples[0]["mask"].shape  # doctest: +SKIP
        (torch.Size([1, 1, 5000]), torch.Size([1, 5000]))
    """

    task_name: str = "ECGDelineationLUDB"
    input_schema: ClassVar[dict[str, Any]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "mask": ("tensor", {"dtype": torch.long})
    }

    def __init__(
        self,
        leads: list[str] | None = None,
        apply_filter: bool = True,
        noise_snr_db: float | None = None,
        noise_seed: int = 0,
    ) -> None:
        """Check the lead names and store the task settings.

        Notes:
            How to write it:

            1. Call ``super().__init__()``.
            2. If ``leads`` is None, use all of ``LUDB_LEADS``. Lower-case
               every name, because the annotation files use lower-case leads
               (``1.avr``, not ``1.aVR``).
            3. Raise ValueError if the list is empty, has duplicates
               (compare ``len(set(...))`` with ``len(...)``) or has a name
               that isn't in ``LUDB_LEADS``.
            4. Store the other three arguments on ``self``.

            set_task caches its samples using these settings. If you change
            only the code of ``__call__``, run
            ``dataset.clear_task_cache(task)`` or you'll get the old samples.
        """
        super().__init__()
        if leads is None:
            leads = list(LUDB_LEADS)
        self.leads = [lead.lower() for lead in leads]
        if len(self.leads) == 0:
            raise ValueError("leads must name at least one lead.")
        if len(set(self.leads)) != len(self.leads):
            raise ValueError(f"leads has duplicates: {self.leads}.")
        for lead in self.leads:
            if lead not in LUDB_LEADS:
                raise ValueError(
                    f"Unknown lead {lead!r}; expected one of {LUDB_LEADS}."
                )
        self.apply_filter = apply_filter
        self.noise_snr_db = noise_snr_db
        self.noise_seed = noise_seed

    def add_noise(self, signal: np.ndarray, record_id: str, lead: str) -> np.ndarray:
        """Add white Gaussian noise to one lead at noise_snr_db.

        The noise power is the signal power divided by 10^(snr/10). The random
        generator is seeded with the noise seed, record and lead, so the same
        lead always gets the same noise. Returns the signal unchanged if
        noise_snr_db is None.

        Notes:
            How to write it:

            1. If ``noise_snr_db`` is None, return the signal unchanged.
            2. Make a generator with ``np.random.default_rng([noise_seed,
               int(record_id), lead_index])``, where ``lead_index`` is
               ``LUDB_LEADS.index(lead)``. Seeding per record and lead keeps
               the noise identical no matter what order PyHealth processes
               patients in.
            3. Signal power = mean of ``signal ** 2``. Noise power = signal
               power / ``10 ** (snr / 10)``.
            4. Return ``signal + rng.normal(0, sqrt(noise power),
               size=len(signal))``.

            To check it: measure the SNR of what you added; it should come
            out at the requested dB (within a few tenths). This is called
            before filtering on purpose, to mimic noise picked up while
            recording.
        """
        if self.noise_snr_db is None:
            return signal
        rng = np.random.default_rng(
            [self.noise_seed, int(record_id), LUDB_LEADS.index(lead)]
        )
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (self.noise_snr_db / 10))
        return signal + rng.normal(0.0, np.sqrt(noise_power), size=len(signal))

    def filter_signal(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply the paper's Butterworth band-pass to one lead.

        Uses a 2nd-order filter run forwards and backwards so the waves are
        not shifted in time. The record is mirrored at both ends before
        filtering: the 0.05 Hz high-pass reacts very slowly, and scipy's
        default short padding left a slow drift near the edges.

        Raises:
            ValueError: If the sampling rate is too low for the 150 Hz cut-off.

        Notes:
            How to write it (scipy.signal):

            1. If ``apply_filter`` is False, return the signal unchanged.
            2. A low-pass at 150 Hz only works if the sampling rate is above
               300 Hz (twice the cut-off), so raise ValueError otherwise.
               LUDB is 500 Hz.
            3. Design the filter with ``butter(2, [0.05, 150],
               btype="bandpass", fs=sampling_rate, output="sos")``.
            4. Apply it with ``sosfiltfilt(sos, signal, padtype="even",
               padlen=len(signal) - 1)``. ``sosfiltfilt`` runs forwards then
               backwards, so the waves don't shift in time.

            The padding in step 4 is the non-obvious part. With scipy's
            default padding, a 10 Hz test wave came out 1.39x too big
            because of a slow drift from the record edges. Mirroring the
            whole record fixes most of it. To check your version, filter a
            pure 10 Hz wave (should stay about the same size), a constant
            (should go to ~0) and a 200 Hz wave (should mostly disappear).
            The paper doesn't give the filter order; 2 is our choice.
        """
        if not self.apply_filter:
            return signal
        low, high = FILTER_BAND_HZ
        if high >= sampling_rate / 2:
            raise ValueError(
                f"The {high} Hz low-pass needs a sampling rate above {2 * high} "
                f"Hz; this record is {sampling_rate} Hz."
            )
        sos = butter(2, [low, high], btype="bandpass", fs=sampling_rate, output="sos")
        return sosfiltfilt(sos, signal, padtype="even", padlen=len(signal) - 1)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Make one sample per lead of the patient's record.

        Leads with no annotation file or no annotated wave are skipped with a
        warning, and so are records whose signal cannot be read.

        Notes:
            How to write it (the wfdb package reads PhysioNet files):

            1. Import wfdb inside this method, not at the top of the file,
               because it's optional and ``import pyhealth.tasks`` must work
               without it. Raise a helpful ImportError if it's missing.
            2. Loop over ``patient.get_events("records")`` (one per record).
            3. Read the signal with ``signal, fields =
               wfdb.rdsamp(event.signal_file)``. ``signal`` is (n_samples,
               12), so one lead is a column; ``fields["sig_name"]`` gives
               the lead names and ``fields["fs"]`` the sampling rate. If the
               file can't be read, log a warning and skip the record.
            4. For each lead in ``self.leads``:

               a. The annotation file is ``signal_file + "." + lead``. If it
                  doesn't exist, warn and skip the lead.
               b. Read it with ``wfdb.rdann(event.signal_file, lead)`` and
                  pass ``.symbol`` and ``.sample`` to
                  ``annotations_to_mask``. If the mask is all 255, warn and
                  skip.
               c. Take the lead's column, call ``add_noise`` then
                  ``filter_signal`` (noise first, then filter).
               d. Append a dict with ``patient_id``, ``lead``, ``signal``
                  reshaped to (1, 1, n) as float32, and ``mask`` reshaped to
                  (1, n).

            The (1, 1, n) shape only exists so UNet2D sees a one-row image.
            On real LUDB this makes all 2,400 samples in about 15 s.
        """
        try:
            import wfdb
        except ImportError as error:
            raise ImportError(
                "Reading LUDB needs the wfdb package: pip install wfdb"
            ) from error

        samples = []
        for event in patient.get_events("records"):
            try:
                signal, fields = wfdb.rdsamp(event.signal_file)
            except (FileNotFoundError, ValueError) as error:
                logger.warning("Skipping record %s: %s", event.signal_file, error)
                continue
            lead_names = [name.lower() for name in fields["sig_name"]]

            for lead in self.leads:
                annotation_file = Path(f"{event.signal_file}.{lead}")
                if lead not in lead_names or not annotation_file.exists():
                    logger.warning(
                        "Record %s has no %s lead annotations.",
                        patient.patient_id,
                        lead,
                    )
                    continue

                annotation = wfdb.rdann(event.signal_file, lead)
                mask = annotations_to_mask(
                    annotation.symbol, list(annotation.sample), len(signal)
                )
                if np.all(mask == IGNORE_INDEX):
                    logger.warning(
                        "Record %s lead %s has no waves.", patient.patient_id, lead
                    )
                    continue

                values = signal[:, lead_names.index(lead)]
                values = self.add_noise(values, patient.patient_id, lead)
                values = self.filter_signal(values, fields["fs"])
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "lead": lead,
                        "signal": values.astype(np.float32).reshape(1, 1, -1),
                        "mask": mask.reshape(1, -1),
                    }
                )
        return samples
