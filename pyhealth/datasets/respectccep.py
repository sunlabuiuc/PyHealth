"""PyHealth dataset for RESPect CCEP

Contributor: Jacky Chen
NetID: jackyc3
Paper Title: Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a CNN Transformer
Paper Link: https://proceedings.mlr.press/v252/norris24a.html
Description: PyHealth dataset loader for RESPect CCEP (OpenNeuro ds004080) handling BIDS-style timeseries data.

PyHealth dataset for RESPect CCEP (OpenNeuro ds004080).

Dataset link:
    https://openneuro.org/datasets/ds004080

This loader scans a BIDS-style RESPect CCEP directory, extracts one
trial-averaged response per ``(recording_electrode, stimulation_pair)``,
and writes a compact metadata CSV for ``BaseDataset``. Each output row stores:

- the mean and std evoked response timeseries for one recording electrode
- the recording electrode coordinates
- the SOZ label and participant demographics

Example:
    >>> from pyhealth.datasets import RESPectCCEPDataset
    >>> dataset = RESPectCCEPDataset(root="/path/to/ds004080")
    >>> dataset.stats()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

import pyhealth.processors.label_processor as lp

_original_binary_fit = lp.BinaryLabelProcessor.fit

logger = logging.getLogger(__name__)


def _safe_fit(
    self: Any,
    samples: List[Dict[str, Any]],
    field: str,
) -> None:
    """Fit PyHealth's binary label processor with single-class safety.

    PyHealth's default binary processor expects both classes to be present
    during fitting. Small synthetic or single-patient RESPect CCEP subsets can
    legitimately contain only one label. In that case, force a binary vocab so
    downstream task setup still succeeds.

    Args:
        self: The processor instance being fitted.
        samples: Raw sample dictionaries produced by a task.
        field: Label field name to inspect.
    """
    all_labels = {sample[field] for sample in samples}
    if len(all_labels) == 1 and all_labels.issubset({0, 1}):
        self.label_vocab = {0: 0, 1: 1}
        logger.warning(
            "Only found labels %s for field '%s'; forcing binary vocab "
            "{0: 0, 1: 1}.",
            all_labels,
            field,
        )
        return
    _original_binary_fit(self, samples, field)


def _ensure_binary_label_patch() -> None:
    """Install the single-class binary-label workaround exactly once."""
    if lp.BinaryLabelProcessor.fit is not _safe_fit:
        lp.BinaryLabelProcessor.fit = _safe_fit


class RESPectCCEPDataset(BaseDataset):
    """Dataset class for RESPect CCEP SPES responses.

    The output CSV is ``respect_ccep_data-pyhealth.csv``.
    Each row corresponds to a recording electrode's evoked response to a
    stimulation pair, forming the unit of prediction for SOZ classification.

    Example:
        >>> from pyhealth.datasets import RESPectCCEPDataset
        >>> dataset = RESPectCCEPDataset(root="/path/to/ds004080")
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        tmin_s: float = 0.0,
        tmax_s: float = 1.0,
        filter_low_hz: float = 1.0,
        filter_high_hz: float = 150.0,
        resample_hz: float = 512.0,
        min_trials: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize RESPect CCEP dataset.

        Args:
            root: Root directory containing ``participants.tsv`` and ``sub-*``
                folders.
            dataset_name: Optional custom dataset name. Defaults to
                ``"respect_ccep"``.
            config_path: Optional path to YAML config file.
            tmin_s: Epoch start (seconds, post-stimulus) after cropping.
            tmax_s: Epoch end (seconds, post-stimulus) after cropping.
            filter_low_hz: Bandpass low cutoff.
            filter_high_hz: Bandpass high cutoff.
            resample_hz: Target sampling frequency for extracted epochs.
            min_trials: Minimum valid trials required for a stimulation pair.
            **kwargs: Forwarded to ``BaseDataset``.

        Raises:
            ValueError: If preprocessing arguments are inconsistent.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = str(Path(__file__).parent / "configs" / "respect_ccep.yaml")

        if tmin_s >= tmax_s:
            raise ValueError(
                f"Expected tmin_s < tmax_s, got {tmin_s} and {tmax_s}."
            )
        if filter_low_hz < 0 or filter_high_hz <= 0:
            raise ValueError(
                "Expected non-negative filter_low_hz and positive "
                f"filter_high_hz, got {filter_low_hz} and {filter_high_hz}."
            )
        if filter_low_hz >= filter_high_hz:
            raise ValueError(
                "Expected filter_low_hz < filter_high_hz, got "
                f"{filter_low_hz} and {filter_high_hz}."
            )
        if resample_hz <= 0:
            raise ValueError(
                f"Expected resample_hz > 0, got {resample_hz}."
            )
        if min_trials < 1:
            raise ValueError(f"Expected min_trials >= 1, got {min_trials}.")

        _ensure_binary_label_patch()

        self.tmin_s = tmin_s
        self.tmax_s = tmax_s
        self.filter_low_hz = filter_low_hz
        self.filter_high_hz = filter_high_hz
        self.resample_hz = resample_hz
        self.min_trials = min_trials

        self._pyhealth_csv = str(Path(root) / "respect_ccep_data-pyhealth.csv")
        if not Path(self._pyhealth_csv).exists():
            self.prepare_data(root)

        super().__init__(
            root=root,
            tables=["respectccep"],
            dataset_name=dataset_name or "respect_ccep",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _extract_bids_entities(file_path: Path) -> Dict[str, Optional[str]]:
        """Extract BIDS entities from a BIDS-style file name.

        Args:
            file_path: File path containing BIDS key-value entities.

        Returns:
            Dictionary with ``participant_id``, ``session_id``, and ``run_id``.
            Missing entities are returned as ``None``.
        """
        entities: Dict[str, Optional[str]] = {
            "participant_id": None,
            "session_id": None,
            "run_id": None,
        }
        for part in file_path.stem.split("_"):
            if part.startswith("sub-"):
                entities["participant_id"] = part
            elif part.startswith("ses-"):
                entities["session_id"] = part
            elif part.startswith("run-"):
                entities["run_id"] = part
        return entities

    @staticmethod
    def _find_run_sidecar(events_path: Path, suffix: str) -> Optional[Path]:
        """Find a run-level sidecar by replacing ``_events.tsv`` suffix.

        Args:
            events_path: Path to run events TSV.
            suffix: Target suffix such as ``"_channels.tsv"`` or ``"_ieeg.vhdr"``.

        Returns:
            Matching file path if present; otherwise ``None``.
        """
        expected = events_path.with_name(
            events_path.name.replace("_events.tsv", suffix)
        )
        if expected.exists():
            return expected
        return None

    @staticmethod
    def _find_session_sidecar(
        ieeg_dir: Path,
        participant_id: Optional[str],
        session_id: Optional[str],
        suffix: str,
    ) -> Optional[Path]:
        """Find a session-level sidecar using common BIDS naming variants.

        Args:
            ieeg_dir: Session iEEG directory.
            participant_id: Subject entity (e.g., ``sub-01``).
            session_id: Session entity (e.g., ``ses-1``).
            suffix: Target suffix such as ``"electrodes.tsv"``.

        Returns:
            Resolved sidecar path, or ``None`` if not found.
        """
        candidates: List[Path] = []
        if participant_id and session_id:
            candidates.append(ieeg_dir / f"{participant_id}_{session_id}_{suffix}")
        if participant_id:
            candidates.append(ieeg_dir / f"{participant_id}_{suffix}")
        candidates.extend(sorted(ieeg_dir.glob(f"*_{suffix}")))

        for candidate in candidates:
            if candidate.exists() and ":Zone.Identifier" not in candidate.name:
                return candidate
        return None

    @staticmethod
    def _resolve_annex_pointer(path: Path) -> Path:
        """Resolve a git-annex pointer file when needed.

        Args:
            path: Candidate file path.

        Returns:
            The resolved annex target if the file is a valid pointer;
            otherwise the original path.
        """
        if not path.exists():
            return path
        try:
            first_line = path.read_text(errors="ignore").splitlines()[0].strip()
        except Exception:
            return path

        if first_line.startswith("../") and ".git/annex/" in first_line:
            resolved = (path.parent / first_line).resolve()
            if resolved.exists():
                return resolved
        return path

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Convert a value to float.

        Args:
            value: Input value to convert.

        Returns:
            Parsed float value, or ``nan`` if conversion fails.
        """
        try:
            return float(value)
        except Exception:
            return float("nan")

    @staticmethod
    def _is_overlap(event_row: pd.Series, other_row: pd.Series) -> bool:
        """Check interval overlap using ``sample_start`` and ``sample_end``.

        Args:
            event_row: First interval row.
            other_row: Second interval row.

        Returns:
            ``True`` if the half-open intervals overlap, else ``False``.
        """
        return (
            RESPectCCEPDataset._safe_float(event_row["sample_start"])
            < RESPectCCEPDataset._safe_float(other_row["sample_end"])
            and RESPectCCEPDataset._safe_float(other_row["sample_start"])
            < RESPectCCEPDataset._safe_float(event_row["sample_end"])
        )

    @staticmethod
    def _canonical_stim_site(site: Any) -> Optional[str]:
        """Canonicalize a stimulation pair label.

        Args:
            site: Raw stimulation-site string.

        Returns:
            A sorted ``"E1-E2"`` label, or ``None`` if parsing fails.
        """
        if not isinstance(site, str):
            return None
        parts = [p.strip() for p in site.split("-") if p.strip()]
        if len(parts) < 2:
            return None
        parts = sorted(parts[:2])
        return f"{parts[0]}-{parts[1]}"

    @staticmethod
    def _coord_tuple(
        electrodes_df: pd.DataFrame,
        channel_name: str,
    ) -> Tuple[float, float, float]:
        """Fetch channel coordinates from an ``electrodes.tsv``-like table.

        Args:
            electrodes_df: Electrode metadata dataframe.
            channel_name: Recording electrode name.

        Returns:
            A ``(x, y, z)`` tuple. Missing coordinates are returned as ``nan``.
        """
        row = electrodes_df[electrodes_df["name"] == channel_name]
        if row.empty:
            return (float("nan"), float("nan"), float("nan"))
        return (
            RESPectCCEPDataset._safe_float(row.iloc[0].get("x")),
            RESPectCCEPDataset._safe_float(row.iloc[0].get("y")),
            RESPectCCEPDataset._safe_float(row.iloc[0].get("z")),
        )

    @staticmethod
    def _to_json_1d(arr: np.ndarray) -> str:
        """Serialize a 1D numeric response vector to compact JSON.

        Args:
            arr: Response vector to serialize.

        Returns:
            Compact JSON string with float32 values.
        """
        return json.dumps(arr.astype(np.float32).tolist(), separators=(",", ":"))

    def _participant_maps(
        self, participants_df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
        """Build participant-level and session-level demographics maps.

        Args:
            participants_df: Parsed ``participants.tsv`` dataframe.

        Returns:
            Tuple of ``(by_participant, by_participant_session)`` lookup maps.
        """
        by_participant = (
            participants_df.drop_duplicates(subset=["participant_id"], keep="first")
            .set_index("participant_id")
            .to_dict(orient="index")
        )
        by_participant_session: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if "session" in participants_df.columns:
            by_participant_session = (
                participants_df.drop_duplicates(
                    subset=["participant_id", "session"],
                    keep="first",
                )
                .set_index(["participant_id", "session"])
                .to_dict(orient="index")
            )
        return by_participant, by_participant_session

    def _process_run(
        self,
        participant_id: str,
        session_id: Optional[str],
        run_id: Optional[str],
        demographics: Dict[str, Any],
        events_path: Path,
    ) -> List[Dict[str, Any]]:
        """Process one run into model-ready response rows.

        Args:
            participant_id: Subject identifier.
            session_id: Session identifier.
            run_id: Run identifier.
            demographics: Demographic dictionary merged from participants table.
            events_path: Path to ``*_events.tsv``.

        Returns:
            A list of row dictionaries conforming to ``respect_ccep.yaml``.
        """
        import mne

        rows: List[Dict[str, Any]] = []

        ieeg_dir = events_path.parent
        channels_path = self._find_run_sidecar(events_path, "_channels.tsv")
        vhdr_path = self._find_run_sidecar(events_path, "_ieeg.vhdr")
        electrodes_path = self._find_session_sidecar(
            ieeg_dir, participant_id, session_id, "electrodes.tsv"
        )

        if channels_path is None or vhdr_path is None or electrodes_path is None:
            logger.warning("Missing sidecar files for %s; skipping run", events_path)
            return rows

        try:
            events_df = pd.read_csv(events_path, sep="\t")
            channels_df = pd.read_csv(channels_path, sep="\t")
            electrodes_df = pd.read_csv(electrodes_path, sep="\t")
        except Exception as exc:
            logger.warning("Failed reading TSV sidecars for %s: %s", events_path, exc)
            return rows

        resolved_vhdr = self._resolve_annex_pointer(vhdr_path)
        try:
            raw = mne.io.read_raw_brainvision(
                str(resolved_vhdr), preload=True, verbose=False
            )
        except Exception as exc:
            logger.warning("Could not read BrainVision file %s: %s", resolved_vhdr, exc)
            return rows

        if "name" not in channels_df.columns:
            logger.warning("channels.tsv missing `name` column for %s", events_path)
            return rows
        if "name" not in electrodes_df.columns:
            logger.warning("electrodes.tsv missing `name` column for %s", events_path)
            return rows

        channels_df["name"] = channels_df["name"].astype(str)
        if "status_description" in channels_df.columns:
            include_mask = (
                channels_df["status_description"]
                .astype(str)
                .str.lower()
                .eq("included")
            )
        elif "status" in channels_df.columns:
            include_mask = channels_df["status"].astype(str).str.lower().eq("good")
        else:
            include_mask = pd.Series([True] * len(channels_df), index=channels_df.index)
        chans_to_use = [
            channel
            for channel in channels_df.loc[include_mask, "name"].tolist()
            if channel in raw.ch_names
        ]
        if len(chans_to_use) < 3:
            return rows

        raw.pick(chans_to_use)
        raw.filter(
            self.filter_low_hz,
            self.filter_high_hz,
            n_jobs=1,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )

        required_cols = {
            "trial_type",
            "sample_start",
            "sample_end",
            "electrical_stimulation_site",
        }
        if not required_cols.issubset(set(events_df.columns)):
            logger.warning("events.tsv missing required columns for %s", events_path)
            return rows

        events_df["sample_start"] = pd.to_numeric(
            events_df["sample_start"],
            errors="coerce",
        )
        events_df["sample_end"] = pd.to_numeric(
            events_df["sample_end"],
            errors="coerce",
        )

        stim_events = events_df[
            events_df["trial_type"]
            .astype(str)
            .isin(["electrical_stimulation", "stimulation"])
        ].copy()
        stim_events = stim_events.dropna(
            subset=[
                "sample_start",
                "sample_end",
                "electrical_stimulation_site",
            ]
        )
        stim_events = stim_events[stim_events["sample_start"] < raw.n_times]
        stim_events["electrical_stimulation_site"] = (
            stim_events["electrical_stimulation_site"].apply(
                self._canonical_stim_site
            )
        )
        stim_events = stim_events.dropna(subset=["electrical_stimulation_site"])
        if stim_events.empty:
            return rows

        artefacts_all = events_df[
            (events_df["trial_type"].astype(str) == "artefact")
            & (events_df.get("electrodes_involved_onset", "n/a").astype(str) == "all")
        ].copy()
        seizures = events_df[events_df["trial_type"].astype(str) == "seizure"].copy()
        focal_artefacts = events_df[
            (events_df["trial_type"].astype(str) == "artefact")
            & (events_df.get("electrodes_involved_onset", "all").astype(str) != "all")
        ].copy()

        valid_rows: List[int] = []
        for idx, stim_row in stim_events.iterrows():
            overlaps_global_art = any(
                self._is_overlap(stim_row, art_row)
                for _, art_row in artefacts_all.iterrows()
            )
            overlaps_seizure = any(
                self._is_overlap(stim_row, sei_row)
                for _, sei_row in seizures.iterrows()
            )
            if not overlaps_global_art and not overlaps_seizure:
                valid_rows.append(idx)
        stim_events = stim_events.loc[valid_rows]
        if stim_events.empty:
            return rows

        for stim_site, site_df in stim_events.groupby("electrical_stimulation_site"):
            stim_pair = self._canonical_stim_site(stim_site)
            if stim_pair is None:
                continue
            stim_1, stim_2 = stim_pair.split("-")

            remove_chans: set[str] = set()
            for _, stim_row in site_df.iterrows():
                for _, focal_row in focal_artefacts.iterrows():
                    if not self._is_overlap(stim_row, focal_row):
                        continue
                    involved = str(focal_row.get("electrodes_involved_onset", ""))
                    for chan in involved.split(","):
                        chan = chan.strip()
                        if chan:
                            remove_chans.add(chan)

            recording_channels = [
                ch
                for ch in raw.ch_names
                if ch not in {stim_1, stim_2} and ch not in remove_chans
            ]
            if len(recording_channels) == 0:
                continue

            events_arr = np.zeros((len(site_df), 3), dtype=int)
            events_arr[:, 0] = site_df["sample_start"].astype(int).to_numpy()
            events_arr[:, 2] = 1

            try:
                epochs = mne.Epochs(
                    raw,
                    events_arr,
                    event_id={"stim": 1},
                    tmin=self.tmin_s - 1.0,
                    tmax=self.tmax_s,
                    picks=recording_channels,
                    preload=True,
                    baseline=(None, -0.1),
                    reject_by_annotation=False,
                    verbose=False,
                )
                # Clamp crop bounds to available epoch support to avoid
                # repeated MNE warnings when floating-point sample grids make
                # requested tmax marginally out-of-bounds.
                crop_tmin = max(float(self.tmin_s), float(epochs.tmin))
                crop_tmax = min(float(self.tmax_s), float(epochs.tmax))
                epochs.crop(tmin=crop_tmin, tmax=crop_tmax)
                epochs.resample(self.resample_hz, verbose=False)
            except Exception as exc:
                logger.debug(
                    "Epoch extraction failed for %s %s: %s",
                    participant_id,
                    stim_site,
                    exc,
                )
                continue

            if len(epochs) < self.min_trials:
                continue

            epoch_data = epochs.get_data()  # [trials, channels, time]
            mean_resp = epoch_data.mean(axis=0).astype(np.float32)
            std_resp = epoch_data.std(axis=0).astype(np.float32)

            for ch_idx, rec_chan in enumerate(epochs.ch_names):
                rec_coord = self._coord_tuple(electrodes_df, rec_chan)

                rec_row = electrodes_df[electrodes_df["name"] == rec_chan]
                soz_label = 0
                if not rec_row.empty:
                    soz_label = int(
                        str(rec_row.iloc[0].get("soz", "no")).lower() == "yes"
                    )

                rows.append(
                    {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "run_id": run_id,
                        "age": demographics.get("age"),
                        "sex": demographics.get("sex"),
                        "recording_electrode": rec_chan,
                        "stim_1": stim_1,
                        "stim_2": stim_2,
                        "response_ts": self._to_json_1d(mean_resp[ch_idx]),
                        "response_ts_std": self._to_json_1d(std_resp[ch_idx]),
                        "soz_label": soz_label,
                        "recording_x": rec_coord[0],
                        "recording_y": rec_coord[1],
                        "recording_z": rec_coord[2],
                    }
                )

        return rows

    def prepare_data(self, root: str) -> None:
        """Generate the RESPect CCEP data table used by ``BaseDataset``.

        Args:
            root: Dataset root containing ``participants.tsv`` and ``sub-*`` folders.

        Raises:
            FileNotFoundError: If the dataset root or ``participants.tsv`` is
                missing.
            ValueError: If ``participants.tsv`` lacks ``participant_id``.
        """
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        participants_path = root_path / "participants.tsv"
        if not participants_path.exists():
            raise FileNotFoundError(f"participants.tsv not found: {participants_path}")

        participants_df = pd.read_csv(participants_path, sep="\t")
        if "participant_id" not in participants_df.columns:
            raise ValueError("participants.tsv must contain a 'participant_id' column")

        by_participant, by_participant_session = self._participant_maps(participants_df)

        event_files: List[Path] = []
        for subject_dir in sorted(root_path.glob("sub-*")):
            if subject_dir.is_dir():
                event_files.extend(sorted(subject_dir.glob("ses-*/ieeg/*_events.tsv")))

        all_rows: List[Dict[str, Any]] = []
        for events_path in event_files:
            if ":Zone.Identifier" in events_path.name:
                continue
            entities = self._extract_bids_entities(events_path)
            participant_id = entities["participant_id"]
            session_id = entities["session_id"]
            run_id = entities["run_id"]
            if participant_id is None:
                continue

            demographics = by_participant.get(participant_id, {}).copy()
            if session_id is not None:
                demographics.update(
                    by_participant_session.get(
                        (participant_id, session_id),
                        {},
                    )
                )

            run_rows = self._process_run(
                participant_id=participant_id,
                session_id=session_id,
                run_id=run_id,
                demographics=demographics,
                events_path=events_path,
            )
            all_rows.extend(run_rows)

        cols = [
            "participant_id",
            "session_id",
            "run_id",
            "age",
            "sex",
            "recording_electrode",
            "stim_1",
            "stim_2",
            "response_ts",
            "response_ts_std",
            "soz_label",
            "recording_x",
            "recording_y",
            "recording_z",
        ]
        out_df = pd.DataFrame(all_rows, columns=cols)
        out_df.to_csv(self._pyhealth_csv, index=False)
        logger.info("Wrote %d rows to %s", len(out_df), self._pyhealth_csv)
