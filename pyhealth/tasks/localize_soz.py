import mne
import numpy as np
import pandas as pd
import pyreadr
from pathlib import Path
from typing import Any, Dict, List

from pyhealth.tasks import BaseTask


def _with_distance_first(df):
    """Reorder DataFrame columns so 'distances' is first (position 0)."""
    if 'distances' not in df.columns:
        return df
    return df[['distances'] + [c for c in df.columns if c != 'distances']]


def pad_and_stack(arrays, max_rows, pad_value=0):
    """Pad each 2D array to max_rows and stack them into one array."""
    padded_arrays = []
    for array in arrays:
        rows_to_add = max_rows - array.shape[0]
        if rows_to_add > 0:
            array = np.pad(
                array,
                ((0, rows_to_add), (0, 0)),
                "constant",
                constant_values=(pad_value,),
            )
        padded_arrays.append(array)

    return np.stack(padded_arrays)


def process_stimulation_sites(site):
    """Sort a bipolar stimulation site string so E2-E1 and E1-E2 match."""
    parts = site.split("-")
    parts.sort()
    return "-".join(parts)


def is_overlap(event, artifact):
    """Return True if *event* and *artifact* overlap in sample space."""
    return (
        event["sample_start"] < artifact["sample_end"]
        and artifact["sample_start"] < event["sample_end"]
    )


def combine_stats(group):
    """Combine mean/std rows for duplicated stimulation-recording groups."""
    means = group[group["metric"] == "mean"].select_dtypes(include=[np.number])
    stds = group[group["metric"] == "std"].select_dtypes(include=[np.number])

    n = 10
    if len(means) > 1 or len(stds) > 1:
        total_samples = n * len(means)
        combined_mean = means.mean()
        combined_std = np.sqrt(
            (stds**2).mean()
            + ((means - combined_mean) ** 2).sum() / (total_samples - 1)
        )
    else:
        combined_mean = means.iloc[0]
        combined_std = stds.iloc[0]

    mean_df = pd.DataFrame(combined_mean).transpose()
    std_df = pd.DataFrame(combined_std).transpose()

    for col in ["subject", "recording", "stim_1", "stim_2"]:
        mean_df[col] = group[col].iloc[0]
        std_df[col] = group[col].iloc[0]

    mean_df["metric"] = "mean"
    std_df["metric"] = "std"

    return mean_df, std_df


regions = ["Frontal", "Insula", "Limbic", "Occipital", "Parietal", "Temporal", "Unknown"]
region_to_index = {region: index for index, region in enumerate(regions)}
_destrieux_df = None


def get_destrieux_lobe(label):
    """Return the Destrieux lobe name for a numeric atlas label."""
    try:
        is_missing = bool(np.isnan(label))
    except (TypeError, ValueError):
        is_missing = label is None

    if is_missing or label == 0:
        return "Unknown"

    global _destrieux_df
    if _destrieux_df is None:
        try:
            rda_path = Path(__file__).resolve().parent.parent / "destrieux.rda"
            _destrieux_df = pyreadr.read_r(str(rda_path))["destrieux"]
        except Exception:
            return "Unknown"

    try:
        return _destrieux_df[_destrieux_df.index == int(label) - 1].lobe.values[0]
    except Exception:
        return "Unknown"


class StimulationDataProcessor:
    """Extracts per-stimulation-site EEG epochs and computes mean/std responses.

    Args:
        tmin: Start of the epoch window in seconds relative to the stimulus.
        tmax: End of the epoch window in seconds relative to the stimulus.
    """

    def __init__(self, tmin, tmax):
        """
        Args:
            tmin: Start of the epoch window in seconds relative to the stimulus.
            tmax: End of the epoch window in seconds relative to the stimulus.
        """
        self.tmin = tmin
        self.tmax = tmax

    def process_run_data(self, eeg, events_df, channels_df, subject):
        """Process and extract stimulation response data for a single EEG run.

        Filters for valid electrical-stimulation events (excluding artifacts and
        seizures), epochs the data around each stimulation site, and returns a
        DataFrame with mean and std responses across trials.

        Args:
            eeg: MNE Raw object for the run.
            events_df: DataFrame of BIDS events (``*_events.tsv``).
            channels_df: DataFrame of channel metadata (``*_channels.tsv``).
            subject: Subject identifier string used in warning messages.

        Returns:
            DataFrame with columns ``[subject, recording, stim_1, stim_2,
            metric, <timestep columns>]``, or ``None`` if no valid stimulation
            events were found.
        """
        # Filter channels
        chans_to_use = channels_df[channels_df.status_description == "included"].index.tolist()
        eeg.pick(chans_to_use)

        # Filter EEG data
        eeg.filter(1, 150, n_jobs=1, method='fir', fir_design='firwin')

        # Filter for electrical stimulation events
        stim_events = events_df[events_df.trial_type == "electrical_stimulation"].copy()
        
        # Filter for events that occur within the EEG data
        before = stim_events.shape[0]
        stim_events = stim_events[stim_events['sample_start'] < eeg.n_times]
        after = stim_events.shape[0]

        if before != after:
            print(subject, before, after)

        # Filter for artifact events
        artifacts = events_df[np.logical_and(events_df.trial_type == "artifact", 
                                             events_df.electrodes_involved_onset == "all")].copy()

        # Filter for seizure events
        seizures = events_df[events_df.trial_type == "seizure"].copy()

        # Creating artifact mask
        artifact_mask = []
        for _, stim_event in stim_events.iterrows():
            overlap = any(is_overlap(stim_event, artifact) for _, artifact in artifacts.iterrows())
            artifact_mask.append(overlap)

        # Creating the artmask
        seizure_mask = []
        for _, stim_event in stim_events.iterrows():
            overlap = any(is_overlap(stim_event, seizure) for _, seizure in seizures.iterrows())
            seizure_mask.append(overlap)

        # Create mask of valid events
        valid_mask = np.logical_not(np.logical_or(artifact_mask, seizure_mask))

        # Filter for valid events
        stim_events = stim_events[valid_mask]

        # Filter for artifact events occurring on only some channels
        focal_artifacts = events_df[np.logical_and(events_df.trial_type == "artifact", 
                                                   events_df.electrodes_involved_onset != "all")].copy()

        # Sort the 'electrical_stimulation_site' column - this ensures that E2-E1 and E1-E2 are treated the same
        stim_events['electrical_stimulation_site'] = stim_events['electrical_stimulation_site'].apply(process_stimulation_sites)

        # Step 1: Convert 'electrical_stimulation_site' to a categorical datatype
        stim_events['electrical_stimulation_site'] = stim_events['electrical_stimulation_site'].astype('category')

        # Get mapping of categories to integer codes for 'electrical_stimulation_site'
        category_mapping = dict(enumerate(stim_events['electrical_stimulation_site'].cat.categories))

        # Create a new column 'electrical_stimulation_site_cat' with integer codes for 'electrical_stimulation_site'
        stim_events['electrical_stimulation_site_cat'] = stim_events['electrical_stimulation_site'].cat.codes

        # Step 2: Add a column 'zero_column' to the DataFrame with all values set to 0
        stim_events['zero_column'] = 0

        # Step 3: Extract the columns 'sample_start', 'zero_column', and 'electrical_stimulation_site_cat' and convert to a NumPy array
        result_array = stim_events[['sample_start', 'zero_column', 'electrical_stimulation_site_cat']].values

        # Creating the artmask
        overlap_list = []

        for _, stim_event in stim_events.iterrows():
            overlap_found = False
            for _, artifact in focal_artifacts.iterrows():
                if is_overlap(stim_event, artifact):
                    overlap_list.append(artifact.electrodes_involved_onset)
                    overlap_found = True
                    break  # Stop checking after the first overlap is found
            if not overlap_found:
                overlap_list.append(None)
        overlap_list = np.array(overlap_list)

        response_dfs = []
        
        for event_id in np.unique(result_array[:, 2]):

            # Chans to remove due to artifact
            remove_chans = overlap_list[result_array[:, 2] == event_id]
            remove_chans = np.unique([chan for chan_list in remove_chans if chan_list is not None for chan in chan_list.split(',')])

            response_df = self._extract_epochs(eeg, 
                                               result_array, 
                                               event_id, 
                                               category_mapping, 
                                               subject, 
                                               remove_chans)
            if response_df is not None:
                response_dfs.append(response_df)

        try:
            response_dfs = pd.concat(response_dfs)
        except ValueError:
            print(f"No stimulation events found for subject {subject}")
            return None        

        return response_dfs

    def _extract_epochs(self, eeg, result_array, event_id, category_mapping, subject, remove_chans):
        """Extract epochs for one stimulation site and return mean/std DataFrame.

        Args:
            eeg: MNE Raw object for the run.
            result_array: Array of shape ``(n_events, 3)`` with columns
                ``[sample_start, zero, event_id]``.
            event_id: Integer code identifying the stimulation site to epoch.
            category_mapping: Dict mapping event_id codes to site strings.
            subject: Subject identifier string inserted into the output DataFrame.
            remove_chans: Array of channel names to exclude (focal artifacts).

        Returns:
            DataFrame with mean and std rows for this stimulation site, or
            ``None`` if fewer than 5 artifact-free trials remain.
        """
        stimulated_electrodes = category_mapping[event_id].split('-')

        # When polarity is reversed, ensure no duplication
        stimulated_electrodes.sort()

        # Get list of electrodes excluding stimulated electrodes
        recording_channels = [chan for chan in eeg.info['ch_names'] if chan not in stimulated_electrodes]

        # Remove channels that are artifacted
        recording_channels = [chan for chan in recording_channels if chan not in remove_chans]

        try:
            epochs = mne.Epochs(eeg, result_array, event_id=event_id, tmin=self.tmin - 1
                                , tmax=self.tmax, picks=recording_channels, preload=True, baseline=(None, -0.1))
            if len(epochs) < 5:
                return None
            epochs.crop(tmin=self.tmin)
            epochs.resample(512)
        except RuntimeError as e:
            if "empty" in str(e).lower() or "epochs were dropped" in str(e).lower():
                return None
            raise

        # If less than 5 trials, return None
        if (epochs._data.shape[0]) < 5:
            return None
                
        # Calculate mean and standard deviation
        mean_response = epochs.average()._data  # Shape: (channels, time steps)
        std_response = epochs._data.std(axis=0)  # Shape: (channels, time steps)

        # Create DataFrame for mean response
        df_mean_response = pd.DataFrame(mean_response).astype('float32')
        df_mean_response.insert(0, 'subject', [subject] * mean_response.shape[0])
        df_mean_response.insert(1, 'recording', epochs.info['ch_names'])
        df_mean_response.insert(2, 'stim_1', stimulated_electrodes[0])
        df_mean_response.insert(3, 'stim_2', stimulated_electrodes[1])
        df_mean_response.insert(4, 'metric', 'mean')

        # Create DataFrame for std response
        df_std_response = pd.DataFrame(std_response).astype('float32')
        df_std_response.insert(0, 'subject', [subject] * std_response.shape[0])
        df_std_response.insert(1, 'recording', epochs.info['ch_names'])
        df_std_response.insert(2, 'stim_1', stimulated_electrodes[0])
        df_std_response.insert(3, 'stim_2', stimulated_electrodes[1])
        df_std_response.insert(4, 'metric', 'std')

        # Concatenate the two DataFrames
        response_df = pd.concat([df_mean_response, df_std_response], ignore_index=True)

        return response_df


class DatasetCreator:
    """Converts per-run response DataFrames into analysis-ready arrays.

    Args:
        response_df: DataFrame produced by :class:`StimulationDataProcessor`
            containing mean and std CCEP responses for one or more subjects.
    """

    def __init__(self, response_df):
        self.response_df = response_df

    def process_for_analysis(self, subject, electrodes_df):
        """Build paired mean/std tensors and labels for all electrodes.

        Calls :meth:`process_metric_for_analysis` for both ``"mean"`` and
        ``"std"`` metrics and aligns the results to channels present in both.

        Args:
            subject: Subject identifier string.
            electrodes_df: DataFrame indexed by electrode name with columns
                ``x``, ``y``, ``z``, ``soz``, and ``Destrieux_label``.

        Returns:
            A 6-tuple ``(channels, electrode_lobes, y, electrode_coords,
            X_stim, X_recording)`` where ``X_stim`` and ``X_recording`` have
            shape ``(n_electrodes, 2, n_trials, n_timesteps)``, or ``None``
            if no paired channels were found.
        """
        mean_output = self.process_metric_for_analysis(
            subject, electrodes_df, "mean", labels=True
        )
        std_output = self.process_metric_for_analysis(
            subject, electrodes_df, "std", labels=True
        )
        if mean_output is None or std_output is None:
            return None

        (
            mean_channels,
            mean_lobes,
            mean_y,
            mean_coords,
            mean_X_stim,
            mean_X_recording,
        ) = mean_output
        (
            std_channels,
            _std_lobes,
            _std_y,
            _std_coords,
            std_X_stim,
            std_X_recording,
        ) = std_output

        std_channel_to_index = {
            channel: index for index, channel in enumerate(std_channels)
        }
        common_channels = [
            channel for channel in mean_channels if channel in std_channel_to_index
        ]
        if len(common_channels) == 0:
            print(f"No paired mean/std stimulation events found for subject {subject}")
            return None

        X_stim = []
        X_recording = []
        electrode_lobes = []
        electrode_coords = []
        y = []
        for mean_index, channel in enumerate(mean_channels):
            if channel not in std_channel_to_index:
                continue
            std_index = std_channel_to_index[channel]
            X_stim.append(np.stack([mean_X_stim[mean_index], std_X_stim[std_index]]))
            X_recording.append(
                np.stack([mean_X_recording[mean_index], std_X_recording[std_index]])
            )
            electrode_lobes.append(mean_lobes[mean_index])
            electrode_coords.append(mean_coords[mean_index])
            y.append(mean_y[mean_index])

        return (
            common_channels,
            electrode_lobes,
            np.array(y, dtype=np.int32),
            electrode_coords,
            np.stack(X_stim).astype(np.float32),
            np.stack(X_recording).astype(np.float32),
        )

    def process_metric_for_analysis(self, subject, electrodes_df, metric, labels=False):
        """Build stimulus/recording tensors for one response metric.

        Computes Euclidean distances between stimulation and recording
        electrode pairs, filters pairs closer than 13 mm, sorts by distance,
        and stacks per-electrode arrays into padded tensors.

        Args:
            subject: Subject identifier string.
            electrodes_df: DataFrame indexed by electrode name with columns
                ``x``, ``y``, ``z``, ``soz``, and ``Destrieux_label``.
            metric: Either ``"mean"`` or ``"std"``.
            labels: If ``True``, extract SOZ labels from ``electrodes_df``.
                Default is ``False``.

        Returns:
            A 6-tuple ``(channels, electrode_lobes, y, electrode_coords,
            X_stim, X_recording)`` where ``X_stim`` and ``X_recording`` are
            float32 arrays of shape ``(n_electrodes, n_trials, n_timesteps)``,
            or ``None`` if no valid stimulation events were found or all SOZ
            labels are negative.
        """
        response_df = self.response_df[self.response_df.subject == subject]
        response_df = response_df[response_df.metric == metric]

        try:
            # Calculate stimulation and recording coordinates
            stim_1_coords = np.array([
                                        [electrodes_df[electrodes_df.index == stimulated_electrode].x,
                                        electrodes_df[electrodes_df.index == stimulated_electrode].y,
                                        electrodes_df[electrodes_df.index == stimulated_electrode].z
                                      ] for stimulated_electrode in response_df.stim_1])
            stim_2_coords = np.array([
                                        [electrodes_df[electrodes_df.index == stimulated_electrode].x,
                                        electrodes_df[electrodes_df.index == stimulated_electrode].y,
                                        electrodes_df[electrodes_df.index == stimulated_electrode].z
                                      ] for stimulated_electrode in response_df.stim_2])
            stim_coords = (stim_1_coords + stim_2_coords) / 2

            recording_coords = np.array([
                                            [electrodes_df[electrodes_df.index == stimulated_electrode].x,
                                            electrodes_df[electrodes_df.index == stimulated_electrode].y,
                                            electrodes_df[electrodes_df.index == stimulated_electrode].z
                                         ] for stimulated_electrode in response_df.recording])

            # Calculate Euclidean distance
            distances = np.sqrt(np.sum((stim_coords - recording_coords) ** 2, axis=1))[:, 0]

            # Keep only distances greater than 13mm
            response_df = response_df[distances > 13]
            distances = distances[distances > 13]

            # Add distances to DataFrame
            response_df['distances'] = distances

            # Sort by distances - used for CNN method
            response_df = response_df.sort_values(by='distances', ascending=True)

        except:
            print("Error with subject", subject)

        # Get channels used for both stimulation and recording
        recording_stim_channels = sorted(
            set(response_df.recording.unique()).intersection(
                set(response_df.stim_2.unique()).union(set(response_df.stim_1.unique()))
            )
        )

        channels_recording_trials, channels_stim_trials = [], []
        
        if labels:
            channel_soz = []

        if len(recording_stim_channels) == 0:
            print(f"No stimulation events found for subject {subject}")
            return None
        
        electrode_coords = []
        electrode_lobes = []

        for channel in recording_stim_channels:

            # Current channel responses when other channels were stimulated
            channel_recording_trials = _with_distance_first(
                response_df[response_df.recording == channel].select_dtypes(include='number').copy()
            )

            # Other channel responses when current channel was stimulated
            channel_stim_trials = _with_distance_first(
                response_df[np.logical_or(response_df.stim_1 == channel,
                                          response_df.stim_2 == channel)].select_dtypes(include='number').copy()
            )

            # Add to corresponding lists, except for recording/stim channel names (i.e., only time series)
            channels_recording_trials.append(np.array(channel_recording_trials))
            channels_stim_trials.append(np.array(channel_stim_trials))

            if labels:
                # Add label for current channel
                channel_soz.append(electrodes_df[electrodes_df.index == channel].soz.iloc[0] == "yes")

            electrode_coords.append([electrodes_df[electrodes_df.index == channel].x.iloc[0],
                                     electrodes_df[electrodes_df.index == channel].y.iloc[0],
                                     electrodes_df[electrodes_df.index == channel].z.iloc[0]])
        
            electrode_lobe = electrodes_df[electrodes_df.index == channel].Destrieux_label.values[0]
            electrode_lobe = get_destrieux_lobe(electrode_lobe)
            electrode_lobe = region_to_index[electrode_lobe]
            electrode_lobes.append(electrode_lobe)
        
        # For recording channels
        max_recording_rows = max(array.shape[0] for array in channels_recording_trials)
        X_recording = pad_and_stack(channels_recording_trials, max_recording_rows).astype(np.float32)

        # For stim channels
        max_stim_rows = max(array.shape[0] for array in channels_stim_trials)
        X_stim = pad_and_stack(channels_stim_trials, max_stim_rows).astype(np.float32)

        if labels:
            y = np.array(channel_soz, dtype=np.int32)
            if y.sum() == 0:
                return None
            #np.save(f'../data/main/lobes_{subject}.npy', np.array(electrode_lobes))
            #np.save(f'../data/main/y_{subject}.npy', y)
            #np.save(f'../data/main/coords_{subject}.npy', np.array(electrode_coords))

        # Save as np arrays
        #np.save(f'../data/{metric}/X_stim_{subject}.npy', X_stim)
        #np.save(f'../data/{metric}/X_recording_{subject}.npy', X_recording)

        return recording_stim_channels, electrode_lobes, y, electrode_coords, X_stim, X_recording


class LocalizeSOZ(BaseTask):
    """Electrode-level seizure onset zone (SOZ) localization from CCEP ECoG.

    Orchestrates two preprocessing stages and produces one ML sample per
    candidate electrode per recording run:

    **Stage 1 — :class:`StimulationDataProcessor`**
        Reads a raw BrainVision EEG file and its associated BIDS metadata.
        The signal is bandpass-filtered (1–150 Hz), then segmented into
        epochs locked to each electrical stimulation event. Events
        contaminated by whole-recording artifacts or seizures are excluded.
        For each unique stimulation site (bipolar pair), epochs are averaged
        and their standard deviation is computed across trials, yielding a
        DataFrame of mean and std CCEP responses with one row per
        recording-channel per stimulation site.

    **Stage 2 — :class:`DatasetCreator`**
        Converts the response DataFrame into analysis-ready arrays. For each
        electrode that appears both as a stimulation site and as a recording
        channel, two tensors are built:

        - ``X_stim``: responses of *other* channels when *this* electrode was
          stimulated (divergent / outgoing connectivity).
        - ``X_recording``: responses of *this* channel when *other* electrodes
          were stimulated (convergent / incoming connectivity).

        Stimulation–recording pairs closer than 13 mm are discarded to
        reduce stimulation artifact contamination. Remaining trials are sorted
        by Euclidean distance and zero-padded so all electrodes within a
        batch share the same tensor shape. Lobe labels are looked up from the
        Destrieux atlas via :func:`get_destrieux_lobe`.

    Each returned sample corresponds to one candidate electrode. Multiple
    samples from the same patient share the same ``patient_id``; downstream
    train/test splits must group by ``patient_id`` to avoid leakage across
    electrodes from the same patient.

    Attributes:
        task_name: ``"LocalizeSOZ"``
        input_schema: Tensor inputs — ``X_stim`` and ``X_recording``
            (mean/std CCEP responses), ``electrode_lobes`` (Destrieux lobe
            index), and ``electrode_coords`` (MNI xyz coordinates).
        output_schema: Binary SOZ label (``1`` = in SOZ, ``0`` = not in SOZ).

    Examples:
        >>> from pyhealth.datasets import CCEPECoGDataset
        >>> from pyhealth.tasks import LocalizeSOZ
        >>> dataset = CCEPECoGDataset(root="/path/to/ds004080")
        >>> task = LocalizeSOZ()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "LocalizeSOZ"
    input_schema: Dict[str, str] = {
        "X_stim": "tensor",
        "X_recording": "tensor",
        "electrode_lobes": "tensor",
        "electrode_coords": "tensor",
    }
    output_schema: Dict[str, str] = {"soz": "binary"}

    def __call__(self, patient) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for split in ("ecog", "train", "eval"):
            try:
                events = patient.get_events(split)
            except (AttributeError, KeyError):
                continue

            for event in events:
                pid = patient.patient_id
                try:
                    header_file = event.header_file
                    events_file = event.events_file
                    channels_file = event.channels_file
                    electrodes_file = event.electrodes_file
                except AttributeError:
                    continue

                if not all(
                    [pid, header_file, events_file, channels_file, electrodes_file]
                ):
                    continue

                try:
                    eeg = mne.io.read_raw_brainvision(
                        header_file,
                        verbose=False,
                        preload=True,
                    )
                    events_df = pd.read_csv(events_file, sep="\t", index_col=0)
                    channels_df = pd.read_csv(channels_file, sep="\t", index_col=0)
                    electrodes_df = pd.read_csv(electrodes_file, sep="\t", index_col=0)

                    stim_processor = StimulationDataProcessor(tmin=0.009, tmax=1)
                    response_df = stim_processor.process_run_data(
                        eeg,
                        events_df,
                        channels_df,
                        pid,
                    )
                    if response_df is None:
                        continue

                    dataset_creator = DatasetCreator(response_df)
                    processed = dataset_creator.process_for_analysis(pid, electrodes_df)
                    if processed is None:
                        continue

                    (
                        electrode_channels,
                        electrode_lobes,
                        y,
                        electrode_coords,
                        X_stim,
                        X_recording,
                    ) = processed
                except (ValueError, KeyError, IndexError, FileNotFoundError, OSError):
                    continue

                for electrode_idx, channel in enumerate(electrode_channels):
                    electrode_id = f"{pid}-{event.session_id}-{event.run_id}-{channel}"
                    samples.append(
                        {
                            "patient_id": pid,
                            "visit_id": electrode_id,
                            "record_id": electrode_id,
                            "session_id": event.session_id,
                            "task_id": event.task_id,
                            "run_id": event.run_id,
                            "channel": channel,
                            "electrode_index": electrode_idx,
                            "header_file": header_file,
                            "events_file": events_file,
                            "channels_file": channels_file,
                            "electrodes_file": electrodes_file,
                            "soz": int(y[electrode_idx]),
                            "X_stim": X_stim[electrode_idx],
                            "X_recording": X_recording[electrode_idx],
                            "electrode_lobes": np.array(
                                [electrode_lobes[electrode_idx]], dtype=np.int64
                            ),
                            "electrode_coords": np.array(
                                electrode_coords[electrode_idx], dtype=np.float32
                            ),
                        }
                    )

        return samples
