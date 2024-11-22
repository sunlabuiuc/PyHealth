from dataclasses import dataclass, field
from typing import Dict
from datetime import timedelta
from pyhealth.tasks.task_template import TaskTemplate
import numpy as np


@dataclass(frozen=True)
class Mortality30DaysMIMIC4(TaskTemplate):
    task_name: str = "Mortality30Days"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"diagnoses": "sequence", "procedures": "sequence"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"mortality": "label"})

    def __call__(self, patient):
        death_datetime = patient.attr_dict["death_datetime"]
        diagnoses = patient.get_events_by_type("diagnoses_icd")
        procedures = patient.get_events_by_type("procedures_icd")
        mortality = 0
        if death_datetime is not None:
            mortality = 1
            # remove events 30 days before death
            diagnoses = [
                diag
                for diag in diagnoses
                if diag.timestamp <= death_datetime - timedelta(days=30)
            ]
            procedures = [
                proc
                for proc in procedures
                if proc.timestamp <= death_datetime - timedelta(days=30)
            ]
        diagnoses = [diag.attr_dict["code"] for diag in diagnoses]
        procedures = [proc.attr_dict["code"] for proc in procedures]

        if len(diagnoses) * len(procedures) == 0:
            return []

        samples = [
            {
                "patient_id": patient.patient_id,
                "diagnoses": diagnoses,
                "procedures": procedures,
                "mortality": mortality,
            }
        ]
        return samples

class Discretizer:
    '''
    Discretizer class for MISTS feature extraction (https://arxiv.org/pdf/2210.12156)
    Code modified from: https://github.com/XZhang97666/MultimodalMIMIC/blob/main/preprocessing.py by Hang Yu
    '''
    def __init__(self, selected_channel_ids, normal_values, timestep=0.8, impute_strategy='zero', start_time='relative'):
        '''
        Args:
            timestep: interval span (hours)
            TODO: other arguments documentation
        '''
        self._selected_channel_ids = selected_channel_ids
        self._timestep = timestep
        self._start_time = start_time
        self._impute_strategy = impute_strategy
        self._normal_values = normal_values

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, T, channel, timespan=None):
        '''
        Args:
            X: list of valuenum
            T: list of timestamp
            channel: the code of lab item
            timespan: the timespan of the data we use
        '''
        eps = 1e-6

        t_ts, x_ts = T, X
        for i in range(len(t_ts) - 1):
            assert t_ts[i] < t_ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = t_ts[0]
        elif self._start_time == 'zero':
            raise NotImplementedError("start_time 'zero' not implemented yet")
        else:
            raise ValueError("start_time is invalid")

        if timespan is None:
            max_hours = max(t_ts)
        else:
            max_hours = timespan

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        data = np.zeros(shape=(N_bins,), dtype=float)
        mask = np.zeros(shape=(N_bins,), dtype=int)
        original_value = ["" for i in range(N_bins)]
        total_data = 0
        unused_data = 0
                
        for i in range(len(t_ts)):
            t = t_ts[i]
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            total_data += 1
            if mask[bin_id] == 1:
                unused_data += 1
            mask[bin_id] = 1
            data[bin_id] = x_ts[i]
            original_value[bin_id] = x_ts[i]

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = []
            for bin_id in range(N_bins):
                if mask[bin_id] == 1:
                    prev_values.append(original_value[bin_id])
                    continue
                if self._impute_strategy == 'normal_value':
                    imputed_value = self._normal_values[channel]
                if self._impute_strategy == 'previous':
                    if len(prev_values) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[-1]
                data[bin_id] = imputed_value
                # write(data, bin_id, channel, imputed_value)

        if self._impute_strategy == 'next':
            prev_values = []
            for bin_id in range(N_bins-1, -1, -1):
                if mask[bin_id] == 1:
                    prev_values.append(original_value[bin_id])
                    continue
                if len(prev_values) == 0:
                    imputed_value = self._normal_values[channel]
                else:
                    imputed_value = prev_values[-1]
                data[bin_id] = imputed_value
                # write(data, bin_id, channel, imputed_value)

        # empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        # self._done_count += 1
        # self._empty_bins_sum += empty_bins / (N_bins + eps)
        # self._unused_data_sum += unused_data / (total_data + eps)

        return (data.tolist(), mask.tolist())
    
@dataclass(frozen=True)
class MIMIC3_48_IHM(TaskTemplate):
    task_name: str = "48_IHM"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"discretized_features": "sequence"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"mortality": "label"})
    __name__: str = task_name
    selected_labitem_ids = {'51221':0, '50971':1, '50983':2, '50912':3, '50902':4} # TODO: use LABEVENTS.csv group statistics to optimize this
    normal_values = {'51221':42.4, '50971':4.3, '50983':139, '50912':0.7, '50902':97}
    discretizer = Discretizer(selected_labitem_ids, normal_values)

    def __call__(self, patient):
        samples = []

        # we will drop the last visit
        for i in range(len(patient)):
            visit: Visit = patient[i]

            assert visit.discharge_status in [0, 1], f"Unexpected discharge status for Visit {visit}"
            mortality_label = int(visit.discharge_status)

            # exclude the event happened after 48 hrs window on admission of the hospital
            labevents = visit.get_event_list(table="LABEVENTS")
            end_timestamp = visit.encounter_time + timedelta(days=2)
            # exclude: visits without lab events
            if len(labevents) == 0 or labevents[0].timestamp > end_timestamp or labevents[-1].timestamp < end_timestamp:
                # if no event happens in this visit within the first 48 hrs or this visit is shorter than 48 hrs (2 days), we skip this visit
                continue
            
            X_ts = [[] for _ in range(len(self.selected_labitem_ids))]
            T_ts = [[] for _ in range(len(self.selected_labitem_ids))]
            for event in labevents:
                if event.timestamp < visit.encounter_time:
                    # TODO: discuss with Zhenbang if this is desired, skip the lab events before the hospital admission
                    continue
                if event.timestamp > visit.encounter_time + timedelta(days=2):
                    break 
                if event.code in self.selected_labitem_ids:
                    l = self.selected_labitem_ids[event.code]
                    X_ts[l].append(event.attr_dict['valuenum'])
                    T_ts[l].append((event.timestamp - visit.encounter_time).total_seconds() / 3600)

            discretized_X, discretized_mask = [], []
            for code in self.selected_labitem_ids:
                l = self.selected_labitem_ids[code]
                x_ts, mask_ts = self.discretizer.transform(X_ts[l], T_ts[l], code, timespan=48) # TODO: add normalizer later
                discretized_X.append(x_ts)
                discretized_mask.append(mask_ts) # not used so far
            
            # TODO: should also exclude visit with age < 18
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": visit.visit_id,
                    "discretized_feature": discretized_X,
                    "x_ts": X_ts,
                    "t_ts": T_ts,
                    "mortality": mortality_label,
                }
            )
        # no cohort selection
        return samples