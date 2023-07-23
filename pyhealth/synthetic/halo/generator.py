import collections
import datetime
import numpy as np
from typing import Callable, Dict, List, Tuple
from tqdm import tqdm
import pickle
import pandas as pd
import torch
import torch.nn as nn
from pyhealth.data.data import Event, Patient, Visit

from pyhealth.synthetic.halo.processor import Processor

class Generator:

    VISITS = 'visits'
    TIME = 'inter-visit_gap'
    LABEL = 'label'

    def __init__(
            self,
            model: nn.Module,
            processor: Processor,
            batch_size: int, # it is recommended to use the same batch size as that for training
            save_path: str,
            device: str,
        ) -> None:
        
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.save_path = f'{save_path}.pkl'
        self.device = device

    # generate context vector, and the probablility of the label occurrence in the dataset
    def generate_context(self, label_vector) -> List:
        stoken = np.zeros((1, self.processor.total_vocab_size))
        stoken[0, self.processor.start_token_index] = 1
        
        if label_vector is None:
            return stoken # probability of label occurrence in dataset
        
        ltoken = np.zeros((1, self.processor.total_vocab_size))
        ltoken[0, self.processor.label_start_index: self.processor.label_end_index] = label_vector

        context = np.concatenate((stoken, ltoken), axis=0)
        context = context[:, np.newaxis, :]
        return context

    # get batches of context vectors with a probability
    def get_contexts(self, contexts, batch_size: int, probability: float):
        idx = np.random.choice(len(contexts), batch_size, replace = True, p = probability) # random selection to generate contexts*batch_size seems inefficient
        return np.array([contexts[i] for i in idx])

    def sample_sequence(self, context, batch_size, sample=True, visit_type=-1):
        empty = torch.zeros((1, 1, self.processor.total_vocab_size), device=self.device, dtype=torch.float32).repeat(batch_size, 1, 1)
        prev = torch.tensor(context, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(self.processor.max_visits - (len(['start_token', 'label_token']))): # visits - (start vector, label vector); iterate # of ti

                prev = self.model.sample(torch.cat((prev,empty), dim=1), sample)

                if torch.sum(torch.sum(prev[:, :, self.processor.end_token_index], dim=1).bool().int(), dim=0).item() == batch_size: # why do we do this?
                    break

        samples = prev.cpu().detach().numpy()

        return samples


    # handle conversion from HALO vector output to samples
    def convert_samples_to_ehr(self, samples) -> List[Dict]:
        ehr_outputs = []
        for i in range(len(samples)):
            sample_as_ehr = []
            sample_time_gaps = []
            sample = samples[i]

            # labels need to be hashable, so we convert them back to tuple representation
            labels_output = tuple(sample[self.processor.LABEL_INDEX][self.processor.label_start_index: self.processor.label_end_index])

            for j in range(self.processor.VISIT_INDEX, len(sample)):
                
                visit = sample[j]

                # handle inter-visit gaps
                visit_time = visit[:self.processor.time_vector_length]
                convert_to_time = self.processor.time_hanlder_inverter
                time_gap = convert_to_time(visit_time) if convert_to_time != None else visit_time
                sample_time_gaps.append(time_gap)

                # handle visit event codes
                visit_events = visit[self.processor.time_vector_length: self.processor.num_global_events]
                visit_code_indices = np.nonzero(visit_events)[0]
                visit_ehr_codes = [self.processor.global_events[self.processor.time_vector_length + index] for index in visit_code_indices]
                sample_as_ehr.append(visit_ehr_codes)

                end = bool(sample[j, self.processor.end_token_index])
                if end: break
            
            ehr_outputs.append({self.VISITS: sample_as_ehr, self.TIME: sample_time_gaps, self.LABEL: labels_output})

        return ehr_outputs
    
    def convert_ehr_to_pyhealth(
            samples: List,
            event_handlers: Dict[str, Callable],
            handle_inter_visit_time: Callable,
            label_mapping: Dict[Tuple, str],
            base_time: datetime
        ) -> List[Patient]:
        patients = []
        for patient_id, sample in enumerate(samples):
            patient_id = str(patient_id)

            patient_label = label_mapping[sample['label']]

            # get timedelta for all visits
            processed_time_gaps = [handle_inter_visit_time(time_gap) for time_gap in sample['inter-visit_gap']]
        
            # get the patient birth date time
            total_time = base_time
            for time_gap in processed_time_gaps:
                total_time = total_time - time_gap

            patient = Patient(
                patient_id=patient_id, 
                birth_datetime=total_time,
                patient_label=patient_label, # replace with semantic meaning for the label
            )

            time_of_last_visit = patient.birth_datetime
            for visit_id in range(0, len(sample['visits'])):

                unique_visit_id = f"{patient_id}_{visit_id}"
                visit, time_gap = sample['visits'][visit_id], sample['inter-visit_gap'][visit_id]
                
                time_since_previous_visit = processed_time_gaps[visit_id]
                try:
                    # todays visit is time of last visit + time since last visit
                    visit_time = time_of_last_visit + time_since_previous_visit
                except OverflowError:
                    visit_time = visit_time

                pyhealth_visit = Visit(
                    visit_id=unique_visit_id,
                    patient_id=patient_id,
                    encounter_time=visit_time
                )

                for event_table, event_data in visit:
                    assert event_table in event_handlers, f"No event handler for {event_table}"
                    event_data = event_handlers[event_table](event_data)
                    event = Event(
                        visit_id=unique_visit_id,
                        patient_id=patient_id,
                        timestamp=visit_time, # currently there is time representation within the visit
                        **event_data
                    )

                    pyhealth_visit.add_event(event)
                patient.add_visit(pyhealth_visit)
                
                time_of_last_visit = visit_time
                visit_id += 1

            patients.append(patient)

        return patients
    
    def pyhealth_to_tables(patient_records):
        # patient table
        visit_level_patient_columns = ['visit_id', 'discharge_status', 'discharge_time', 'encounter_time']
        patient_table_columns=['birth_datetime', 'death_datetime', 'gender', 'ethnicity']
        patient_table_list = []

        # event tables
        event_columns = ['visit_id', 'code', 'timestamp']
        event_columns_per_table = {}

        event_tables = collections.defaultdict(list)

        for patient in patient_records:
            overall_patient_row = [] # represents a row in the patient table
            patient_level_table_data = [] # data which is present in the Patient object
            # get primary table attributes
            for attr in patient_table_columns:
                if hasattr(patient, attr):
                    patient_level_table_data.append(getattr(patient, attr))
            
            # get all additional attributes
            for attr_name, attr_val in patient.attr_dict.items():
                if attr_name not in patient_table_columns:
                    patient_table_columns.append(attr_name)
                patient_level_table_data.append(attr_val)

            
            for visit_id, visit in patient.visits.items():
                # get data which is present in the Visit object for a patient
                visit_data = []
                for attr in visit_level_patient_columns:
                    getattr(visit, attr)
                overall_patient_row.append(visit_data + patient_level_table_data)

                # get all events & place them in their respective tables        
                for event_type, events in visit.event_list_dict.items():
                    for event in events:
                        event_table_row = []
                        
                        for attr in event_columns:
                            if hasattr(event, attr):
                                event_table_row.append(getattr(event, attr))

                        for attr_name, value in event.attr_dict.items():
                            if event_type not in event_columns_per_table:
                                event_columns_per_table[event_type] = event_columns

                            # add a new entry to the table-specific columns for events
                            if attr_name not in event_columns_per_table[event_type]:
                                event_columns_per_table[event_type] = event_columns_per_table[event_type] + [attr_name]
                            
                            event_table_row.append(value)
                        
                        event_tables[event_type].append(event_table_row)

            patient_table_list.extend(overall_patient_row)

        tables = {'patients': pd.DataFrame(patient_table_list, columns=patient_table_columns)}
        for table_type, table_rows in event_tables.items():
            columns = event_columns_per_table[table_type] if table_type in event_columns_per_table else event_columns
            tables[table_type] = pd.DataFrame(event_tables[table_type], columns=columns)

        return tables

    def generate_conditioned(self, labels: List[Tuple[any, int]]):
        
        synthetic_ehr_dataset = []
        for (label, count_per_label) in tqdm(labels, desc=f"Generating samples for labels"):
            context_vectors = self.generate_context(label)
            for i in tqdm(range(0, count_per_label, self.batch_size), leave=False):
                amount_remaining = count_per_label - i
                bs = min(amount_remaining, self.batch_size)
                context = self.get_contexts(context_vectors, bs, probability=None)
                
                batch_synthetic_ehrs = self.sample_sequence(
                    context=context, 
                    batch_size=bs, 
                    sample=True
                )
                
                batch_synthetic_ehrs = self.convert_samples_to_ehr(batch_synthetic_ehrs)
                synthetic_ehr_dataset += batch_synthetic_ehrs
        print("Saving synthetic ehr dataset at:", self.save_path)
        pickle.dump(synthetic_ehr_dataset, open(self.save_path, "wb"))
        return synthetic_ehr_dataset