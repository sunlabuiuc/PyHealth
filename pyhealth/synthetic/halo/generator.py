import collections
import datetime
import os
import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from tqdm import tqdm
import pickle
import pandas as pd
import torch
import torch.nn as nn
from pyhealth.data.data import Event, Patient, Visit

from pyhealth.synthetic.halo.processor import Processor

class Generator:
    """Synthetic Data generator module for HALO model. 

    This module conducts model inference on the HALO model, parametrized by the quantity of samples to generate,
    and in the case of conditional generation, the label of the samples to generate. It then converts the output to a human-readable format.

    This module generates longitudinal EHR records, and has the capability to translate the records from the format produced by the HALO self supervised
    model from multihot vector sequences of the model vocabulary into:
        - a human readable format
        - the PyHealth Patient format
        - a pandas.DataFrame format

    Args:
        model: The HALO model to use for inference.
        processor: The halo.Processor module which contains the training dataset vocabulary.
        batch_size: The batch size used for sample generation.
        save_path: Path to write the synthetic dataset.
        device: Device for model inference.
        handle_digital_time_gap: used for transforming temporal multi-hot into a human readable format. 
    """

    VISITS = 'visits'
    TIME = 'inter-visit_gap'
    LABEL = 'label'

    def __init__(
            self,
            model: nn.Module,
            processor: Processor,
            batch_size: int, # it is recommended to use the same batch size as that for training
            save_dir: str,
            save_name: str,
            device: str,
        ) -> None:
        
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_path = os.path.join(save_dir, f'{save_name}.pkl')
        self.device = device

    def generate_context(self, label_vector) -> List:
        """Generate context vector, and the probablility of the label occurrence in the dataset.

            Will be called in `generate_conditioned` to produce context vectors. 

        Returns:
           a list of context vectors with the label vector specified in the case of conditional generation
        """
        stoken = np.zeros((1, self.processor.total_vocab_size))
        stoken[0, self.processor.start_token_index] = 1
        
        if label_vector is None:
            return stoken # probability of label occurrence in dataset
        
        ltoken = np.zeros((1, self.processor.total_vocab_size))
        ltoken[0, self.processor.label_start_index: self.processor.label_end_index] = label_vector

        context = np.concatenate((stoken, ltoken), axis=0)
        context = context[np.newaxis, :, :]
        return context

    # When we have unconditioned generation, this method will be filled out to do more
    def get_contexts(self, contexts, batch_size: int, probability: float):
        idx = np.random.choice(len(contexts), batch_size, replace = True, p = probability) # random selection to generate contexts*batch_size seems inefficient
        return np.array([contexts[i] for i in idx])

    def sample_sequence(self, context, batch_size, random=True):
        """
        Starting with the context vector, grow the visits in order to generate a sequence of visits.
        
        Args:
            context: the batch of context vectors used to begin the generation. Should include start token and label.
            batch_size: quantity of context vectors to sample per batch.
            random: where we are randomly sampling to generate the next visit, or making predictions through rounding.
        """
        empty = torch.zeros((1, 1, self.processor.total_vocab_size), device=self.device, dtype=torch.float32).repeat(batch_size, 1, 1)
        prev = torch.tensor(context, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(self.processor.max_visits - (len(['start_token', 'label_token']))): # max num - (start vector, label vector)
                prev = self.model.sample(torch.cat((prev,empty), dim=1), random=random)

                # when we have generated end token break early
                if torch.sum(torch.sum(prev[:, :, self.processor.end_token_index], dim=1).bool().int(), dim=0).item() == batch_size:
                    break

        samples = prev.cpu().detach().numpy()
        return samples


    # handle conversion from HALO vector output to samples
    def convert_samples_to_ehr(self, samples) -> List[Dict]:
        """Convert multi hot visit representation used as internal representation of visits in the HALO model into terms in vocabulary.

        Uses the `time_handler_inverter` as well as vocuabulary information present in the `halo.Processor` to translate the
        output of the HALO model into human readable grammer. Namely, the indices of the visit vectors are mapped using the `halo.Processor.global_events`
        back to their unique string identifiers. Additionally, the `time_handler_inverter` function is applied to the inter-visit gap vector,
        allowing the translation of mutli-hot time vectors into an abitrary representation (such as python datetime). 

        Args:
            samples: the list of samples generated by `sample_sequence` which is a series of multi hot vectors representing synthetic patient visits. 

        Returns:
            the sample data translated into EHR outputs, which is a human readable format.
        """
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
                time_gap = visit[:self.processor.time_vector_length]
                time_gap = np.nonzero(time_gap)[0]
                time_as_index = time_gap[0] if len(time_gap) > 0 else np.random.randint(0, self.processor.time_vector_length)
                if j == self.processor.VISIT_INDEX:
                    time_gap_as_hours = np.random.uniform(self.processor.age_bins[time_as_index], self.processor.age_bins[time_as_index+1])
                else:
                    time_gap_as_hours = np.random.uniform(self.processor.visit_bins[time_as_index], self.processor.visit_bins[time_as_index+1])
                sample_time_gaps.append(time_gap_as_hours)

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
            self,
            samples: List,
            event_handlers: Dict[str, Callable],
            base_time: datetime,
            label_mapping: Dict[Tuple, dict] = None
        ) -> List[Patient]:
        """Convert a list of patient samples in the EHR representation into a list of `pyhealth.data.Patient` objects.

        For all patients provided in the samples argument, convert the patient into a `pyhealth.data.Patient`, 
        convert each visit into a `pyhealth.data.Visit`, each event into a `pyhealth.data.Event` using the provided handlers and mappings.

        Args:
            samples: List of ehr patient samples (output from convert_samples_to_ehr).
            event_handlers: Dictionary of Callable functions, which convert an word from the global vocabulary 
                (values in by halo.Processor.global_events dict) into arguments for instantiating a `pyhealth.data.Event` object.
            handle_inter_visit_time: a Callable to handle converting inter-visit gap values into another, arbitrary representation, such as python datetime.
            base_time: the time from which to compute all patient age and time offsets. A sensible default in many cases is datetime.now().
            label_mapping: A mapping to convert synthetic sample label vectors into a keyword, value pair of patient features.

        Returns:
            A list of `pyhealth.data.Patient` objects. 
            
        Example for Event Handlers:

        Where event handlers in processor are:
        ```
            def handle_diagnosis(event: Event):
                # Convert granular ICD codes to more broad ones (ie 428.3 --> 428)
                split_code = event.code.split('.')
                assert len(split_code) <= 2
                return split_code[0]

            def handle_lab(event: Event):
                # turn a lab event into a discrete value
                def digitize_lab_result(x):
                    bins = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                    return np.digitize(x, bins)

                binned_lab_result = digitize_lab_result(event.attr_dict['lab_result'])
                unit = event.attr_dict['lab_measure_name_system']
                return f"{event.code}_{unit}_{binned_lab_result}"
            
            event_handlers = {
                'diagnosis' =  handle_diagnosis,
                'lab' =  handle_lab
            }
        ```

        The appropriate event handlers in the generator, which invert the event handlers in the processor are:
        ```
            def lab_handler(lab_event):
                # Lab events are produced using the string f"{event.code}_{unit}_{binned_lab_result}
                lab_event_data = lab_event.split('_')

                # output a dict which will serve as Args for instantiating a pyhealth.data.Event object.
                return {
                    'code': lab_event_data[0],
                    'lab_measure_name_system': lab_event_data[1],
                    'lab_result': reverse_lab_digitization(int(lab_event_data[2])),
                    'vocabulary': 'eICU_LABNAME_SYNTHETIC',
                    'table': 'lab'
                }

            def diagnosis_handler(diagnosis_event):
                return {
                    'code': diagnosis_event,
                    'vocabulary': 'ICD9CM',
                    'table': 'diagnosis'
                }

            event_handlers = {
                'diagnosis': diagnosis_handler,
                'lab': lab_handler
            }
        ```
        """
        patients = []
        for patient_id, sample in tqdm(enumerate(samples), total=len(samples), desc="Converting samples to PyHealth patients"):
            patient_id = str(patient_id)

            patient_label = label_mapping[sample['label']] if label_mapping else sample['label']

            # get timedelta for all visits
            time_gaps = [t * datetime.timedelta(hours=1) for t in sample['inter-visit_gap']]
        
            # get the patient birth date time
            total_time = base_time
            for time_gap in time_gaps:
                total_time = total_time - time_gap

            patient = Patient(
                patient_id=patient_id, 
                birth_datetime=total_time,
                **patient_label
            )

            time_of_last_visit = patient.birth_datetime
            for visit_id in range(0, len(sample['visits'])):
                unique_visit_id = f"{patient_id}_{visit_id}"
                visit, time_since_previous_visit = sample['visits'][visit_id], time_gaps[visit_id]
                
                try:
                    # todays visit is time of last visit + time since last visit
                    visit_time = time_of_last_visit + time_since_previous_visit
                except OverflowError:
                    visit_time = visit_time

                pyhealth_visit = Visit(
                    visit_id=unique_visit_id,
                    patient_id=patient_id,
                    encounter_time=visit_time,
                    discharge_status='Expired' if visit_id == len(sample['visits']) - 1 and patient.death_datetime is not None else 'Alive'
                )
                
                # {'code': '573.4', 'table': 'diagnosis', 'vocabulary': 'ICD9CM', 'visit_id': '781578', 'patient_id': '006-229574+599399', 'timestamp': Timestamp('2014-01-07 00:39:00'), 'attr_dict': {'diagnosisString': 'gastrointestinal|hepatic disease|hepatic infarction'}}
                # code, table, vocabulary

                for event_table, event_data in visit:
                    assert event_table in event_handlers, f"No event handler for {event_table}"
                    if event_table in self.processor.compute_histograms:
                        event_data = event_handlers[event_table](event_data, self.processor)
                    else:
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

            patients.append(patient)

        return patients
    
    def pyhealth_to_tables(patient_records: List[Patient]):
        """Converts a List of PyHealth patients (pyhealth.data.Patient) of patient records into pandas dataframes.

        The generated dataframes represent the Visit level data, where each event type within the visit has its own table, as well as Patient level data, 
        which is represented in the patient table.

        The generated dataframes are the based on the union of all table types present in the `pyhealth.data.Visit` objects, in addition to the patient table. 

        Args:
            patient_records: List of PyHealth patients.

        Returns: 
            Dictionary of tables, including the patient table, as well as the visit level table. 
            The key of the dict is the table name, and the value is the pd.DataFrame object representing the table. 
        """
        # patient table
        visit_level_patient_columns = ['patient_id', 'visit_id', 'discharge_status', 'discharge_time', 'encounter_time']
        patient_table_columns=['patient_id', 'birth_datetime', 'death_datetime', 'gender', 'ethnicity']
        patient_table_list = []

        # event tables
        event_columns = ['patient_id', 'visit_id', 'code', 'timestamp']
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

    def generate_conditioned(self, labels: List[Tuple[any, int]]) -> List[Dict]:
        """Generate synthetic data generation on conditioned labels. Saves the conditioned dataset at the `halo.Generator.save_path` provided during initialization.
        
        Args:
            labels: a List of Tuples, where the 0th element is the label vector, and the 1st element is the quentity of synthetic samples of the label to generate.

        Returns:
            The synthetic ehr dataset of the samples corresponding to the quantity requested. 
            Writes the samples to the `halo.Generator.save_path` provided during the instantiation of the class.

        Example:
        
        ```
        generator = Generator(
            model=model,
            processor=processor,
            batch_size=batch_size,
            device=device,
            save_path=basedir,
            save_name="synthetic_data" # save at `synthetic_data.pkl`
        )

        labels = [((1), 40000), ((0), 40000)] # 40k samples of each label
        synthetic_dataset = generator.generate_conditioned(labels)
        ```
        """
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
                    random=True
                )
                
                batch_synthetic_ehrs = self.convert_samples_to_ehr(batch_synthetic_ehrs)
                synthetic_ehr_dataset += batch_synthetic_ehrs
        print("Saving synthetic ehr dataset at:", self.save_path)
        pickle.dump(synthetic_ehr_dataset, open(self.save_path, "wb"))
        return synthetic_ehr_dataset