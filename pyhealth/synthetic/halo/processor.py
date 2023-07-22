from datetime import timedelta

import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import pandas

from tqdm import tqdm

from pyhealth.data import Event 
from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset

"""
produce a discrete value given a continuous one
strict_range: when a value is outside of the range, do you throw error, or do you just make the value the range maximum? Similar for minimum. 
"""
class Discretizer:
    
    def __init__(self, strategy: str, strict_range: bool = False, **kwargs):
        if strategy == 'uniform':
            self.start = kwargs['start']
            self.end = kwargs['end']
            step = kwargs['step']
            
            intervals = [self.start]
            while (intervals[-1] < self.end):
                # the last interval may be shorter than all the others, but never exceed range
                intervals.append(min(intervals[-1] + step, self.end))
            
            self.clip_overflow = strict_range
            self.intervals = intervals
            self.to_discrete = self.uniform
        else: 
            # not implemented or discretization strategy is not a correct value
            raise Exception('Invalid discretization strategy provided')
        
    # currently unimplemented
    # todo: bdanek, would be good to write this algorithm without any bugs... & with unit tests    
    def uniform(self, value: float):
        if self.strict_range:
            assert value <= self.end or value >= self.start, f'Strict range enforced; value: {value} not in range [{self.start}, {self.end}].'
        
        return value

class Processor():
    """
    Class used to process a PyHealth dataset so that it can be used in HALO. 

    Args:
        dataset: Dataset to process.
        use_tables: Tables to use during processing. If none, all available tables in the dataset are used. 
        
        event_handlers: A dictionary of handlers for unpacking, or accessing fields of a pyhealth.data.Event. 
            The dict key should be a table name, value is a Callable which must accept a pyhealth.data.Event to unpack.
        continuous_value_handlers: A dictionary of handlers for converting an event from a continuous value to a bucketed/categorical one. 
            This handler is applied after the event handler. Dict key is table name, and should return an integer representing which bucket the value falls in.
        continuous_value_handlers_inverter: An optional dictionary of handlers for inverting the operation performed in `continuous_value_handlers`. 
            The dict key is a table name, and the value is a Callable which accepts a vector (list of integers)
        time_handler: 
            A function which converts a timedelta into a multihot vector representation. 
        time_hanlder_inverter: 
            An optional function which converts the multihot time vector represention into a human readbile value. 
        time_vector_length: 
            The integer representing the length of the multihot time vector produced by `time_handler`
        max_visits: 
            The maximum visits to use for modeling. If not provided, the maximum number of visits present in the source dataset is used.
        label_fn: 
            A function which accepts the keyword argument `patient_data: pyhealth.data.Patient` and produces a vector representation of the patient label.
        label_vector_len: 
            The length of a patient label vector.
        invert_label: 
            An optional function for converting the label vector into a human readable patient label

    """
    
    # visit dimension (dim 1)
    SPECIAL_VOCAB = ('start_code', 'last_visit_code', 'pad_code') 
    START_INDEX = 0
    LABEL_INDEX = 1
    VISIT_INDEX = 2

    # code dimension (dim 2)
    SPECIAL_VISITS = ('start_visit', 'label_visit')

    # the key for the inter_visit_gap handler
    TEMPORAL_INTER_VISIT_GAP = 'inter_visit_gap'
        
    def __init__(
        self,
        dataset: BaseEHRDataset,
        use_tables: List[str],
        
        # allows unpacking/handling patient records into events
        event_handlers: Dict[str, Callable[[Type[Event]], Any]] = {},

        # used to handle continuous values
        continuous_value_handlers: Dict[str, Callable[..., int]] = {},
        continuous_value_handlers_inverter: Dict[str, Callable[[List[int]], Any]] = {},

        # used to discretize time
        time_handler: Callable[[Type[timedelta]], Any] = None,
        time_hanlder_inverter: Callable[[List[int]], Any] = None,
        time_vector_length: int = -1,
        
        max_visits: Union[None, int] = None,
        label_fn: Callable[..., List[int]] = None, 
        label_vector_len: int = -1,
        invert_label: Callable[..., Any] = None,
    ) -> None:
        
        self.dataset = dataset
        
        # whitelisted tables
        self.valid_dataset_tables = use_tables 

        # handle processing of event types
        self.event_handlers = event_handlers 
        
        # generate a HALO label based on a patient record
        assert label_fn != None, "Define the label_fn."
        assert label_vector_len >= 0, "Nonnegative vector_len required. May be due to user error, or value is not defined."
        self.label_fn = label_fn
        self.label_vector_len = label_vector_len
        self.invert_label = invert_label

        self.continuous_value_handlers = continuous_value_handlers
        self.continuous_value_handlers_inverter = continuous_value_handlers_inverter

        assert time_handler != None, "Defining time_handler is not optional. This field converts time values to a discrete one hot/multi hot vector representation."
        self.time_handler = time_handler

        # optional
        self.time_hanlder_inverter = time_hanlder_inverter
        
        assert time_vector_length != None, "Defining time_vector_length is not optional. This field is equivalent to the number of buckets required to discretie time."
        self.time_vector_length = time_vector_length

        self.max_visits = max_visits

        # init the indeces & dynamically computed utility variables used in HALO training later
        self.set_indeces()


    def set_indeces(self) -> None:
        # set aggregate indeces
        self.global_events: Dict = {}
        self.aggregate_event_indeces()

        # assert the processor works as expected
        assert len(self.global_events) % 2 == 0, "Event index processor not bijective"

        # bidirectional mappings
        self.num_global_events = self.time_vector_length + len(self.global_events) // 2

        # define the tokens in the event dimension (visit dimension already specified)
        self.label_start_index = self.num_global_events
        self.label_end_index = self.num_global_events + self.label_vector_len
        self.start_token_index = self.num_global_events + self.label_vector_len
        self.end_token_index = self.num_global_events + self.label_vector_len + 1
        self.pad_token_index = self.num_global_events + self.label_vector_len + 2

        # parameters for generating batch vectors
        self.total_vocab_size = self.num_global_events + self.label_vector_len + len(self.SPECIAL_VOCAB)
        self.total_visit_size = len(self.SPECIAL_VISITS) + self.max_visits

    """
    its necessary to aggregate global event data, prior to transforming the dataset
    """
    def aggregate_event_indeces(self) -> None:

        # two way mapping from global identifier to index & vice-versa
        # possible since index <> global identifier is bijective
        # type: ((table_name: str, event_value: any): index) or (index: (table_name: str, event_value: any))
        max_visits: int = 0
        min_birth_datetime = pandas.Timestamp.now()
        
        for pdata in tqdm(list(self.dataset), desc="HALOAggregator generating indeces"):
            
            max_visits = max(max_visits, len(pdata.visits))
            if pdata.birth_datetime != None:
                min_birth_datetime = min(min_birth_datetime, pdata.birth_datetime)

            # compute global event
            for vid, vdata in pdata.visits.items():

                for table in vdata.available_tables:

                    # valid_tables == None signals we want to use all tables
                    # otherwise, omit any table not in the whitelist
                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    
                    for te_raw in table_events_raw:
                        
                        te = event_handler(te_raw) if event_handler else te_raw.code

                        if table in self.continuous_value_handlers:
                            te = self.continuous_value_handlers[table](te)

                        global_event = (table, te)
                    
                        if global_event not in self.global_events:
                            # keys 0 - self.time_vector_length are reserved for inter-visit information
                            index_of_global_event = self.time_vector_length + (len(self.global_events) // 2)
                            self.global_events[global_event] = index_of_global_event
                            self.global_events[index_of_global_event] = global_event

        
        # if the user does not provide, infer from dataset
        if self.max_visits == None:
            self.max_visits = max_visits
        
        self.min_birth_datetime = min_birth_datetime
    """
    similar to dataset.set_task(...)
    - produce a sampleEHRDataset
    """  
    def process_batch(self, batch) -> Tuple:
        batch_size = len(batch)
        
        # dim 0: batch
        # dim 1: visit vectors
        # dim 2: concat(event multihot, label onehot, metadata)
        sample_multi_hot = np.zeros((batch_size, self.total_visit_size, self.total_vocab_size)) # patient data the model reads
        sample_mask = np.zeros((batch_size, self.total_visit_size, 1)) # visits that are unlabeled
        
        for pidx, pdata in enumerate(batch):

            previous_time = pdata.birth_datetime if pdata.birth_datetime != None else self.min_birth_datetime
            # build multihot vector for patient events
            for visit_index, vid,  in enumerate(pdata.visits):
                
                vdata = pdata.visits[vid]

                # set temporal attributes
                current_time = vdata.encounter_time
                time_since_last_visit = current_time - previous_time                
                
                # vector representation of the gap between last visit and current one
                inter_visit_gap_vector = self.time_handler(time_since_last_visit)
                sample_multi_hot[pidx, self.VISIT_INDEX, :self.time_vector_length] = inter_visit_gap_vector

                # the next timedelta is previous current visit - discharge of previous visit
                previous_time = vdata.discharge_time

                sample_mask[pidx, self.VISIT_INDEX + visit_index] = 1
                
                for table in vdata.available_tables:

                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    continuous_value_handler = self.continuous_value_handlers[table] if table in self.continuous_value_handlers else None

                    for te_raw in table_events_raw:
                        
                        te = event_handler(te_raw) if event_handler else te_raw.code
                        
                        if continuous_value_handler:
                            te = continuous_value_handler(te)

                        global_event = (table, te)
                        event_as_index = self.global_events[global_event]
                        
                        # set table events
                        sample_multi_hot[pidx, self.VISIT_INDEX + visit_index, event_as_index] = 1            
            
            # set patient label
            global_label_vector = self.label_fn(patient_data=pdata)
            sample_multi_hot[pidx, self.LABEL_INDEX, self.num_global_events: self.num_global_events + self.label_vector_len] = global_label_vector
            
            # set the end token
            sample_multi_hot[pidx, self.VISIT_INDEX + (len(pdata.visits) - 1), self.end_token_index] = 1

            # set the remainder of the visits to pads if needed
            sample_multi_hot[pidx, (self.VISIT_INDEX + (len(pdata.visits) - 1)) + 1:, self.pad_token_index] = 1
            
        # set the start token
        sample_multi_hot[:, self.START_INDEX, self.start_token_index] = 1

        # set the mask to include the labels
        sample_mask[:, self.LABEL_INDEX] = 1
        
        # "shift the mask to match the shifted labels & predictions the model will return"
        sample_mask = sample_mask[:, 1:, :]
            
        res = (sample_multi_hot, sample_mask)
        
        return res
    
    def get_batch(self, data_subset: BaseEHRDataset, batch_size: int = 16,):

        batch_size = min(len(data_subset), batch_size)

        batch_offset = 0
        while (batch_offset + batch_size <= len(data_subset)):
            
            batch = data_subset[batch_offset: batch_offset + batch_size]
            batch_offset += batch_size # prepare for next iteration
            
            yield self.process_batch(batch)