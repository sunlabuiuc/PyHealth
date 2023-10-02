import collections
import logging
import os
import pandas

import numpy as np
from typing import Callable, Dict, List, Tuple, Type, Union
import pandas

from tqdm import tqdm

from pyhealth.data import Event 
from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH, hash_str
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)

class Processor():
    """
    Class used to process a PyHealth dataset for HALO usage. The core features this class supports are 
        1. converting the `pyhealth.data.Events` into a vocabulary
        2. facilitating the mapping between `pyhealth.data.Events` and vocuabulary terms
        3. mapping `pyhealth.data.Visit` metadata, and vocuabulary terms to a multi-hot representation vector the HALO method uses
        4. provide get batch function for halo trainer

    Instantiating this class wil trigger `aggregate_indices` and `set_indices` functions, which computes the global vocabulary needed to represent the provided dataset. 

    Args:
        dataset: Dataset to process.
        use_tables: Tables to use during processing. If none, all available tables in the dataset are used. 
        
        event_handlers: A dictionary of handlers for unpacking, or accessing fields of a pyhealth.data.Event. 
            The dict key should be a table name, value is a Callable which must accept a pyhealth.data.Event to unpack.
        continuous_value_handlers: A dictionary of handlers for converting an event from a continuous value to a bucketed/categorical one. 
            This handler is applied after the event handler. Dict key is table name, and should return an integer representing which bucket the value falls in.
        
        time_handler: 
            A function which converts a timedelta into a multihot vector representation. 
        time_vector_length: 
            The integer representing the length of the multihot time vector produced by `time_handler`
        max_visits: 
            The maximum visits to use for modeling. If not provided, the maximum number of visits present in the source dataset is used.
        label_fn: 
            A function which accepts the keyword argument `patient_data: pyhealth.data.Patient` and produces a vector representation of the patient label.
        label_vector_len: 
            The length of a patient label vector.
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

    EPSILON = 0.05

    # alias for a simple hashable type used throughout the processor
    _Hashable = Union[str, Tuple]

    # we need to pickle this instances of this class, and it is forbidden to pickle lambda functions,
    # so we define this utility function
    # effectively should replace: collections.defaultdict(lambda: collections.defaultdict(list)) for level == 2
    def _list_defaultdict(self):
        return collections.defaultdict(list)
    
    def new_nested_defaultdict(self):
        return collections.defaultdict(self._list_defaultdict)

    def __init__(
        self,
        dataset: BaseEHRDataset,
        use_tables: List[str],
        
        # allows unpacking/handling patient records into events
        event_handlers: Dict[str, Callable[[Event], _Hashable]] = {},

        # used to handle continuous values
        compute_histograms: List[str] = [], # tables for which we want to compute histogram
        hist_identifier: Dict[str, Callable[[Event], _Hashable]] = {}, # used to identify the event in the histogram
        size_per_event_bin: Dict[str, int] = {}, # number of bins to use for each tables histograms
        discrete_event_handlers: Dict[str, Callable[[Event, int], _Hashable]] = {}, # after digitization we need to apply another layer of handlers for things like serialization
        max_continuous_per_table: int = 50, # if there are too many continuous events, we only compute histograms for the top k most common

        # used to discretize time
        size_per_time_bin: int = 10,
        
        max_visits: Union[None, int] = None,
        label_fn: Callable[..., List[int]] = None, 
        label_vector_len: int = -1,
        name: str = "halo_processor",
        refresh_cache: bool = False,
        expedited_load: bool = False,
        dataset_filepath: str = None,
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
        
        self.compute_histograms = compute_histograms
        assert set(self.compute_histograms) == set(hist_identifier.keys()), "Histogram identifiers must be defined for each table"
        self.hist_identifier = hist_identifier
        assert all([100 % b  == 0 for b in size_per_event_bin.values()]), "Histogram bins must be a factor of 100" # We make k evenly sized bins, so size_per_event_bin * k = 100%
        self.size_per_event_bin = size_per_event_bin
        
        self.discrete_event_handlers = discrete_event_handlers
        self.max_continuous_per_table = max_continuous_per_table

        assert set(self.compute_histograms) == set(self.size_per_event_bin.keys()), "Histogram bins must be defined for each table"

        # compute this automatically now
        assert size_per_time_bin > 0, "Nonnegative size_per_time_bin required. May be due to user error, or value is not defined."
        assert 100 / size_per_time_bin == int(100 / size_per_time_bin), "size_per_time_bin must be a factor of 100" # We make k evenly sized bins, so size_per_time_bin * k = 100%
        self.size_per_time_bin = size_per_time_bin
        self.time_vector_length = (100 // size_per_time_bin)

        self.max_visits = max_visits

        self.name = name

        self.refresh_cache = refresh_cache
        self.expedited_load = expedited_load
        self.dataset_filepath = dataset_filepath

        # init the indices & dynamically computed utility variables used in HALO training later
        self.set_indices()


    def set_indices(self) -> None:
        """calls `halo.Processor.aggregate_event_indices` and to compute offsets and define vocabulary. 
        Uses offsets to set indices for the HALO visit multi-hot vector representation. Also sets vocabulary metada used when instantiating the `halo.HALO` model. 
        """

        # set aggregate indices; try cache first
        args_to_hash = (
            self.name,
            [self.dataset_filepath if self.dataset_filepath is not None else self.dataset.filepath]
        )
        filename = hash_str("+".join([str(arg) for arg in args_to_hash])) + ".pkl"
        self.filepath = os.path.join(MODULE_CACHE_PATH, filename)
        
        if os.path.exists(self.filepath) and not self.refresh_cache:
            aggregated_results = load_pickle(self.filepath)
            logger.debug(f"Loaded {self.name} from cache at file {self.filepath}")
            if not self.expedited_load:
                self.clean_patients()
        else:
            logger.debug(f"Computing {self.name} from scratch")
            aggregated_results = self.aggregate_event_indices()
            save_pickle(aggregated_results, self.filepath)

        self.global_events = aggregated_results['global_events']

        self.min_birth_datetime = aggregated_results['min_birth_datetime']

        if (self.max_visits == None):
            self.max_visits = aggregated_results['max_visits'] # if the user does not provide, infer from dataset

        self.age_bins = aggregated_results['age_bins'] # bins for age at first visit
        self.visit_bins = aggregated_results['visit_bins'] # bins for time gaps between visits
        self.event_bins = aggregated_results['event_bins'] # bins for continuous values

        # assert the processor works as expected
        assert len(self.global_events) % 2 == 0, "Event index processor not bijective"

        # bidirectional mappings
        self.num_global_events = self.time_vector_length + (len(self.global_events) // 2)

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
    def aggregate_event_indices(self) -> None:
        """Iterates through the provided dataset and computes a vocabulary of events. 
        Each term in the vocabulary is represented as the tuple (`table_name`, `event_handler[table_name](event))` where `table_name` 
        is a string denoting the table which events were parsed from, and `event_hanlder` is a dictionary of callables for unpacking table events.
        Precomputes values used for HALO model initialization.
        """

        # two way mapping from global identifier to index & vice-versa
        # possible since index <> global identifier is bijective
        # type: ((table_name: str, event_value: any): index) or (index: (table_name: str, event_value: any))
        max_visits: int = 0
        min_birth_datetime = pandas.Timestamp.now()
        missing_birth_datetime = 0
        missing_birth_datetime_patients = set()
        for pdata in tqdm(list(self.dataset), desc="Computing min birth datetime"):
                max_visits = max(max_visits, len(pdata.visits))
                if pdata.birth_datetime != None:
                    min_birth_datetime = min(min_birth_datetime, pdata.birth_datetime)
                else:
                    missing_birth_datetime += 1
                    missing_birth_datetime_patients.add(pdata.patient_id)
        logging.debug(f"Missing birth datetime for {missing_birth_datetime} patients")

        global_events = {}

        # used to compute discretized visit gaps in the processor
        age_gaps: int = []
        visit_gaps: int = [] 

        # a set of all table events in the dataset used to compute histograms for automatic continuous value handling
        # keys: table, value: dictionary of key: event_id, value: list of (raw_event, event_value)
        continuous_values_for_hist: Dict[str, Dict[List[float]]] = self.new_nested_defaultdict()
        visits_without_tables = 0
        visits_without_tables_visits = set()
        coinciding_visits = 0
        global_event_set = set()
        for pdata in tqdm(list(self.dataset), desc="Processor computing vocabulary"):
            if pdata.birth_datetime == None:
                continue

            previous_time = pdata.birth_datetime

            # compute global event
            for vdata in sorted(pdata.visits.values(), key=lambda v: v.encounter_time): # visit events are not sorted by default
                # compute time since last visit; if negative, skip
                time_since_last_visit_delta = vdata.encounter_time - previous_time
                time_since_last_visit = time_since_last_visit_delta.days * 24 + time_since_last_visit_delta.seconds / 3600
            
                # omit any events which do not have events to model
                if len(vdata.available_tables) == 0:
                    visits_without_tables += 1
                    visits_without_tables_visits.add((pdata.patient_id, vdata.visit_id))
                    logger.debug(f"Patient {pdata.patient_id} has a visit with no events to model. This is not allowed.")
                    continue

                if time_since_last_visit == 0:
                    coinciding_visits += 1

                if previous_time == pdata.birth_datetime:
                    age_gaps.append(time_since_last_visit)
                else:
                    visit_gaps.append(time_since_last_visit)
                    
                previous_time = vdata.encounter_time

                for table in vdata.available_tables:
                    # valid_tables == None signals we want to use all tables
                    # otherwise, omit any table not in the whitelist
                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    for te_raw in table_events_raw:
                        te = event_handler(te_raw) if event_handler else te_raw.code
                        if table in self.compute_histograms:
                            event_id = self.hist_identifier[table](te_raw)
                            continuous_values_for_hist[table][event_id].append((te_raw, te)) # te only has numerical information, te_raw has the full event with metadata; we give the option to create vocabulary elements with the full event + the discretized value
                        
                        # we can put discrete table events directly into the vocabulary. 
                        # For continuous values, we need to discretize them first, 
                        # which requires the whole dataset to be processed
                        else:
                            global_event = (table, te)
                            if global_event not in global_event_set:
                                global_event_set.add(global_event)
                  
        global_event_set = list(global_event_set)
        np.random.shuffle(global_event_set)
        for global_event in global_event_set:              
            index_of_global_event = self.time_vector_length + (len(global_events) // 2)
            global_events[global_event] = index_of_global_event
            global_events[index_of_global_event] = global_event

        logger.warn(f"Missing birth datetime for {missing_birth_datetime} patients")
        logger.warn(f"Found {visits_without_tables} visits without tables")
        logger.warn(f"Found {coinciding_visits} coinciding visits")
        logger.warn(f"Determined {[(k, len(v)) for k, v in continuous_values_for_hist.items()]} event_ids to compute histograms for")
        
        logger.warn(f'Cleaning dataset of patients with missing birth datetime and visits without tables')
        for p_id in tqdm(missing_birth_datetime_patients, desc="Removing patients with missing birth datetime"):
            self.dataset.patients.pop(p_id)
            self.dataset.patient_ids.remove(p_id)

        for p_id, v_id in tqdm(visits_without_tables_visits, desc="Removing visits without tables"):
            if p_id in self.dataset.patients:
                self.dataset.patients[p_id].visits.pop(v_id)
                self.dataset.patients[p_id].index_to_visit_id = {i: v for i, v in enumerate(self.dataset.patients[p_id].visits.keys())}
                if len(self.dataset.patients[p_id].visits) == 0:
                    self.dataset.patients.pop(p_id)
                    self.dataset.patient_ids.remove(p_id)

        # used to compute discretized ages and visit gaps in the processor
        age_bins = []
        for b in range(0, 100 + 1, self.size_per_time_bin):
            age_bins.append(np.percentile(age_gaps, b))
            
        age_bins[-1] = age_bins[-1] + self.EPSILON # add epsilon to the last bin to ensure we capture the max value in the dataset
        logger.warn("Age bins: %s", [("%.3f" % b) for b in age_bins])
        
        visit_bins = []
        for b in range(0, 100 + 1, self.size_per_time_bin):
            visit_bins.append(np.percentile(visit_gaps, b))

        visit_bins[-1] = visit_bins[-1] + self.EPSILON # add epsilon to the last bin to ensure we capture the max value in the dataset
        logger.warn("Visit bins: %s", [("%.3f" % b) for b in visit_bins])

        # compute the discretization bins for continuous values per event per table (each event_id has its own set of bins) based on the values present for that event in the dataset
        # keys: table, value: dictionary of key: event_id, value: list of bin boundaries
        event_bins = self.new_nested_defaultdict()
        for table in self.compute_histograms:
            # compute the quantity of event bins for each event type within the table
            num_event_bins = self.size_per_event_bin[table]
            table_continuous_events = continuous_values_for_hist[table].items()
            table_continuous_events = sorted(table_continuous_events, key=lambda x: len(x[1]), reverse=True)[:self.max_continuous_per_table]
            for event_id, event_values in table_continuous_events:
                values = [v for _, v in event_values]
                for b in range(0, 100 + 1, num_event_bins):
                    bin_boundary = np.percentile(values, b)
                    event_bins[table][event_id].append(bin_boundary)

                event_bins[table][event_id][-1] = event_bins[table][event_id][-1] + self.EPSILON # add epsilon to the last bin to ensure we capture the max value in the dataset
                logger.warn("Event bins for (%s) %s: %s", table, event_id, [ ("%.3f" % b) for b in event_bins[table][event_id]])

        # generate vocabulary for continuous valued events now that we have bins
        global_event_set = set()
        for table in self.compute_histograms:
            # TODO: Ask if this simpler way is going to mess up due to unit differences (and if we're better off separating/converting those if so)
            # num_event_bins = self.size_per_event_bin[table]
            # for event_id in event_bins[table]:
            #     pyhealth_event_obj = continuous_values_for_hist[table][event_id][0][0]
            #     for bin_id in range(num_event_bins):
            #         if table in self.discrete_event_handlers:
            #             vocabulary_element = self.discrete_event_handlers[table](pyhealth_event_obj, bin_id)
                        
            #         global_event = (table, vocabulary_element)
            #         if global_event not in global_event_set:
            #             global_event_set.add(global_event)
            
            for event_id, event_values in tqdm(table_continuous_events, desc=f"Converting continuous events in {table} to vocabulary"):
                for pyhealth_event_obj, value in event_values:
                    discretization_bins = event_bins[table][event_id]
                    bin_id = np.digitize(value, discretization_bins) - 1 # -1 to account for the 0th bin

                    if table in self.discrete_event_handlers:
                        vocabulary_element = self.discrete_event_handlers[table](pyhealth_event_obj, bin_id)

                    global_event = (table, vocabulary_element)
                    if global_event not in global_event_set:
                        global_event_set.add(global_event)

        global_event_set = list(global_event_set)
        np.random.shuffle(global_event_set)
        for global_event in global_event_set:              
            index_of_global_event = self.time_vector_length + (len(global_events) // 2)
            global_events[global_event] = index_of_global_event
            global_events[index_of_global_event] = global_event

        return {
            'global_events': global_events,
            'max_visits': max_visits,
            'min_birth_datetime': min_birth_datetime,
            'age_bins': age_bins,
            'visit_bins': visit_bins,
            'event_bins': event_bins
        }
        
    """
    If we have loaded the aggregated event indices, we need to do the dataset cleaning separately
    """
    def clean_patients(self) -> None:
        """Iterates through the provided dataset and removes bad datapoints.
        Specifically removes patients without a birth datetime, visits without tables, and finally patients without good visits
        """

        missing_birth_datetime_patients = set()
        for pdata in tqdm(list(self.dataset), desc="Finding patients with missing birth datetime"):
            if pdata.birth_datetime is None:
                missing_birth_datetime_patients.add(pdata.patient_id)

        visits_without_tables_visits = set()
        for pdata in tqdm(list(self.dataset), desc="Finding visits without tables"):
            if pdata.birth_datetime == None:
                continue

            # compute global event
            for vdata in sorted(pdata.visits.values(), key=lambda v: v.encounter_time): # visit events are not sorted by default
                # omit any events which do not have events to model
                if len(vdata.available_tables) == 0:
                    visits_without_tables_visits.add((pdata.patient_id, vdata.visit_id))
                    continue

        logger.warn(f'Cleaning dataset of patients with missing birth datetime and visits without tables')
        for p_id in tqdm(missing_birth_datetime_patients, desc="Removing patients with missing birth datetime"):
            self.dataset.patients.pop(p_id)
            self.dataset.patient_ids.remove(p_id)

        for p_id, v_id in tqdm(visits_without_tables_visits, desc="Removing visits without tables"):
            if p_id in self.dataset.patients:
                self.dataset.patients[p_id].visits.pop(v_id)
                self.dataset.patients[p_id].index_to_visit_id = {i: v for i, v in enumerate(self.dataset.patients[p_id].visits.keys())}
                if len(self.dataset.patients[p_id].visits) == 0:
                    self.dataset.patients.pop(p_id)
                    self.dataset.patient_ids.remove(p_id)
    
    def process_batch(self, batch) -> Tuple:
        """Convert a batch of `pyhealth.data.Patient` objects into a batch of multi-hot vectors for the HALO model.

        Use the indices from `halo.Processor.set_indices` to produce a series of visit vectors; each visit vector will include the following:
            - patient label
            - visit events
            - inter-visit gap
            - visit metadata
        Towards this, the following operations will be done: 
        1. for each `pyhealth.data.Patient` compute the patient label using the label function using the label function provided during object instantiation.
        2. translate each visit into a mulit-hot vector where indices using the global vocabulary computed during the object instantiation.
        3. translate inter-visit gap and patient age into multihot vector 
        
        Returns:
            the tuple: (sample_multi_hot, sample_mask)
                sample_multi_hot: the vector (batch_size, total_visit_size, total_vocab_size) representing samples within the batch. Each sample is a multihot binary vector converted using the aforementioned operations. 
                sample_mask: the vector (batch_size, self.total_visit_size, 1) representing which visits do not have any data. This mask denotes whether a visit should be ignored or not in the HALO model.
        """
        batch_size = len(batch)
        
        # dim 0: batch
        # dim 1: visit vectors
        # dim 2: concat(event multihot, label onehot, metadata)
        sample_multi_hot = np.zeros((batch_size, self.total_visit_size, self.total_vocab_size)) # patient data the model reads
        sample_mask = np.zeros((batch_size, self.total_visit_size, 1)) # visits that are unlabeled        
        for pidx, pdata in enumerate(batch):
            if pdata.birth_datetime == None:
                continue

            previous_time = pdata.birth_datetime
            # build multihot vector for patient events
            for visit_index, vdata in enumerate(sorted(pdata.visits.values(), key=lambda v: v.encounter_time)):
                time_since_last_visit_delta = vdata.encounter_time - previous_time
                time_since_last_visit = time_since_last_visit_delta.days * 24 + time_since_last_visit_delta.seconds / 3600
                if visit_index == 0:
                    time_since_last_visit = np.digitize(time_since_last_visit, self.age_bins) - 1 # -1 to account for the 0th bin
                else:
                    time_since_last_visit = np.digitize(time_since_last_visit, self.visit_bins) - 1 # -1 to account for the 0th bin
                inter_visit_gap_vector = np.zeros(self.time_vector_length)
                inter_visit_gap_vector[time_since_last_visit] = 1

                # the next timedelta is previous current visit - start of previous visit 
                # (note: not discharge because we don't model visit durations)
                sample_multi_hot[pidx, self.VISIT_INDEX + visit_index, :self.time_vector_length] = inter_visit_gap_vector
                previous_time = vdata.encounter_time

                sample_mask[pidx, self.VISIT_INDEX + visit_index] = 1
                for table in vdata.available_tables:
                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    for te_raw in table_events_raw:
                        te = event_handler(te_raw) if event_handler else te_raw.code
                        if table in self.compute_histograms:
                            event_id = self.hist_identifier[table](te_raw)
                            if event_id not in self.event_bins[table]:
                                continue
                            
                            te = np.digitize(te, self.event_bins[table][event_id]) - 1 # -1 to account for the 0th bin
                            if table in self.discrete_event_handlers:
                                te = self.discrete_event_handlers[table](te_raw, te)

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
        """Processing function which takes in a subset of data (such as training data) and returns an interator to get the next batche of the data subset.
        No data shuffling is performed, and the batches have the `halo.Processor.process_batch` function applied to it. 

        Args:
            data_subset: an instance of BaseEHRDataset, which contains patient samples
            batch_size: the batch size for the number of samples to convert at a time

        Returns:
            Python generator object accessing batches of the data_subset
        """

        batch_size = min(len(data_subset), batch_size)

        batch_offset = 0
        while (batch_offset + batch_size <= len(data_subset)):
            batch = data_subset[batch_offset: batch_offset + batch_size]
            batch_offset += batch_size # prepare for next iteration
            yield self.process_batch(batch)