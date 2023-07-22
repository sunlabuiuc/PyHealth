import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn

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
            parsed_labels_output = self.processor.invert_label(labels_output) if self.processor.invert_label != None else labels_output

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
            
            ehr_outputs.append({self.VISITS: sample_as_ehr, self.TIME: sample_time_gaps, self.LABEL: parsed_labels_output})

        return ehr_outputs

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