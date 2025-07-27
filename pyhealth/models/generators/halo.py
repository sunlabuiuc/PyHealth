import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

import os
import numpy as np
import random
import pickle
from tqdm import tqdm
from sklearn import metrics

from pyhealth.datasets import SampleEHRDataset, HALO_MIMIC3Dataset
from pyhealth.models.base_model import BaseModel

# Import the HALO transformer implementation
from pyhealth.models.generators.halo_resources.halo_model import HALOModel
from pyhealth.models.generators.halo_resources.halo_config import HALOConfig

class HALO:

    def __init__(
        self,
        dataset: HALO_MIMIC3Dataset,
        config: HALOConfig,
        save_dir: str = "./save/",
        train_on_init: bool = True
    ) -> None:
        SEED = 4
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        self.config = config
        self.dataset = dataset
        self.save_dir = save_dir

        local_rank = -1
        fp16 = False
        if local_rank == -1:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        self.train_ehr_dataset = pickle.load(open(f'{self.dataset.pkl_data_dir}trainDataset.pkl', 'rb'))
        self.val_ehr_dataset = pickle.load(open(f'{self.dataset.pkl_data_dir}valDataset.pkl', 'rb'))
        self.index_to_code = pickle.load(open(f"{self.dataset.pkl_data_dir}indexToCode.pkl", "rb"))
        self.id_to_label = pickle.load(open(f"{self.dataset.pkl_data_dir}idToLabel.pkl", "rb"))
        test_ehr_dataset = pickle.load(open(f'{self.dataset.pkl_data_dir}testDataset.pkl', 'rb'))
        try:
            print(f"test data head 1: \n{test_ehr_dataset.head()}")
        except:
            print(f"test data head 2: \n{test_ehr_dataset[:5]}")
        train_c = set([c for p in self.train_ehr_dataset for v in p['visits'] for c in v])
        self.test_ehr_dataset = [{'labels': p['labels'], 'visits': [[c for c in v if c in train_c] for v in p['visits']]} for p in test_ehr_dataset]

        if train_on_init:
            self.train()


    def train(self) -> None:

        # HELPER:
        def get_batch(loc, batch_size, mode):
            # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
            #   Where each patient P is [V_1, V_2, ... , V_j]
            #     Where each visit V is [C_1, C_2, ... , C_k]
            #   And where each Label L is a binary vector [L_1 ... L_n]
            if mode == 'train':
                ehr = self.train_ehr_dataset[loc:loc+batch_size]
            elif mode == 'valid':
                ehr = self.val_ehr_dataset[loc:loc+batch_size]
            else:
                ehr = self.test_ehr_dataset[loc:loc+batch_size]
                
            batch_ehr = np.zeros((len(ehr), self.config.n_ctx, self.config.total_vocab_size))
            batch_mask = np.zeros((len(ehr), self.config.n_ctx, 1))
            for i, p in enumerate(ehr):
                visits = p['visits']
                for j, v in enumerate(visits):
                    batch_ehr[i,j+2][v] = 1
                    batch_mask[i,j+2] = 1
                batch_ehr[i,1,self.config.code_vocab_size:self.config.code_vocab_size+self.config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
                batch_ehr[i,len(visits)+1,self.config.code_vocab_size+self.config.label_vocab_size+1] = 1 # Set the final visit to have the end token
                batch_ehr[i,len(visits)+2:,self.config.code_vocab_size+self.config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
            
            batch_mask[:,1] = 1 # Set the mask to cover the labels
            batch_ehr[:,0,self.config.code_vocab_size+self.config.label_vocab_size] = 1 # Set the first visits to be the start token
            batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
            return batch_ehr, batch_mask

        # HELPER
        def shuffle_training_data(train_ehr_dataset):
            np.random.shuffle(train_ehr_dataset)

        # TRAIN PIPELINE:

        # Load any previous checkpoint if exists
        self.model = HALOModel(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if os.path.exists(f"{self.save_dir}halo_model"):
            # print("Loading previous model")
            checkpoint = torch.load(f'{self.save_dir}halo_model', map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Train
        global_loss = 1e10
        for e in tqdm(range(self.config.epoch)):
            shuffle_training_data(self.train_ehr_dataset)
            for i in range(0, len(self.train_ehr_dataset), self.config.batch_size):
                self.model.train()
                
                batch_ehr, batch_mask = get_batch(i, self.config.batch_size, 'train')
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=self.config.pos_loss_weight)
                loss.backward()
                self.optimizer.step()
                
                if i % (500*self.config.batch_size) == 0:
                    print("Epoch %d, Iter %d: Training Loss:%.6f"%(e, i, loss * 8))
                if i % (500*self.config.batch_size) == 0:
                    if i == 0:
                        continue
                
                    self.model.eval()
                    with torch.no_grad():
                        val_l = []
                        for v_i in range(0, len(self.val_ehr_dataset), self.config.batch_size):
                            batch_ehr, batch_mask = get_batch(v_i, self.config.batch_size, 'valid')
                            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                            batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                    
                            val_loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=self.config.pos_loss_weight)
                            val_l.append((val_loss).cpu().detach().numpy())
                    
                        cur_val_loss = np.mean(val_l)
                        print("Epoch %d Validation Loss:%.7f"%(e, cur_val_loss))
                        if cur_val_loss < global_loss:
                            global_loss = cur_val_loss
                            state = {
                                    'model': self.model.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'iteration': i
                                }
                            torch.save(state, f'{self.save_dir}halo_model')
                            print('\n------------ Save best model ------------\n')

    def test(self, testing_results_dir: str = "./results/testing_stats/") -> None:

        ## HELPER FUNC:
        def get_batch(loc, batch_size, mode):
            # EHR data saved as [(P_1, L_1), (P_2, L_2), ... , (P_i, L_i)]
            #   Where each patient P is [V_1, V_2, ... , V_j]
            #     Where each visit V is [C_1, C_2, ... , C_k]
            #   And where each Label L is a binary vector [L_1 ... L_n]
            if mode == 'train':
                ehr = self.train_ehr_dataset[loc:loc+batch_size]
            elif mode == 'valid':
                ehr = self.val_ehr_dataset[loc:loc+batch_size]
            else:
                ehr = self.test_ehr_dataset[loc:loc+batch_size]
                
            batch_ehr = np.zeros((len(ehr), self.config.n_ctx, self.config.total_vocab_size))
            batch_mask = np.zeros((len(ehr), self.config.n_ctx, 1))
            for i, p in enumerate(ehr):
                visits = p['visits']
                for j, v in enumerate(visits):
                    batch_ehr[i,j+2][v] = 1
                    batch_mask[i,j+2] = 1
                batch_ehr[i,1,self.config.code_vocab_size:self.config.code_vocab_size+self.config.label_vocab_size] = np.array(p['labels']) # Set the patient labels
                batch_ehr[i,len(visits)+1,self.config.code_vocab_size+self.config.label_vocab_size+1] = 1 # Set the final visit to have the end token
                batch_ehr[i,len(visits)+2:,self.config.code_vocab_size+self.config.label_vocab_size+2] = 1 # Set the rest to the padded visit token
            
            batch_mask[:,1] = 1 # Set the mask to cover the labels
            batch_ehr[:,0,self.config.code_vocab_size+self.config.label_vocab_size] = 1 # Set the first visits to be the start token
            batch_mask = batch_mask[:,1:,:] # Shift the mask to match the shifted labels and predictions the model will return
            return batch_ehr, batch_mask

        ## HELPER FUNC:
        def conf_mat(x, y):
            totaltrue = np.sum(x)
            totalfalse = len(x) - totaltrue
            truepos, totalpos = np.sum(x & y), np.sum(y)
            falsepos = totalpos - truepos
            return np.array([[totalfalse - falsepos, falsepos], #true negatives, false positives
                            [totaltrue - truepos, truepos]]) #false negatives, true positives

        ## MAIN TEST FUNC:
        checkpoint = torch.load(f'{self.save_dir}halo_model', map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        confusion_matrix = None
        probability_list = []
        loss_list = []
        n_visits = 0
        n_pos_codes = 0
        n_total_codes = 0
        self.model.eval()
        with torch.no_grad():
            for v_i in tqdm(range(0, len(self.test_ehr_dataset), 2*self.config.batch_size)):
                # Get batch inputs
                batch_ehr, batch_mask = get_batch(v_i, 2*self.config.batch_size, 'test')
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                    
                # Get batch outputs
                test_loss, predictions, labels = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask, pos_loss_weight=self.config.pos_loss_weight)
                batch_mask_array = batch_mask.squeeze().cpu().detach().numpy()
                rounded_preds = np.around(predictions.squeeze().cpu().detach().numpy()).transpose((2,0,1)) 
                rounded_preds = rounded_preds + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
                rounded_preds = rounded_preds.flatten()
                true_values = labels.squeeze().cpu().detach().numpy().transpose((2,0,1))
                true_values = true_values + batch_mask_array - 1 # Setting the masked visits to be -1 to be ignored by the confusion matrix
                true_values = true_values.flatten()

                # Append test lost
                loss_list.append(test_loss.cpu().detach().numpy())
                    
                # Add number of visits and codes
                n_visits += torch.sum(batch_mask).cpu().item()
                n_pos_codes += torch.sum(labels).cpu().item()
                n_total_codes += (torch.sum(batch_mask) * self.config.total_vocab_size).cpu().item()

                # Add confusion matrix
                batch_cmatrix = conf_mat(true_values == 1, rounded_preds == 1)
                batch_cmatrix[0][0] = torch.sum(batch_mask) * self.config.total_vocab_size - batch_cmatrix[0][1] - batch_cmatrix[1][0] - batch_cmatrix[1][1] # Remove the masked values
                confusion_matrix = batch_cmatrix if confusion_matrix is None else confusion_matrix + batch_cmatrix

                # Calculate and add probabilities 
                # Note that the masked codes will have probability 1 and be ignored
                label_probs = torch.abs(labels - 1.0 + predictions)
                log_prob = torch.sum(torch.log(label_probs)).cpu().item()
                probability_list.append(log_prob)

        # Save intermediate values in case of error
        intermediate = {}
        intermediate["Losses"] = loss_list
        intermediate["Confusion Matrix"] = confusion_matrix
        intermediate["Probabilities"] = probability_list
        intermediate["Num Visits"] = n_visits
        intermediate["Num Positive Codes"] = n_pos_codes
        intermediate["Num Total Codes"] = n_total_codes
        pickle.dump(intermediate, open(f"{testing_results_dir}HALO_intermediate_results.pkl", "wb"))

        #Extract, save, and display test metrics
        avg_loss = np.nanmean(loss_list)
        tn, fp, fn, tp = confusion_matrix.ravel()
        acc = (tn + tp)/(tn+fp+fn+tp)
        prc = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = (2 * prc * rec)/(prc + rec)
        log_probability = np.sum(probability_list)
        pp_visit = np.exp(-log_probability/n_visits)
        pp_positive = np.exp(-log_probability/n_pos_codes)
        pp_possible = np.exp(-log_probability/n_total_codes)

        metrics_dict = {}
        metrics_dict['Test Loss'] = avg_loss
        metrics_dict['Confusion Matrix'] = confusion_matrix
        metrics_dict['Accuracy'] = acc
        metrics_dict['Precision'] = prc
        metrics_dict['Recall'] = rec
        metrics_dict['F1 Score'] = f1
        metrics_dict['Test Log Probability'] = log_probability
        metrics_dict['Perplexity Per Visit'] = pp_visit
        metrics_dict['Perplexity Per Positive Code'] = pp_positive
        metrics_dict['Perplexity Per Possible Code'] = pp_possible
        pickle.dump(metrics_dict, open(f"{testing_results_dir}HALO_Metrics.pkl", "wb"))


    def convert_ehr(self, ehrs, index_to_code=None):
        ehr_outputs = []
        for i in range(len(ehrs)):
            ehr = ehrs[i]
            ehr_output = []
            labels_output = ehr[1][self.config.code_vocab_size:self.config.code_vocab_size+self.config.label_vocab_size]
            if index_to_code is not None:
                labels_output = [index_to_code[idx + self.config.code_vocab_size] for idx in np.nonzero(labels_output)[0]]
            for j in range(2, len(ehr)):
                visit = ehr[j]
                visit_output = []
                indices = np.nonzero(visit)[0]
                end = False
                for idx in indices:
                    if idx < self.config.code_vocab_size: 
                        visit_output.append(index_to_code[idx] if index_to_code is not None else idx)
                    elif idx == self.config.code_vocab_size+self.config.label_vocab_size+1:
                        end = True
                if visit_output != []:
                    ehr_output.append(visit_output)
                if end:
                    break
            ehr_outputs.append({'visits': ehr_output, 'labels': labels_output})
        ehr = None
        ehr_output = None
        labels_output = None
        visit = None
        visit_output = None
        indices = None
        return ehr_outputs


    def synthesize_dataset(self, pkl_save_dir: str = "./results/datasets/") -> None:

        ## HELPER:
        def sample_sequence(model, length, context, batch_size, device='cuda', sample=True):
            empty = torch.zeros((1,1,self.config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
            context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
            prev = context.unsqueeze(1)
            context = None
            with torch.no_grad():
                for _ in range(length-1):
                    prev = model.sample(torch.cat((prev,empty), dim=1), sample)
                    if torch.sum(torch.sum(prev[:,:,self.config.code_vocab_size+self.config.label_vocab_size+1], dim=1).bool().int(), dim=0).item() == batch_size:
                        break
            ehr = prev.cpu().detach().numpy()
            prev = None
            empty = None
            return ehr

        ## MAIN FUNC:
        synthetic_ehr_dataset = []
        stoken = np.zeros(self.config.total_vocab_size)
        stoken[self.config.code_vocab_size+self.config.label_vocab_size] = 1
        for i in tqdm(range(0, len(self.train_ehr_dataset), self.config.sample_batch_size)):
            bs = min([len(self.train_ehr_dataset)-i, self.config.sample_batch_size])
            batch_synthetic_ehrs = sample_sequence(self.model, self.config.n_ctx, stoken, batch_size=bs, device=self.device, sample=True)
            batch_synthetic_ehrs = self.convert_ehr(batch_synthetic_ehrs)
            synthetic_ehr_dataset += batch_synthetic_ehrs

        pickle.dump(synthetic_ehr_dataset, open(f"{pkl_save_dir}haloDataset.pkl", 'wb'))






# class HALO(BaseModel):
#     """
#     HALO model wrapper for PyHealth, leveraging the BaseModel interface.

#     Args:
#         dataset: SampleEHRDataset for EHR sequences (visits of codes).
#         config: Transformer configuration object for HALOModel.
#         feature_key: key in dataset samples for code-based visits (e.g., "list_codes").
#         mode: one of "binary", "multiclass", or "multilabel" for loss function.
#         label_key: key in dataset samples for target visits; defaults to feature_key for next-visit prediction.
#         pos_loss_weight: weight applied to positive labels in BCE loss.
#     """
#     def __init__(
#         self,
#         dataset: SampleEHRDataset,
#         config: HALOConfig,
#         feature_key: str = "list_codes",
#         mode: str = "multilabel",
#         label_key: Optional[str] = None,
#         pos_loss_weight: float = 1.0,
#     ):
#         super(HALO, self).__init__(dataset)
#         self.feature_key = feature_key
#         self.label_key = label_key or feature_key
#         self.pos_loss_weight = pos_loss_weight

#         # Set mode for loss and evaluation
#         self.mode = mode

#         # Tokenizer for the code-based input
#         self.tokenizer = dataset.input_processors[feature_key]
#         self.vocab_size = self.tokenizer.size()

#         # Instantiate the underlying HALO transformer model
#         self.halo = HALOModel(config)

#     def _prepare_input_visits(self, codes: List[List[Any]]) -> torch.Tensor:
#         """
#         Convert list of visits of codes into multi-hot tensor.

#         Args:
#             codes: nested list of shape (batch, num_visits, codes_in_visit)

#         Returns:
#             Tensor of shape (batch, num_visits, vocab_size) with 0/1 entries.
#         """
#         # batch_encode_3d returns List[List[List[int]]]
#         token_ids = self.tokenizer.batch_encode_3d(codes)
#         batch_size = len(token_ids)
#         max_visits = len(token_ids[0])

#         visits = torch.zeros(
#             batch_size, max_visits, self.vocab_size, device=self.device
#         )
#         for i in range(batch_size):
#             for t, visit_ids in enumerate(token_ids[i]):
#                 for cid in visit_ids:
#                     if cid is None:
#                         continue
#                     visits[i, t, cid] = 1.0
#         return visits

#     def forward(
#         self,
#         **kwargs: Any
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Forward propagation for HALO within PyHealth.

#         Expects kwargs to contain:
#           - feature_key: raw visit code lists.
#           - optional label_key: same format for next-visit targets.
#           - optional masks: tensor or nested lists for masking visits.

#         Returns a dict with keys:
#           - loss: training loss (if labels provided).
#           - y_prob: predicted probabilities of codes per visit.
#           - y_true: ground-truth multi-hot labels (shifted by one visit).
#           - logits: raw logits from the HALO transformer.
#         """
#         # Prepare input tensor
#         raw_codes = kwargs[self.feature_key]
#         input_visits = self._prepare_input_visits(raw_codes)

#         # Gather optional training labels and masks
#         ehr_labels = None
#         ehr_masks = None
#         if self.label_key in kwargs:
#             # similarly convert label visits to multi-hot
#             ehr_labels = self._prepare_input_visits(kwargs[self.label_key])
#         if "masks" in kwargs:
#             ehr_masks = torch.tensor(kwargs["masks"], device=self.device, dtype=torch.float)

#         # Call HALOModel: returns loss & probabilities if labels, else probabilities
#         if ehr_labels is not None:
#             loss, code_probs, shift_labels = self.halo(
#                 input_visits,
#                 ehr_labels=ehr_labels,
#                 ehr_masks=ehr_masks,
#                 pos_loss_weight=self.pos_loss_weight,
#             )
#             results = {"loss": loss, "y_prob": code_probs, "y_true": shift_labels}
#         else:
#             code_probs = self.halo(input_visits)
#             results = {"y_prob": code_probs}

#         # Attach logits if needed
#         if hasattr(self.halo, 'last_logits'):
#             results['logits'] = self.halo.last_logits

#         return results

# # Example usage:
# if __name__ == "__main__":
#     from pyhealth.datasets import SampleEHRDataset
#     # define samples with visits of codes as nested lists
#     samples = [
#         {"patient_id": "p0", "visit_id": "v0", "list_codes": [["A", "B"], ["C"]]},
#         {"patient_id": "p1", "visit_id": "v0", "list_codes": [["D"], ["E", "F"]]},
#     ]
#     dataset = SampleEHRDataset(samples=samples, dataset_name="halo_test")

#     # Build transformer config
#     config = HALOConfig(
#         n_layer=4,
#         n_head=8,
#         n_embd=128,
#         total_vocab_size=dataset.input_processors['list_codes'].size(),
#         n_positions=dataset.max_visit_length,
#         n_ctx=dataset.max_visit_length,
#         layer_norm_epsilon=1e-5,
#     )

#     model = HALO(
#         dataset=dataset,
#         config=config,
#         feature_key='list_codes',
#         mode='multilabel',
#     )
#     from pyhealth.datasets import get_dataloader
#     loader = get_dataloader(dataset, batch_size=2, shuffle=False)
#     batch = next(iter(loader))
#     output = model(**batch)
#     print(output)
