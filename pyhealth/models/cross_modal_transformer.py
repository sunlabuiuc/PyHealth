

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d


from typing import Dict, List, Optional, Tuple
from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

try:
    from IPython import display
    from ipywidgets import  interactive, IntSlider
except:
    raise ImportError("IPython is not installed. Please install it by running 'pip install ipython'"
                      "and 'pip install ipywidgets'")



'''
Colab tutorial can be found at:
https://colab.research.google.com/drive/1jwD5NX8cR47MRtnOW_drKIS6lP1tbWFc?usp=sharing
'''

try:
    from einops import rearrange, reduce, repeat
    from einops.layers.torch import Rearrange, Reduce
except:
    raise ImportError("einops is not installed. Please install it by running"
                      "'pip install einops'")

class PositionalEncoding(nn.Module):
    '''
    Positional encoding for the Cross-Modal Transformer

    Args:
        d_model: int, embedding dim
        dropout: float, dropout rate
        max_len: int, max length of the input sequence
    '''

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, input sequence
        Returns:
            Tensor, input sequence with positional encoding
        '''
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class Window_Embedding(nn.Module): 
    '''
    Multi-Scale 1D-CNN for Representation Learning from non-overlapping 
    time windows in a epoch
    Args:
        in_channels: int, number of input channels
        window_size: int, size of the window
        emb_size: int, embedding size
    '''

    def __init__(self, in_channels: int = 1, window_size: int = 50, emb_size: int = 64):
        super(Window_Embedding, self).__init__()

        self.projection_1 =  nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//4, kernel_size = window_size, stride = window_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size//4),
            # Rearrange('b e s -> b s e'),
            )
        self.projection_2 =  nn.Sequential(#################
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//8, kernel_size = 5, stride = 5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//8, emb_size//4, kernel_size = 5, stride = 5),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size-emb_size//4)//2, kernel_size = 2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size-emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        
        self.projection_3 =  nn.Sequential(#################
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(in_channels, emb_size//4, kernel_size = 25, stride = 25),
            nn.LeakyReLU(),
            nn.Conv1d(emb_size//4, (emb_size-emb_size//4)//2, kernel_size =2, stride = 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d((emb_size-emb_size//4)//2),
            # Rearrange('b e s -> b s e'),
            )
        
        
        self.projection_4 = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains, in=>B,1,3000 out=>B,64,60
            nn.Conv1d(emb_size, emb_size, kernel_size = 1, stride = 1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(emb_size),
            Rearrange('b e s -> b s e'),)
            
        #in=>B,64,60 out=>B,64,61
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.arrange1 = Rearrange('b s e -> s b e')
        #in=>61,B,64 out=>61,B,64
        self.pos = PositionalEncoding(d_model=emb_size)
        #in=>61,B,64 out=>B,61,64
        self.arrange2 = Rearrange('s b e -> b s e ')

    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, input sequence
        Returns:
            1D-CNN embedding of the input sequence with positional encoding + additional cls token
        '''
        if x.shape[0]!=1:
            x = x.squeeze().unsqueeze(dim = 1)
        b,_, _ = x.shape
        x_1 = self.projection_1(x)  ########################
        x_2 = self.projection_2(x) ###########
        x_3 = self.projection_3(x) 
        x = torch.cat([x_1,x_2,x_3],dim = 1)##### 2)
        x = self.projection_4(x) 
        cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x = self.arrange1(x)
        x = self.pos(x)
        x = self.arrange2(x)
        return x

class Intra_modal_atten(nn.Module): 
    '''
    Intra-modal attention for the Cross-Modal Transformer
    Args:
        d_model: int, embedding dim
        nhead: int, number of heads
        dropout: float, dropout rate
        layer_norm_eps: float, epsilon for layer norm
        window_size: int, size of the window
        First: bool, whether it is the first intra-modal attention in the model
    '''
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 layer_norm_eps=1e-5, window_size = 25, First = True,
                 device=None, dtype=None) -> None:
        super(Intra_modal_atten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
       
        if First == True:
            self.window_embed = Window_Embedding(in_channels = 1, window_size = window_size, emb_size = d_model)
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        self.dropout = Dropout(dropout) 
        self.First = First

    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, input sequence after window embedding
        Returns:
            Tensor, output sequence after intra-modal attention
        '''
        if self.First == True:
            src = self.window_embed(x)
        else:
            src = x

        src2 = self.self_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)   ########
        return out  
    

class Cross_modal_atten(nn.Module): 
    '''
    Cross-modal attention for the Cross-Modal Transformer
    Args:
        d_model: int, embedding dim
        nhead: int, number of heads
        dropout: float, dropout rate
        layer_norm_eps: float, epsilon for layer norm
        First: bool, whether it is the first cross-modal attention in the model
    '''
    def __init__(self, d_model=64, nhead=8, dropout=0.1,
                 layer_norm_eps=1e-5, First = False,
                 device=None, dtype=None) -> None:

        super(Cross_modal_atten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        if First == True:
            self.cls_token = nn.Parameter(torch.randn(1,1, d_model)) ######
        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True,
                                            **factory_kwargs)
        self.dropout = Dropout(dropout) 
        self.First = First

    def forward(self, x1: Tensor,x2: Tensor) -> Tensor:
        '''
        Args:
            x1: Tensor, input sequence from the first modality
            x2: Tensor, input sequence from the second modality
        Returns:
            Tensor, output sequence after cross-modal attention + additional cross-cls token
        '''
        if len(x1.shape) == 2:
            x = torch.cat([x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)], dim=1)
        else:
            x = torch.cat([x1, x2.unsqueeze(dim=1)], dim=1)
        b,_, _ = x.shape
        if self.First == True:
            cls_tokens = repeat(self.cls_token, '() s e -> b s e', b=b)  ######
            # prepend the cls token to the input
            src = torch.cat([cls_tokens, x], dim=1)  #####
        else:
            src = x
        src2 = self.cross_attn(src, src, src)[0]
        out = src + self.dropout(src2)
        out = self.norm(out)
        return out 

class Feed_forward(nn.Module):   
    '''
    Feed-forward network for the Cross-Modal Transformer
    Args:
        d_model: int, embedding dim
        dropout: float, dropout rate
        dim_feedforward: int, dim of the feed-forward network
        layer_norm_eps: float, epsilon for layer norm
    '''

    def __init__(self, d_model=64,dropout=0.1,dim_feedforward=512,
                 layer_norm_eps=1e-5,
                 device=None, dtype=None) -> None:

        super(Feed_forward, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.relu = nn.ReLU()
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, input sequence
        Returns:
            Tensor, output sequence after feed-forward network
        '''        
        src = x
        src2 = self.linear2(self.dropout1(self.relu(self.linear1(src))))
        out = src + self.dropout2(src2)
        out = self.norm(out)
        return out
    

class Epoch_Cross_Transformer(nn.Module):
    '''
    Epoch-level Cross-Modal Transformer for the Cross-Modal Transformer
    Learns a representation for each epoch considering the intra-modal and cross-modal
    interactions between the EEG and EOG signals.

    Args:
        d_model: int, embedding dim
        dim_feedforward: int, dim of the feed-forward network
        window_size: int, size of the window
    '''
    def __init__(self,d_model = 64, dim_feedforward=512,window_size = 25): #  filt_ch = 4
        super(Epoch_Cross_Transformer, self).__init__()
        
        self.eeg_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1,
                                            window_size =window_size, First = True )
        self.eog_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = True )
        
        self.cross_atten = Cross_modal_atten(d_model=d_model, nhead=8, dropout=0.1, First = True )
        
        

    def forward(self, eeg: Tensor,eog: Tensor):#,finetune = False): 
        '''
        Args:
            eeg: Tensor, EEG signal
            eog: Tensor, EOG signal
            
        Returns:
            Tensor, output sequence after the epoch-level Cross-Modal Transformer (representation for a epoch)
        '''
        self_eeg = self.eeg_atten(eeg)
        self_eog = self.eog_atten(eog)

        cross = self.cross_atten(self_eeg[:,0,:],self_eog[:,0,:])

        cross_cls = cross[:,0,:].unsqueeze(dim=1)
        cross_eeg = cross[:,1,:].unsqueeze(dim=1)
        cross_eog = cross[:,2,:].unsqueeze(dim=1)

        # feat_list = [self_eeg,self_eog,cross] 
        ch_feat = torch.cat([self_eeg[:,1:,:].unsqueeze(1),self_eog[:,1:,:].unsqueeze(1)],dim = 1) 
        cross_feat = torch.cat([cross_eeg,cross_eog],dim = 1)
        return cross_cls,ch_feat, cross_feat
    


class Seq_Cross_Modal_Transformer_Network(nn.Module):
    '''
    Sequence Cross-Modal Transformer for Many-to-Many Sleep Stage Classification
    Learns a representation for each epoch considering the intra-modal and cross-modal
    interactions between the EEG and EOG signals. Extracts representations for each epoch
    and learns inter-epoch interactions using a transformer encoder. Finally, MLPs are used for 
    each epoch to predict the sleep stages.

    Args:
        d_model: int, embedding dim
        num_epoch_seq: int, number of epochs in a sequence
        dim_feedforward: int, dim of the feed-forward network
        window_size: int, size of the window
        num_classes: int, number of sleep stages
    '''
    def __init__(self,
                d_model = 128,
                num_epoch_seq = 5,
                dim_feedforward=512,
                window_size = 50,
                num_classes = 6): #  filt_ch = 4
        super(Seq_Cross_Modal_Transformer_Network, self).__init__()

        self.num_epoch_seq = num_epoch_seq
        self.num_classes = num_classes

        self.epoch_transformers = nn.ModuleList([Epoch_Cross_Transformer(d_model=d_model, dim_feedforward=dim_feedforward, window_size=window_size)
                                     for _ in range(num_epoch_seq)])
        
        self.seq_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1,window_size =window_size, First = False )

        self.ff_net = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)


        self.mlp_list = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.Linear(d_model, num_classes)) for _ in range(num_epoch_seq)])

    def forward(self, eeg: Tensor,eog: Tensor):
        '''
        Args:
            eeg: Tensor, EEG signal
            eog: Tensor, EOG signal

        Returns:
            out_list: Sleep stage predictions for each epoch in a sequence (Shape: (B, num_epoch_seq, num_classes)))  
            ch_feat_list: List of tensors containing the representations for each epoch from the epoch-level Cross-Modal Transformer 
                          (Shape: (B, num_epoch_seq, ['EEG','EOG'], num time windows in a epoch, d_model))
            cross_feat_list: List of tensors containing the aggregated representations for each epoch from the epoch-level Cross-Modal Transformer
                            (Shape: (B, num_epoch_seq, ['EEG','EOG'], d_model))
            seq: Aggregated representation for each epoch (Shape: (B, num_epoch_seq, d_model))
        '''

        epoch_list = []
        ch_feat_list = []
        cross_feat_list = []

        for i in range(self.num_epoch_seq):
            epoch, ch_feat, cross_feat = self.epoch_transformers[i](eeg[:, :, i, :], eog[:, :, i, :])

            epoch_list.append(epoch)
            ch_feat_list.append(ch_feat)
            cross_feat_list.append(cross_feat)
        
        seq = torch.cat(epoch_list, dim=1)
        seq = self.seq_atten(seq)

        seq = self.ff_net(seq)

        out_list = [mlp(seq[:, i, :]) for i, mlp in enumerate(self.mlp_list)]

        # Convert to list of tensors
        out_list = torch.stack(out_list, dim=1)
        ch_feat_list = torch.stack(ch_feat_list, dim=1)
        cross_feat_list = torch.stack(cross_feat_list, dim=1)

        return out_list, ch_feat_list, cross_feat_list, seq



class Seq_Cross_Modal_Transformer_PyHealth(BaseModel):
    '''
    Sequence Cross-Modal Transformer for Many-to-Many Sleep Stage Classification

    Learns a representation for each epoch considering the intra-modal and cross-modal
    interactions between the EEG and EOG signals. Extracts representations for each epoch
    and learns inter-epoch interactions using a transformer encoder. Finally, MLPs are used for
    each epoch to predict the sleep stages.

    Paper:  Pradeepkumar, Jathurshan, Mithunjha Anandakumar, Vinith Kugathasan, Dhinesh Suntharalingham, 
            Simon L. Kappel, Anjula C. De Silva, and Chamira US Edussooriya. 
            "Towards interpretable sleep stage classification using cross-modal transformers." 
            arXiv preprint arXiv:2208.06991 (2022).
    Code Repository : https://github.com/Jathurshan0330/Cross-Modal-Transformer


    Args:
        dataset: the dataset to train the model.
        feature_keys: list of keys in samples to use as features    
                      e.g. ['signal']
        label_key: key in samples to use as label
                     e.g. ['label']
        mode: 'multiclass'
        d_model: int, embedding dim
        num_epoch_seq: int, number of epochs in a sequence
        dim_feedforward: int, dim of the feed-forward network
        window_size: int, size of the time window in a epoch
        num_classes: int, number of sleep stages

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> sleepedf_ds = SleepEDFDataset(
        ...                 root="/content/storage.googleapis.com/pyhealth/sleepedf-sample",
        ...                 dev=True,
        ...                 refresh_cache=True)
        >>> sleepedf_ds.stat()
        >>> from pyhealth.tasks.sleep_staging import multi_epoch_multi_modal_sleep_staging_sleepedf_fn
        >>> modality = ['EEG Fpz-Cz','EOG horizontal']
        >>> num_epoch_seq = 5
        >>> sleepedf_task_ds = sleepedf_ds.set_task(lambda x: multi_epoch_multi_modal_sleep_staging_sleepedf_fn(x,modality = modality,num_epoch_seq = num_epoch_seq))
        >>> from pyhealth.datasets.splitter import split_by_patient
        >>> from pyhealth.datasets import split_by_patient, get_dataloader
        >>> # data split
        >>> train_dataset, val_dataset, test_dataset = split_by_patient(sleepedf_task_ds, [0.34, 0.33, 0.33])
        >>> # create dataloaders (they are <torch.data.DataLoader> object)
        >>> train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
        >>> val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
        >>> test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

        
        >>> from pyhealth.models import Seq_Cross_Modal_Transformer_PyHealth
        >>> model = Seq_Cross_Modal_Transformer_PyHealth( dataset= sleepedf_task_ds, 
        ...                                    feature_keys= ['signal'], 
        ...                                    label_key= ['label'], 
        ...                                    mode= 'multiclass', 
        ...                                    d_model = 128,
        ...                                    num_epoch_seq = num_epoch_seq,
        ...                                    dim_feedforward=512,
        ...                                    window_size = 50,
        ...                                    num_classes = 6,).to("cuda:0")
        >>> # data batch
        >>> data_batch = next(iter(train_loader))
        >>> # try the model
        >>> ret = model(**data_batch)
        >>> print(ret)
        >>> #try loss backward
        >>> ret["loss"].backward()

    '''

    def __init__(self, 
                 dataset: BaseSignalDataset, 
                 feature_keys: List[str], 
                 label_key: str, 
                 mode: str, 
                 d_model = 128,
                 num_epoch_seq = 5,
                 dim_feedforward=512,
                 window_size = 50,
                 num_classes = 6,
                 **kwargs):
        super().__init__(dataset, feature_keys, label_key, mode)

    
        self.num_epoch_seq = num_epoch_seq
        self.num_classes = num_classes
        
        self.seq_cross_modal_transformer = Seq_Cross_Modal_Transformer_Network(d_model = d_model,
                                                                                num_epoch_seq = num_epoch_seq,
                                                                                dim_feedforward=dim_feedforward,
                                                                                window_size = window_size,
                                                                                num_classes = num_classes)
        
    def get_sliced_seq(self, signal,epoch_seconds = 30,sample_rate = 100):
        '''
        Slice the signal into 30s epochs
        Args:
            signal: Tensor, input signal
            epoch_seconds: int, length of the epoch
            sample_rate: int, sampling rate of the signal
        Returns:
            Tensor, sliced signal
        '''
        sample_length = sample_rate * epoch_seconds
        sliced_seq = []
        for slice_index in range(signal.shape[-1] // sample_length):
            sliced_seq.append(signal[:,:,slice_index * sample_length : (slice_index + 1) * sample_length])
        return torch.cat(sliced_seq,dim = 1)
    
    def forward(self,**kwargs):
        '''
        Args:
            kwargs: dict, input data
        Returns:
            results: dict, model outputs
        '''
        signal = torch.tensor(kwargs[self.feature_keys[0]],device=self.device)
        y_true = torch.tensor(np.array(kwargs[self.label_key[0]]),device=self.device) # (B, num_epoch_seq)


        # Get EEG and EOG signals and slice them into 30s epochs
        eeg = signal[:,0,:].unsqueeze(1).float()
        eog = signal[:,1,:].unsqueeze(1).float()

        eeg = self.get_sliced_seq(eeg).unsqueeze(1)
        eog = self.get_sliced_seq(eog).unsqueeze(1) 
        
        # Get predictions from the model
        #logit_list: (B, num_epoch_seq, num_classes) 
        #ch_feat_list: (B, num_epoch_seq, ['EEG','EOG'], num time windows in a epoch, d_model)
        #cross_feat_list: (B, num_epoch_seq, ['EEG','EOG'], d_model)
        #seq: (B, num_epoch_seq, d_model)
        logit_list, ch_feat_list, cross_feat_list, seq = self.seq_cross_modal_transformer(eeg, eog)  
        
        

        # Prepare results
        loss = 0
        for i in range(self.num_epoch_seq):
            loss += self.get_loss_function()(logit_list[:,i,:], y_true[:, i])
            logit_list[:,i,:] = self.prepare_y_prob(logit_list[:,i,:])


        # Reshape the y_true to (batch_size * num_epoch_seq, 1)
        y_true = y_true.reshape(-1,1)

        # Reshape the logit_list to (batch_size * num_epoch_seq, num_classes)
        logit_list = logit_list.reshape(-1, self.num_classes)

        results = {
            'loss': loss,
            'y_prob': logit_list,
            'y_true': y_true,
            'channel_features': ch_feat_list, 
            'cross_modal_features': cross_feat_list,
            'sequence_features' :seq
        }

        if kwargs.get("embed", False):
            results["embed"] = seq

        return results
    

# Function and interative visulaization for intrepreting the Cross-Modal Transformer

def atten_interpret(q, k):
  atten_weights = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
  return atten_weights

def interpret_cmt(trainer,test_loader,num_epoch_seq,num_classes):
    '''
    Interpret the predictions from the Cross-Modal Transformer as intra-modal, cross-modal 
    and inter-epoch relationships

    Args:
        trainer: trainer object
        test_loader: test data loader
        num_epoch_seq: int, number of epochs in a sequence
        num_classes: int, number of sleep stages
    Returns:
        test_data: dict, test data with predictions and attention scores
                    {'patient_id': patient_id,
                    'record_id': record_id,
                    'signal': signal,
                    'label': label,
                    'y_prob': y_prob (predictions),
                    'seq_attn': sequence level attention scores (Shape: (B, num_epoch_seq, num_epoch_seq)),
                    'cross_attn': cross-modal attention scores (Shape: (B, num_epoch_seq,2)),
                    'eeg_attn': EEG attention scores (Shape: (B, num_epoch_seq, num_time_windows_in_a_epoch)),
                    'eog_attn': EOG attention scores (Shape: (B, num_epoch_seq, num_time_windows_in_a_epoch))
                    }
    '''
    y_true, y_prob, loss,feat = trainer.inference(test_loader,additional_outputs=['channel_features','cross_modal_features','sequence_features'])
    
    # ch_feat, cross_feat, seq_feat
    ch_feat = feat['channel_features']
    cross_feat = feat['cross_modal_features']
    seq_feat = feat['sequence_features']


    #reshape y_true and y_prob back to original shape
    y_true = y_true.reshape(-1,num_epoch_seq)  # (B,num_epoch_seq)
    y_prob = y_prob.reshape(-1,num_epoch_seq,num_classes) # (B,num_epoch_seq,num_classes)

    # load each sample from test_loader
    test_data = {'patient_id':[],'record_id':[],'signal':[],'label':[]}
    for batch in test_loader:
        patien_id = batch['patient_id']
        signal = batch['signal']
        label = batch['label']
        record_id = batch['record_id']
        
        for i in range(len(patien_id)):
            test_data['patient_id'].append(patien_id[i])
            test_data['record_id'].append(record_id[i])
            test_data['signal'].append(signal[i])
            test_data['label'].append(label[i])

    print('Test data statistics:')
    print(f'Number of samples: {len(test_data["patient_id"])}')


    test_data['y_prob'] = y_prob

    # Get sequence level attention scores
    seq_attn = []
    cross_attn = []
    eeg_attn = []
    eog_attn = []
    for i in range(len(test_data['patient_id'])):
        
        seq_epoch_attn = []
        cross_epoch_attn = []
        eeg_epoch_attn = []
        eog_epoch_attn = []
        for j in range(num_epoch_seq):
            seq_epoch_attn.append(atten_interpret(torch.tensor(seq_feat[i,j,:]),torch.tensor(seq_feat[i])).detach().numpy())
            cross_epoch_attn.append(atten_interpret(torch.tensor(seq_feat[i][j]),torch.tensor(cross_feat[i][j])).detach().numpy())
            eeg_epoch_attn.append(atten_interpret(torch.tensor(seq_feat[i][j]),torch.tensor(ch_feat[i][j][0])).detach().numpy())
            eog_epoch_attn.append(atten_interpret(torch.tensor(seq_feat[i][j]),torch.tensor(ch_feat[i][j][1])).detach().numpy())

        
        seq_attn.append(seq_epoch_attn)
        cross_attn.append(cross_epoch_attn)
        eeg_attn.append(eeg_epoch_attn)
        eog_attn.append(eog_epoch_attn)


    test_data['seq_attn'] = np.array(seq_attn)
    test_data['cross_attn'] = np.array(cross_attn)
    test_data['eeg_attn'] = np.array(eeg_attn)
    test_data['eog_attn'] = np.array(eog_attn)
    
    print(f'Sequence level attention scores shape: {test_data["seq_attn"].shape}')
    print(f'Cross-modal attention scores shape: {test_data["cross_attn"].shape}')
    print(f'EEG attention scores shape: {test_data["eeg_attn"].shape}')
    print(f'EOG attention scores shape: {test_data["eog_attn"].shape}')


    return test_data

def plot_signal_attentions(x,y,dydx,axs,axs_no,title_font_size,signal_type = "EEG"):
    '''
    Plot the attention scores on the EEG and EOG signals
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='Reds', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs[axs_no[0]][axs_no[1]].add_collection(lc)
    axs[axs_no[0]][axs_no[1]].set_xlabel(f'{signal_type}',fontsize = title_font_size)
    axs[axs_no[0]][axs_no[1]].xaxis.set_tick_params(labelbottom=False)
    axs[axs_no[0]][axs_no[1]].yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    axs[axs_no[0]][axs_no[1]].set_xticks([])
    axs[axs_no[0]][axs_no[1]].set_yticks([])
    axs[axs_no[0]][axs_no[1]].set_xlim(x.min(), x.max())
    axs[axs_no[0]][axs_no[1]].set_ylim(y.min()-0.2,y.max()+0.2)
    line.colorbar = plt.colorbar(line, ax=axs[axs_no[0]][axs_no[1]],format='',ticks=[])
    

def plot_interpret_cmt(test_data,data_no):
    '''
    Plot and visualize the interpretions for each sequence of epochs
    Args:
        test_data: dict, test data with predictions and attention scores
        data_no: int, index of the sample to visualize
    returns:
        Figure
    '''
    num_epoch_seq = test_data['y_prob'].shape[1]
    # print(num_epoch_seq)

    record_id = test_data['record_id'][data_no]
    seq_attn = test_data['seq_attn'][data_no]
    cross_attn = test_data['cross_attn'][data_no]
    eeg_attn = test_data['eeg_attn'][data_no]
    eog_attn = test_data['eog_attn'][data_no]
    eeg = test_data['signal'][data_no][0]
    eog = test_data['signal'][data_no][1]
    y_true = test_data['label'][data_no]
    y_prob = test_data['y_prob'][data_no]

    all_labels = ["Sleep stage W",
            "Sleep stage 1",
            "Sleep stage 2",
            "Sleep stage 3",
            "Sleep stage 4",
            "Sleep stage R"]

    print('True labels and Predictions')
    print(f'True labels: {[all_labels[i] for i in y_true]}')
    print(f'Predictions: {[all_labels[i] for i in y_prob.argmax(axis = -1)]}')

    epoch_len = eeg.shape[0]//num_epoch_seq

    fig, axs = plt.subplots(num_epoch_seq, 4,figsize=(50, 5*num_epoch_seq),gridspec_kw={'width_ratios': [1,1,5,5]})
    
    title_font_size = fig.dpi * 0.2
    label_font_size = fig.dpi * 0.1
    for i in range(num_epoch_seq):

            # Plotting inter-epoch attention ##############################
            rgba_colors = np.zeros((num_epoch_seq,4))
            rgba_colors[:,0]=0#value of red intensity divided by 256
            rgba_colors[i,0]=0.4#value of red intensity divided by 256
            rgba_colors[:,1]=0  #value of green intensity divided by 256
            rgba_colors[:,2]=0.4  #value of blue intensity divided by 256
            rgba_colors[i,2]=0
            rgba_colors[:,-1]= seq_attn[i]/seq_attn[i].max()
            axs[i][0].bar(np.arange(1,num_epoch_seq+1), seq_attn[i]/seq_attn[i].max(),#/seq_attn[i].max(),# color ='blue',
                    color =rgba_colors,align='center')
            # axs[i//5][i%5].set_title('')
            axs[i][0].tick_params(axis='x' ,labelsize=label_font_size)
            axs[i][0].tick_params(axis='y' ,labelsize=label_font_size)
            axs[i][0].set_xlabel('Epochs',fontsize = title_font_size)
            yticks = axs[i][0].yaxis.get_major_ticks()
            yticks[0].label1.set_visible(False)


            # Plotting cross-modal attention ##############################
            rgba_colors = np.zeros((2,4))
            rgba_colors[:,0]=0.4 #value of red intensity divided by 256
            rgba_colors[:,1]=0  #value of green intensity divided by 256
            rgba_colors[:,2]=0 #value of blue intensity divided by 256
            rgba_colors[:,-1]= cross_attn[i]
            axs[i][1].bar(['EEG','EOG'], cross_attn[i],# color ='red',
                    color =rgba_colors,align='center')
            axs[i][1].tick_params(axis='x',labelsize=label_font_size)
            axs[i][1].tick_params(axis='y',labelsize=label_font_size)
            axs[i][1].set_ylim(0,1.02)
            axs[i][1].set_xlabel('Signal',fontsize = title_font_size)


            # # Plotting EEG attention ##############################
            eeg_atten_epoch = eeg_attn[i]
            eeg_epoch = eeg[i*epoch_len:(i+1)*epoch_len]
            #upsample eeg attention to match eeg_epoch
            eeg_atten_epoch = np.repeat(eeg_atten_epoch,epoch_len//eeg_atten_epoch.shape[0])
            
            t1 = np.arange(0,epoch_len,1)
            plot_signal_attentions(t1,eeg_epoch,eeg_atten_epoch,axs,[i,2],title_font_size,signal_type = "EEG")
            
            
            # # Plotting EOG attention ##############################
            eog_atten_epoch = eog_attn[i]
            eog_epoch = eog[i*epoch_len:(i+1)*epoch_len]
            #upsample eog attention to match eog_epoch
            eog_atten_epoch = np.repeat(eog_atten_epoch,epoch_len//eog_atten_epoch.shape[0])
            plot_signal_attentions(t1,eog_epoch,eog_atten_epoch,axs,[i,3],title_font_size,signal_type = "EOG")

    time = [int(record_id.split('-')[1].split('_')[i]) for i in range(num_epoch_seq)]
    # plt.subplots_adjust(wspace=0.2)
    fig.suptitle('Interpretation for patient '+str(record_id.split('-')[0])+' for 30s epochs '+f'{time}',fontsize = title_font_size*2)
    plt.show()

def interactive_plot_cmt(test_data):
    '''
    Interactive plot for interpreting the Cross-Modal Transformer
    Args:
        test_data: dict, test data with predictions and attention scores
    Returns:
        interactive_plot: interactive plot
    '''
    select_data = IntSlider(min=0, max=len(test_data['patient_id'])-1, step=1, value=0, description='Data Index:')
    # plot_interpret_cmt(test_data,0)

    def update_plot(data_no):
        display.clear_output(wait=True)
        plot_interpret_cmt(test_data,data_no)

    interactive_plot = interactive(update_plot, data_no=select_data)


    return interactive_plot


if __name__ == "__main__":

    from pyhealth.datasets import SleepEDFDataset
    import subprocess
    urls = ["https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4001E0-PSG.edf",
            "https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4001EC-Hypnogram.edf",
            "https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4002E0-PSG.edf",
            "https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4002EC-Hypnogram.edf",
            "https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4011E0-PSG.edf",
            "https://storage.googleapis.com/pyhealth/sleepedf-sample/SC4011EH-Hypnogram.edf"]

    try:
        for url in urls:
            subprocess.run(["wget", "-r", url])
        print("File downloaded successfully.")
    except Exception as e:
        print(f"Error: {e}")

    sleepedf_ds = SleepEDFDataset(
        root="./storage.googleapis.com/pyhealth/sleepedf-sample",
        refresh_cache=True)
    
    from pyhealth.tasks.sleep_staging import multi_epoch_multi_modal_sleep_staging_sleepedf_fn
    modality = ['EEG Fpz-Cz','EOG horizontal']
    num_epoch_seq = 5
    sleepedf_task_ds = sleepedf_ds.set_task(lambda x: multi_epoch_multi_modal_sleep_staging_sleepedf_fn(x,modality = modality,num_epoch_seq = num_epoch_seq))
    
    from pyhealth.datasets import get_dataloader
    train_loader = get_dataloader(sleepedf_task_ds, batch_size=32, shuffle=True)
    from pyhealth.models import Seq_Cross_Modal_Transformer_PyHealth
    model = Seq_Cross_Modal_Transformer_PyHealth( dataset= sleepedf_task_ds, 
                                        feature_keys= ['signal'], 
                                        label_key= ['label'], 
                                       mode= 'multiclass', 
                                       d_model = 128,
                                       num_epoch_seq = num_epoch_seq,
                                        dim_feedforward=512,
                                        window_size = 50,
                                         num_classes = 6,).to("cuda:0")
    # data batch
    data_batch = next(iter(train_loader))
    # try the model
    ret = model(**data_batch)
    print(ret)
    #try loss backward
    ret["loss"].backward()



