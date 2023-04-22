from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from pyhealth.datasets import SampleImageCaptionDataset
from pyhealth.models import BaseModel
from pyhealth.datasets.utils import flatten_list

class WordSATEncoder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.densenet121 = models.densenet121(weights='DEFAULT')
        self.densenet121.classifier = nn.Identity()

    def forward(self, x):
        x = self.densenet121.features(x)
        x = F.relu(x)
        return x

class Attention(nn.Module):
    """
    """
    def __init__(self, k_size, v_size, affine_size=512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_size, bias=False)
        self.affine_v = nn.Linear(v_size, affine_size, bias=False)
        self.affine = nn.Linear(affine_size, 1, bias=False)

    def forward(self, k, v):
        # k: batch size x hidden size
        # v: batch size x spatial size x hidden size
        # z: batch size x spatial size
        # TODO other ways of attention?
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class WordSATDecoder(nn.Module):
    """
    """
    def __init__(
        self, 
        vocab_size: int,
        n_encoder_inputs: int,
        feature_dim: int, 
        embedding_dim: int, 
        hidden_dim: int,
        dropout: int = 0.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_encoder_inputs = n_encoder_inputs
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.atten = Attention(self.hidden_dim, self.feature_dim)
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.init_h = nn.Linear(self.n_encoder_inputs * self.feature_dim,
                                self.hidden_dim)
        self.init_c = nn.Linear(self.n_encoder_inputs * feature_dim, hidden_dim)
        self.lstmcell = nn.LSTMCell(self.embedding_dim + 
                                    self.n_encoder_inputs * feature_dim, 
                                    hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, cnn_features, captions=None, max_len=100):
        batch_size = cnn_features[0].size(0)
        if captions is not None:
            seq_len = captions.size(1)
        else:
            seq_len = max_len

        cnn_feats_t = [ cnn_feat.view(batch_size, self.feature_dim, -1) \
                        .permute(0, 2, 1)
                        for cnn_feat in cnn_features ]                    
        global_feats = [cnn_feat.mean(dim=(2, 3)) for cnn_feat in cnn_features]
        
        h = self.init_h(torch.cat(global_feats, dim=1))
        c = self.init_c(torch.cat(global_feats, dim=1))

        logits = cnn_features[0].new_zeros((batch_size, 
                                            seq_len, 
                                            self.vocab_size), dtype=torch.float)

        if captions is not None:
            embeddings = self.embed(captions)
            for t in range(seq_len):
                contexts = [self.atten(h, cnn_feat_t)[0] 
                            for cnn_feat_t in cnn_feats_t]
                context = torch.cat(contexts, dim=1)
                h, c = self.lstmcell(torch.cat((embeddings[:, t], context), 
                                                dim=1),
                                     (h, c))
                logits[:, t] = self.fc(self.dropout(h))

            return logits

        else:
            x_t = cnn_features[0].new_full((batch_size,), 1, dtype=torch.long)
            for t in range(seq_len):
                embedding = self.embed(x_t)
                contexts = [self.atten(h, cnn_feat_t)[0] 
                            for cnn_feat_t in cnn_feats_t]
                context = torch.cat(contexts, dim=1)
                h, c = self.lstmcell(torch.cat((embedding, context), dim=1), 
                                                (h, c))
                logit  =self.fc(h)
                x_t = logit.argmax(dim=1)
                logits[:, t] = logit

            return logits.argmax(dim=2)

class WordSAT(BaseModel):
    """Word Show Attend & Tell model.
    Argument list of class
    """
   
    def __init__(
        self,
        dataset: SampleImageCaptionDataset,
        feature_keys: List[str],
        label_key: str,
        tokenizer: object,
        mode: str,
        encoder_pretrained_weights: object = None,
        encoder_freeze_weights: bool = True,
        decoder_maxlen: int = 100,
        decoder_embed_dim: int = 256,
        decoder_hidden_dim: int = 512,
        decoder_feature_dim: int = 1024,
        decoder_dropout: float = 0.5,
        save_generated_caption: bool = False,
        **kwargs
    ):
        super(WordSAT, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )

        self.encoder = WordSATEncoder()
        self.save_generated_caption = save_generated_caption

        if encoder_pretrained_weights:
            print(f'Loading encoder pretrained model')
            self.encoder.load_state_dict(encoder_pretrained_weights)

        if encoder_freeze_weights:
            self.encoder.eval()
        
        self.decoder_maxlen = decoder_maxlen
        self.caption_tokenizer = tokenizer
        vocab_size = self.caption_tokenizer.get_vocabulary_size()   
        self.decoder = WordSATDecoder(
                                        vocab_size,
                                        len(feature_keys),
                                        decoder_feature_dim, 
                                        decoder_embed_dim, 
                                        decoder_hidden_dim,
                                        decoder_dropout
                                    )

    def _prepare_batch_images(self,kwargs):
        """Prepare images for input.
        """
        print(self.n_input_image)
        
        return images

    def _prepare_batch_captions(self,captions):
        """Prepare caption for input.
        """
        samples = []
        for caption in captions:
            tokens = []
            tokens.extend(flatten_list(caption))
            text = ' '.join(tokens).replace('. .','.')
            samples.append([text.split()])
        #print(caption)
        x = self.caption_tokenizer.batch_encode_3d(samples)
        captions = torch.tensor(x, dtype=torch.long, device=self.device)
        masks = torch.sum(captions,dim=1) !=0
        captions = captions.squeeze(1)
        
        return captions,masks

    def forward(self, **kwargs):
        """Forward propagation.
        """
        patient_ids = kwargs['patient_id']

        image_features = [ feature for feature in self.feature_keys 
                            if 'image_' in feature]

        images = [ torch.stack(kwargs[image_feature], 0)
            for image_feature in image_features
        ]

        cnn_features = [self.encoder(image.to(self.device)) for image in images]
        output = {}
        if self.training:
            captions,masks = self._prepare_batch_captions(kwargs[self.label_key])
            logits = self.decoder(cnn_features, captions[:,:-1], 
                                    self.decoder_maxlen)
            logits = logits.permute(0, 2, 1).contiguous()
            captions = captions[:, 1:].contiguous()
            masks = masks[:, 1:].contiguous()

            loss = self.get_loss_function()(logits, captions)
            loss = loss.masked_select(masks).mean()

            output["loss"] = loss
        else:
            output["y_generated"] = self._forward_inference(patient_ids,
                                                            cnn_features
                                                        )
        output["y_true"] = self._forward_get_ground_truths(patient_ids,
                                                          kwargs[self.label_key]
                                                        )       
        return output

    def _forward_inference(self,patient_ids,cnn_features):
        """
        """
        generated_results = {}
        for idx, patient_id in enumerate(patient_ids):
            generated_results[patient_id] = [""]
            cnn_feature = [cnn_feat[idx].unsqueeze(0) 
                            for cnn_feat in cnn_features]
            pred = self.decoder(cnn_feature, None, self.decoder_maxlen)[0]
            pred = pred.detach().cpu()
            pred_tokens = self.caption_tokenizer \
                              .convert_indices_to_tokens(pred.tolist())
            generated_results[patient_id] = [""]
            words = []
            for token in pred_tokens:
                if token == '<start>' or token == '<pad>':
                    continue
                if token == '<end>':
                    break
                words.append(token)
                
            generated_results[patient_id][0] = " ".join(words)

        return generated_results

    def _forward_get_ground_truths(self,patient_ids,captions):
        """
        """
        ground_truths = {}
        for idx, caption in enumerate(captions):
            ground_truths[patient_ids[idx]] = [""]
            tokens = []
            tokens.extend(flatten_list(caption))
            ground_truths[patient_ids[idx]][0] = ' '.join(tokens) \
                                                 .replace('. .','.') \
                                                 .replace("<start>","") \
                                                 .replace("<end>","") \
                                                 .strip()

        return ground_truths



