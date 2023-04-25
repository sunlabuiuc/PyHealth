from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from pyhealth.datasets import SampleImageCaptionDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets.utils import flatten_list

class WordSATEncoder(nn.Module):
    """ SAT CNN(Densenet121) Encoder model"""
    def __init__(self):
        super().__init__()
        self.densenet121 = models.densenet121(weights='DEFAULT')
        self.densenet121.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.
        Extract fixed-length feature vectors from the input image.

        Args:
            x: A tensor of transfomed image of size
                [batch_size,3,512,512]
        Return:
            x: A tensor of image feature vectors of size
                [batch_size,1024,16,16]
        """
        x = self.densenet121.features(x)
        x = F.relu(x)
        return x

class WordSATAttention(nn.Module):
    """SAT Attention Module

    Computes a set of attention weights based on the current hidden state of
    the RNN and the feature vectors from the CNN, which are then used to
    compute a weighted average of the feature vectors.

    Args:
        k_size: key vector size
        v_size: value vector size
        affine_dim: affine dimension. Default is 512
    """
    def __init__(
        self,
        k_size: int,
        v_size: int,
        affine_dim: int =512):
        super().__init__()
        self.affine_k = nn.Linear(k_size, affine_dim, bias=False)
        self.affine_v = nn.Linear(v_size, affine_dim, bias=False)
        self.affine = nn.Linear(affine_dim, 1, bias=False)

    def forward(
            self,
            k: torch.Tensor,
            v: torch.Tensor) -> (torch.Tensor,torch.Tensor):
        """Forward propagation

        Args:
            k: a tensor of size [batch_size, hidden_dim]
            v: a tensor of size [batch_size, spatial_size, hidden_dim]

        Returns:
            context: a tensor of size [batch_size, feature_dim]
            alpha: a tensor of size [batch_size, spatial_size]
        """
        content_v = self.affine_k(k).unsqueeze(1) + self.affine_v(v)
        # z: batch size x spatial size
        z = self.affine(torch.tanh(content_v)).squeeze(2)
        alpha = torch.softmax(z, dim=1)
        context = (v * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha

class WordSATDecoder(nn.Module):
    """ Word SAT decoder model for one sentence

    An LSTM based model that takes as input the attention-weighted feature
    vector and generates a sequence of words, one at a time.

    Args:
        attention: attention module instance
        vocab_size: vocabulary size
        n_encoder_inputs: number of image inputs given to the encoder
        feature_dim: encoder output feature dimension
        embedding_dim: decoder embedding dimension
        hidden_dim: LSTM hidden dimension
        dropout: dropout rate between [0,1]
    """
    def __init__(
        self,
        attention: object,
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

        self.attend = attention
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.init_h = nn.Linear(self.n_encoder_inputs * self.feature_dim,
                                self.hidden_dim)
        self.init_c = nn.Linear(self.n_encoder_inputs * feature_dim,
                                hidden_dim)
        self.lstmcell = nn.LSTMCell(self.embedding_dim +
                                    self.n_encoder_inputs * feature_dim,
                                    hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            cnn_features: List[torch.Tensor],
            captions: List[torch.Tensor] = None,
            max_len: int = 100) -> torch.Tensor:

        """Forward propagation

        Args:
            cnn_features: a list of tensors where each tensor is of
                size [batch_size, feature_dim, spatial_size].
            captions: a list of tensors.
            max_len: maximum length of training or generated caption

        Returns:
            logits: a tensor

        """
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
                                            self.vocab_size),dtype=torch.float)
        # Training phase
        if captions is not None:
            embeddings = self.embed(captions)
            for t in range(seq_len):
                contexts = [self.attend(h, cnn_feat_t)[0]
                            for cnn_feat_t in cnn_feats_t]
                context = torch.cat(contexts, dim=1)
                h, c = self.lstmcell(torch.cat((embeddings[:, t], context),
                                                dim=1),
                                     (h, c))
                logits[:, t] = self.fc(self.dropout(h))

            return logits

        # Evaluation/Inference phase
        else:
            x_t = cnn_features[0].new_full((batch_size,), 1, dtype=torch.long)
            for t in range(seq_len):
                embedding = self.embed(x_t)
                contexts = [self.attend(h, cnn_feat_t)[0]
                            for cnn_feat_t in cnn_feats_t]
                context = torch.cat(contexts, dim=1)
                h, c = self.lstmcell(torch.cat((embedding, context), dim=1),
                                                (h, c))
                logit = self.fc(h)
                x_t = logit.argmax(dim=1)
                logits[:, t] = logit

            return logits.argmax(dim=2)

class WordSAT(BaseModel):
    """Show Attend & Tell model, treating entire caption as one sentence.
    This model is based on Show, Attend, and Tell (SAT) paper. The model uses
    convolutional neural networks (CNNs) to encode the image and a
    recurrent neural network (RNN) with an attention mechanism to generate
    the corresponding caption.

    The model consists of three main components:
        - Encoder: The encoder is a CNN that extracts a fixed-length feature
            vectors from the input image.
        - Attention Mechanism: The attention mechanism is used to select
            relevant parts of the image at each time step of the RNN, to
            generate the next word in the caption.
        - Decoder: The decoder is a language model implemented as an RNN that
            takes as input the attention-weighted feature vector and generates
            a sequence of words, one at a time.

    Args:
        dataset: the dataset to train the model.
        n_input_images: number of images passed as input to each sample.
        label_key: key in the samples to use as label (e.g., "caption").
        tokenizer: pyhealth tokenizer instance created using sample texts.
        encoder_pretrained_weights: pretrained state dictionary for encoder.
            Default is None.
        encoder_freeze_weights: freeze encoder weights so that they are not
            updated during training. This is useful when the encoder is trained
            separately as a classifier. Default is True.
        decoder_embed_dim: decoder embedding dimesion. Default is 256.
        decoder_hidden_dim: decoder hidden state dimension. Default is 512.
        decoder_feaure_dim: decoder input cell state dimension.
            Default is 1024
        decoder_dropout: decoder dropout rate between [0,1]. Default is 0.5
        attention_affine_dim: output dimension of affine layer in attention.
            Default is 512.
        save_generated_caption: save the generated caption during training.
            This is used for evaluating the quality of generated captions.
            Default is False.
    """

    def __init__(
        self,
        dataset: SampleImageCaptionDataset,
        n_input_images: int,
        label_key: str,
        tokenizer: Tokenizer,
        encoder_pretrained_weights: Dict[str,float] = None,
        encoder_freeze_weights: bool = True,
        decoder_embed_dim: int = 256,
        decoder_hidden_dim: int = 512,
        decoder_feature_dim: int = 1024,
        decoder_dropout: float = 0.5,
        attention_affine_dim: int = 512,
        save_generated_caption: bool = False,
        **kwargs
    ):
        super(WordSAT, self).__init__(
            dataset=dataset,
            feature_keys=[f'image_{i+1}' for i in range(n_input_images)],
            label_key=label_key,
            mode="sequence",
            save_generated_caption = save_generated_caption
        )
        self.n_input_images = n_input_images
        self.save_generated_caption = save_generated_caption

        # Encoder component
        self.encoder = WordSATEncoder()
        if encoder_pretrained_weights:
            print(f'Loading encoder pretrained model')
            self.encoder.load_state_dict(encoder_pretrained_weights)
        if encoder_freeze_weights:
            self.encoder.eval()

        # Attention component
        self.attention = WordSATAttention(decoder_hidden_dim,
                                          decoder_feature_dim,
                                          attention_affine_dim)

        # Decoder component
        self.caption_tokenizer = tokenizer
        vocab_size = self.caption_tokenizer.get_vocabulary_size()
        self.decoder = WordSATDecoder(  self.attention,
                                        vocab_size,
                                        n_input_images,
                                        decoder_feature_dim,
                                        decoder_embed_dim,
                                        decoder_hidden_dim,
                                        decoder_dropout
                                    )

    def forward(self, decoder_maxlen:int = 100, **kwargs) -> Dict[str,str]:
        """Forward propagation.

        The features `kwargs[self.feature_keys]` is a list of feature keys
        associated to every input image.

        The label `kwargs[self.label_key]` is a key of the report caption
        for each patient.

        Args:
            decder_maxlen: maximum caption length used during training or
                generated during inference. Default is 100.
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_generated: a dictionary with following key
                    - "patient_id": list of text representing the generated
                                    text.
                    e.g.
                        {123: ["generated text"], 456: ["generated text"]}
                y_true: a dictionary with following key
                    - "patient_id": list of text representing the true text.
                    e.g.
                        {123: ["true text"], 456: ["true text"]}
        """
        # Initialize the output
        output = {"loss": None,"y_generated": [""],"y_true": [""]}

        # Get list of patient_ids
        patient_ids = kwargs["patient_id"]

        # Get CNN features
        images = self._prepare_batch_images(kwargs)
        cnn_features = [self.encoder(image.to(self.device))
                        for image in images]

        if self.training:
            # Get caption as indicies and corresponding masks
            captions,masks=self._prepare_batch_captions(kwargs[self.label_key])

            # Perform predictions
            logits = self.decoder(cnn_features, captions[:,:-1],
                                  decoder_maxlen)
            logits = logits.permute(0, 2, 1).contiguous()
            captions = captions[:, 1:].contiguous()
            masks = masks[:, 1:].contiguous()

            # Compute loss
            loss = self.get_loss_function()(logits, captions)
            loss = loss.masked_select(masks).mean()
            output["loss"] = loss

        with torch.no_grad():
            output["y_generated"] = self._forward_inference(patient_ids,
                                                            decoder_maxlen,
                                                            cnn_features)
            output["y_true"] = self._forward_ground_truths(patient_ids,
                                                        kwargs[self.label_key])
        return output

    def _prepare_batch_images(self,kwargs):
        """Prepare images for input.
        Args:
            kwargs: keyword arguments for the model.
        Returns:
            images: a list of input images represented as tensors. Every tensor
                in the list has shape [batch_size,3,image_size,image_size]
        """
        images = [torch.stack(kwargs[image_feature], 0)
                  for image_feature in self.feature_keys]

        return images

    def _prepare_batch_captions(
            self,
            captions:List[List[str]]
            ) -> (torch.Tensor,torch.Tensor):
        """Prepare caption idx for input.

        Args:
            captions: list of captions. Each caption is a list of list, where
                each list represents a sentence in the caption.
                Following is an example of a caption
                    [
                     ["<start>","first","sentence","."],
                     [".", "second", "sentence",".", "<end>"]
                    ]
        Returns:
            captions_idx: an int tensor
            masks: a bool tensor
        """

        # Combine all sentences in each caption to create a single sentence
        samples = []
        for caption in captions:
            tokens = []
            tokens.extend(flatten_list(caption))
            text = ' '.join(tokens).replace('. .','.')
            samples.append([text.split()])

        x = self.caption_tokenizer.batch_encode_3d(samples)
        captions_idx = torch.tensor(x, dtype=torch.long,
                                    device=self.device)
        masks = torch.sum(captions_idx,dim=1) !=0
        captions_idx = captions_idx.squeeze(1)

        return captions_idx,masks

    def _forward_inference(
            self,
            patient_ids: List[int],
            decoder_maxlen: int,
            cnn_features:List[torch.Tensor]) -> Dict[int,str]:
        """Forward propagation during inference

        Args:
            patient_ids: a list of patient ids
            decoder_maxlen: maximum length of generated caption
            cnn_features: a list of tensors

        Returns:
            generated_results: a dict with following keys
                - patient_id: int
                - generated_results: List[str]
        """
        generated_results = {}
        for idx, patient_id in enumerate(patient_ids):
            generated_results[patient_id] = [""]
            cnn_feature = [cnn_feat[idx].unsqueeze(0)
                            for cnn_feat in cnn_features]
            pred = self.decoder(cnn_feature, None, decoder_maxlen)[0]
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

    def _forward_ground_truths(
            self,
            patient_ids: List[int],
            captions:List[List[str]]) -> Dict[int,str]:
        """Forward propagation for ground truth

        Args:
            patient_ids: a list of patient ids
            cnn_features: a list of tensors

        Returns:
            ground_results: a dict with following keys
                - patient_id: int
                - generated_results: List[str]
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



