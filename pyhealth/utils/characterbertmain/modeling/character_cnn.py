# Functions are imported/adapted from AllenAI's AllenNLP library:
# https://github.com/allenai/allennlp/

"""Defines the character embedding module (adapted from ELMo)"""

import json
from typing import Dict, Callable

import numpy
import torch

from pyhealth.utils.characterbertmain.utils.character_cnn import CharacterMapper, CharacterIndexer


class Highway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class CharacterCNN(torch.nn.Module):
    """
    Computes context insensitive token representations from each token's characters.
    """

    def __init__(self,
            output_dim: int = 768,
            requires_grad: bool = True) -> None:
        super().__init__()

        self._options = {
            'char_cnn': {
                'activation': 'relu',
                'filters': [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]
                    ],
                'n_highway': 2,
                'embedding': {'dim': 16},
                'n_characters': 262,
                'max_characters_per_token': 50
            }
        }
        self.output_dim = output_dim
        self.requires_grad = requires_grad

        self._init_weights()

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
            numpy.array(CharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
            numpy.array(CharacterMapper.end_of_sentence_characters) + 1
        )

    def _init_weights(self):
        self._init_char_embedding()
        self._init_cnn_weights()
        self._init_highway()
        self._init_projection()

    def _init_char_embedding(self):
        weights = numpy.zeros(
            (
                self._options["char_cnn"]["n_characters"] + 1,
                self._options["char_cnn"]["embedding"]["dim"]
            ),
            dtype="float32")
        weights[-1, :] *= 0.  # padding
        self._char_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _init_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num,
                kernel_size=width, bias=True)
            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad
            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)
        self._convolutions = convolutions

    def _init_highway(self):
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            self._highways._layers[k].weight.requires_grad = self.requires_grad
            self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _init_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        self._projection.weight.requires_grad = self.requires_grad
        self._projection.bias.requires_grad = self.requires_grad

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.
        Returns
        -------
        embeddings: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, embedding_dim)`` tensor with context
            insensitive token representations.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        #character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
        #    inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters
        #)
        character_ids_with_bos_eos, mask_with_bos_eos = inputs, mask

        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise Exception("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return token_embedding.view(batch_size, sequence_length, -1)


if __name__ == "__main__":

    model = CharacterCNN(output_dim=768)

    # Test with a batch of 2 sentences
    mapper = CharacterMapper()
    sentences = [
        '[CLS] hi , my name is Hicham [SEP]'.split(),
        '[CLS] hello Hicham [SEP]'.split()
    ]
    print('Input sequences:', sentences)

    indexer = CharacterIndexer()
    inputs = indexer.as_padded_tensor(sentences)
    print('Input shape:', inputs.shape)

    print('Output shape:', model.forward(inputs).shape)
