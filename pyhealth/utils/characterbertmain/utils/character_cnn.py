# Functions are imported/adapted from AllenAI's AllenNLP library:
# https://github.com/allenai/allennlp/

""" Indexer functions for ELMo-style character embeddings. """
from typing import Dict, List, Callable, Any
import torch

PADDING_VALUE = 0


def _make_bos_eos(
        character: int,
        padding_character: int,
        beginning_of_word_character: int,
        end_of_word_character: int,
        max_word_length: int):

    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


def pad_sequence_to_length(
    sequence: List,
    desired_length: int,
    default_value: Callable[[], Any] = lambda: 0,
    padding_on_right: bool = True,
) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.
    The original list is not modified.
    """

    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


class CharacterMapper:
    """
    Maps individual tokens to sequences of character ids.
    """

    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>
    mask_character = 261  # <mask>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    mask_characters = _make_bos_eos(
        mask_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    pad_characters = [PADDING_VALUE - 1] * max_word_length

    bos_token = "[CLS]"
    eos_token = "[SEP]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"

    def __init__(self, tokens_to_add: Dict[str, int] = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = [CharacterMapper.padding_character] * CharacterMapper.max_word_length
            char_ids[0] = CharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = CharacterMapper.end_of_word_character
        elif word == CharacterMapper.bos_token:
            char_ids = CharacterMapper.beginning_of_sentence_characters
        elif word == CharacterMapper.eos_token:
            char_ids = CharacterMapper.end_of_sentence_characters
        elif word == CharacterMapper.mask_token:
            char_ids = CharacterMapper.mask_characters
        elif word == CharacterMapper.pad_token:
            char_ids = CharacterMapper.pad_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[
                : (CharacterMapper.max_word_length - 2)
            ]
            char_ids = [CharacterMapper.padding_character] * CharacterMapper.max_word_length
            char_ids[0] = CharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = CharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class CharacterIndexer:
    def __init__(self) -> None:
        self._mapper = CharacterMapper()

    def tokens_to_indices(self, tokens: List[str]) -> List[List[int]]:
        return [self._mapper.convert_word_to_char_ids(token) for token in tokens]

    def _default_value_for_padding(self):
        return [PADDING_VALUE] * CharacterMapper.max_word_length

    def as_padded_tensor(self, batch: List[List[str]], as_tensor=True, maxlen=None) -> torch.Tensor:
        if maxlen is None:
            maxlen = max(map(len, batch))
        batch_indices = [self.tokens_to_indices(tokens) for tokens in batch]
        padded_batch = [
            pad_sequence_to_length(
                indices, maxlen,
                default_value=self._default_value_for_padding)
            for indices in batch_indices
        ]
        if as_tensor:
            return torch.LongTensor(padded_batch)
        else:
            return padded_batch


if __name__ == "__main__":
    inputs = [
        '[CLS] hi [PAD] [SEP]'.split(),
        '[CLS] hello , my [MASK] is hicham [SEP]'.split()
    ]
    output = CharacterIndexer().as_padded_tensor(inputs)
    print('input:', inputs)
    print('output.shape:', output.shape)
    print('output:', output)
