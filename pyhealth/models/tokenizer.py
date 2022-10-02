from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


def to_index(sequence, vocab, prefix="", suffix=""):
    """convert code to index"""
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    @classmethod
    def build_vocabulary(cls, tokens):
        vocabulary = cls()
        for token in tokens:
            vocabulary.add_word(token)
        return vocabulary

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Tokenizer:
    def __init__(self, tokens):
        tokens = sorted(tokens)
        self.vocabulary = Vocabulary.build_vocabulary(tokens)

    def get_vocabulary_size(self):
        return len(self.vocabulary)

    def batch_tokenize(self, batch):
        """tokenize a batch of data
        INPUT
            batch: [up_to_visit1, up_to_visit2, ...]
                - visit1: [[code list of visit0], [code list of visit1]]
                - visit2: [[code list of visit0], [code list of visit1], [code list of visit2]]
                - ...
        OUTPUT
            tensor: (#batch, #max_visit, #max_code)
        """
        N_pat, N_visit, N_code = len(batch), 0, 0
        for sample in batch:
            N_visit = max(N_visit, len(sample))
            for visit in sample:
                N_code = max(N_code, len(visit))

        tensor = torch.zeros(N_pat, N_visit, N_code, dtype=torch.long)
        for i, sample in enumerate(batch):
            sample = self(sample)
            tensor[i, : sample.shape[0], : sample.shape[1]] = sample
        return tensor

    def __call__(self, text: List[List[str]], padding=True, prefix="", suffix=""):
        text_tokenized = []
        for sent in text:
            text_tokenized.append(
                torch.tensor(
                    to_index(sent, self.vocabulary, prefix=prefix, suffix=suffix),
                    dtype=torch.long,
                )
            )
        if padding:
            text_tokenized = pad_sequence(text_tokenized, batch_first=True)
        return text_tokenized


if __name__ == "__main__":
    from pyhealth.datasets.mimic3 import MIMIC3BaseDataset
    from pyhealth.data.dataset import DrugRecommendationDataset
    from torch.utils.data import DataLoader

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    task_taskset = DrugRecommendationDataset(base_dataset)
    conditions = task_taskset.all_tokens["conditions"]
    tokenizer = Tokenizer(conditions)
    print(tokenizer.get_vocabulary_size())
    data_loader = DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0])
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    print(tokenizer(batch[0]).shape)
