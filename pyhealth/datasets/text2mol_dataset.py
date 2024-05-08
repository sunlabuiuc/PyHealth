"""
File: text2mol_dataset.ipynb

PyHealth Text2Mol Dataset
"""

import numpy as np
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import os
import csv
import urllib.request
import requests
import io

INFO_MSG = """
The datasets contain the following information:

1. Training, Validation, and Test Datasets contain the following:
CID	mol2vec embedding	Description

2. ChEBI defintions substructure corpus contains the molecule token "sentences".
It is formatted:
cid: tokenid1 tokenid2 tokenid3 ... tokenidn

3. Token embedding dictionary is a dictionary mapping molecule tokens to their
embeddings.
"""

class Text2MolDataset(Dataset):
    """ This class defines a PyTorch Dataset (Text2MolDataset) for processing
    text and molecule data. It loads raw data from specified paths, tokenizes
    text using a BERT tokenizer, and prepares molecule embeddings. It also
    generates examples for training. Finally, it initializes data loaders for
    training, validation, and testing.
    """

    # Inner class for data generation
    class GenerateData():
        def __init__(self, path_raw_data, path_molecules, path_token_embs):
            # Paths to raw data, molecules, and token embeddings
            self.path_raw_data = path_raw_data
            self.path_molecules = path_molecules
            self.path_token_embs = path_token_embs

            # Dictionaries to store molecule sentences and tokens
            self.molecule_sentences = {}
            self.molecule_tokens = {}

            # Set to store all tokens and maximum molecule length
            self.total_tokens = set()
            self.max_mol_length = 0

            # Dictionaries to store descriptions and molecule vectors
            self.descriptions = {}
            self.mols = {}
            self.data_cids = []

            # Parameters for text processing
            self.text_trunc_length = 256
            self.batch_size = 32

            # Initialize text tokenizer
            self.prep_text_tokenizer()

            # Load molecule substructures
            self.load_substructures()

            # Store descriptions
            self.store_descriptions()

        def load_substructures(self):
            # Load molecule sentences and tokens from the given path
            with urllib.request.urlopen(self.path_molecules) as f:
                for line in f.readlines():
                    spl = line.decode('utf-8').split(":")
                    cid = spl[0]
                    tokens = spl[1].strip()
                    self.molecule_sentences[cid] = tokens
                    t = tokens.split()
                    self.total_tokens.update(t)
                    size = len(t)
                    if size > self.max_mol_length:
                        self.max_mol_length = size

            # Load token embeddings
            self.response = requests.get(self.path_token_embs)
            self.response.raise_for_status()
            self.token_embs = np.load(io.BytesIO(self.response.content), allow_pickle=True)[()]

        def prep_text_tokenizer(self):
            # Initialize text tokenizer
            self.text_tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")

        def store_descriptions(self):
            # Store descriptions and molecule vectors
            with urllib.request.urlopen(self.path_raw_data) as response:
                lines = [line.decode('utf-8') for line in response.readlines()]
                reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames=['cid', 'mol2vec', 'desc'])
                for n, line in enumerate(reader):
                    self.descriptions[line['cid']] = line['desc']
                    self.mols[line['cid']] = line['mol2vec']
                    self.data_cids.append(line['cid'])

        def generate_examples_train(self):
            """Yields examples."""
            np.random.shuffle(self.data_cids)

            # Generate examples for training
            for cid in self.data_cids:
                text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, max_length=self.text_trunc_length,
                                                  padding='max_length', return_tensors='np')

                yield {
                    'cid': cid,
                    'input': {
                        'text': {
                            'input_ids': text_input['input_ids'].squeeze(),
                            'attention_mask': text_input['attention_mask'].squeeze(),
                        },
                        'molecule': {
                            'mol2vec': np.fromstring(self.mols[cid], sep=" "),
                            'cid': cid
                        },
                    },
                }

    # Initialization of Text2MolDataset
    def __init__(self, root: str, dataset_type: str):
        'Initialization'
        # Determine the path to raw data based on dataset type
        self.dataset_type = dataset_type
        if dataset_type == 'test':
            self.path_raw_data = os.path.join(root, 'test.txt')
        elif dataset_type == 'validation':
            self.path_raw_data = os.path.join(root, 'val.txt')
        else:
            self.path_raw_data = os.path.join(root, 'training.txt')

        self.dataset_name = dataset_type
        self.path = root

        # Paths to other data files
        self.substrcture = 'ChEBI_defintions_substructure_corpus.cp'
        self.token_embedding = 'token_embedding_dict.npy'
        path_molecules = os.path.join(root, self.substrcture)
        path_token_embs = os.path.join(root, self.token_embedding)

        # Initialize data generation
        self.gen = self.GenerateData(self.path_raw_data, path_molecules, path_token_embs)
        self.it = iter(self.gen.generate_examples_train())
        self.length = len(self.gen.data_cids)

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one sample of data'

        try:
            ex = next(self.it)
        except StopIteration:
            self.it = iter(self.gen.generate_examples_train())
            ex = next(self.it)

        X = ex['input']
        y = 1  # Placeholder for target variable (not used in this example)

        return X, y

    @staticmethod
    def info():
        """Prints the dataset information."""
        print(INFO_MSG)

    def stat(self) -> str:
            """Returns statistics of the dataset."""
            lines = list()
            lines.append("")
            lines.append(f"Statistics of Dataset:")
            lines.append(f"\t- Dataset Name: {self.dataset_name}")
            lines.append(f"\t- Number of samples: {self.length}")
            lines.append(f"\t- Number of Text Descriptions: {len(self.gen.descriptions)}")
            lines.append(f"\t- Max Molecule Length: {self.gen.max_mol_length}")
            lines.append(f"\t- Number of Tokens: {len(self.gen.total_tokens)}")
            lines.append(f"\t- Number of Molecule Sentences: {len(self.gen.molecule_sentences)}")
            lines.append(f"\t- Number of Molecules: {len(self.gen.mols)}")
            lines.append(f"\t- Text Truncate Length: {self.gen.text_trunc_length}")
            lines.append(f"\t- Batch Size: {self.gen.batch_size}")
            lines.append(f"\t- Dataset Path: {self.path}")
            lines.append(f"\t- Substrcuture File: {self.substrcture}")
            lines.append(f"\t- Token Embedding File: {self.token_embedding}")
            lines.append("")
            print("\n".join(lines))
            return "\n".join(lines)

# Main block
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Constants
    BATCH_SIZE = 32

    # Initialize datasets for training, validation, and testing
    training_dataset = Text2MolDataset(root="https://pyhealth4text2mol.blob.core.windows.net/text2mol",
                                        dataset_type="training")
    validation_dataset = Text2MolDataset(root="https://pyhealth4text2mol.blob.core.windows.net/text2mol",
                                        dataset_type="validation")
    test_dataset = Text2MolDataset(root="https://pyhealth4text2mol.blob.core.windows.net/text2mol",
                                    dataset_type="test")

    training_dataset.info()
    training_dataset.stat()
    validation_dataset.stat()
    test_dataset.stat()

    # Parameters for data loaders
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 1}

    # Create data loaders for training, validation, and testing
    training_generator = DataLoader(training_dataset, **params)
    validation_generator = DataLoader(validation_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

### Output:
# The datasets contain the following informaiton:

# 1. Training, Validation, and Test Datasets contain following information:
# CID	mol2vec embedding	Description

# 2. ChEBI defintions substructure corpus contains the molecule token "sentences".
# It is formatted:
# cid: tokenid1 tokenid2 tokenid3 ... tokenidn

# 3. Token embedding dictionary is a dictionary mapping molecule tokens to their 
# embeddings.


# Statistics of Dataset:
# 	- Dataset Name: training
# 	- Number of samples: 26408
# 	- Number of Text Descriptions: 26408
# 	- Max Molecule Length: 1148
# 	- Number of Tokens: 9447
# 	- Number of Molecule Sentences: 102980
# 	- Number of Molecules: 26408
# 	- Text Truncate Length: 256
# 	- Batch Size: 32
# 	- Dataset Path: https://pyhealth4text2mol.blob.core.windows.net/text2mol
# 	- Substrcuture File: ChEBI_defintions_substructure_corpus.cp
# 	- Token Embedding File: token_embedding_dict.npy


# Statistics of Dataset:
# 	- Dataset Name: validation
# 	- Number of samples: 3301
# 	- Number of Text Descriptions: 3301
# 	- Max Molecule Length: 1148
# 	- Number of Tokens: 9447
# 	- Number of Molecule Sentences: 102980
# 	- Number of Molecules: 3301
# 	- Text Truncate Length: 256
# 	- Batch Size: 32
# 	- Dataset Path: https://pyhealth4text2mol.blob.core.windows.net/text2mol
# 	- Substrcuture File: ChEBI_defintions_substructure_corpus.cp
# 	- Token Embedding File: token_embedding_dict.npy


# Statistics of Dataset:
# 	- Dataset Name: test
# 	- Number of samples: 3301
# 	- Number of Text Descriptions: 3301
# 	- Max Molecule Length: 1148
# 	- Number of Tokens: 9447
# 	- Number of Molecule Sentences: 102980
# 	- Number of Molecules: 3301
# 	- Text Truncate Length: 256
# 	- Batch Size: 32
# 	- Dataset Path: https://pyhealth4text2mol.blob.core.windows.net/text2mol
# 	- Substrcuture File: ChEBI_defintions_substructure_corpus.cp
# 	- Token Embedding File: token_embedding_dict.npy