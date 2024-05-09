"""
File: gpm_dataset.py

PyHealth GPM Dataset
"""

from torch.utils.data import Dataset
import os
import requests
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from pyhealth.datasets.utils import MODULE_CACHE_PATH

# Default variables

AMINO_ACID_DICT = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "O": 16,
    "S": 17,
    "U": 18,
    "T": 19,
    "W": 20,
    "Y": 21,
    "V": 22,
}

SPECIES = {
    "eukaryotes": [
        "Anopheles gambiae",
        "Arabidopsis thaliana",
        "Aspergillus niger",
        "Bos taurus",
        "Caenorhabditis elegans",
        "Candida albicans",
        "Chlamydomonas reinhardtii",
        "Dictyostelium_discoideum",
        "Drosophila melanogaster",
        "Danio rerio",
        "Gallus gallus",
        "Homo sapiens",
        "Mus musculus",
        "Plasmodium falciparum",
        "Rattus norvegicus",
        "Schizosaccharomyces pombe",
        "Saccharomyces cerevisiae",
        "Trypanosoma_brucei",
    ],
    "prokaryotes": [
        "Bacillus anthracis",
        "Bacillus_subtilis_168",
        "Bacteroides thetaiotaomicron VPI-5482",
        "Escherichia coli K-12",
        "Halobacterium salinarum R1",
        "Helicobacter_pylori_26695",
        "Lactococcus lactis Il1403",
        "Legionella_pneumophila_Philadelphia_1",
        "Listeria_monocytogenes_EGD_e",
        "Mycobacterium tuberculosis",
        "Neisseria_meningitidis_MC58",
        "Pseudomonas_aeruginosa_PAO1",
        "Salmonella_enterica_serovar_Typhimurium_14028S",
        "Shewanella_oneidensis_MR_1",
        "Shigella dysenteriae",
        "Staphylococcus_aureus_JH1",
        "Streptococcus pyogenes",
        "Yersinia_pestis_CO92",
    ],
    "viruses": [
        "Human adenovirus A",
        "Human adenovirus C",
        "Human herpesvirus 1",
        "Human herpesvirus 4",
        "Human herpesvirus 5",
        "Human immunodeficiency virus 1",
        "Influenza A virus H5N1",
        "Macacine herpesvirus 3",
        "Moloney murine leukemia virus",
        "Monkeypox_virus_Zaire_96_I_16",
        "Murid herpesvirus 4",
        "Murine_type_C_retrovirus",
        "Pandoravirus salinus",
        "Pithovirus_sibericum",
        "Saccharomyces_cerevisiae_virus_L_A",
        "Saccharomyces_cerevisiae_virus_L_BC",
        "Vaccinia_virus",
    ],
    "special": ["cRAP"],
}

BASE_URL = "https://gpmdb.thegpm.org/thegpm-cgi/peptides_by_species.pl"
DATA_PATH = MODULE_CACHE_PATH

MAX_SEQ_LENGTH = 81


class GPMDataset(Dataset):
    """This class defines a PyTorch Dataset for processing proteomics data
    for multiple species from the Global Proteomics Database at:
    https://gpmdb.thegpm.org/thegpm-cgi/peptides_by_species.pl

    The data contains sequences of amino acids samples from different species
    in tsv files. Depending on the download option, this dataset can be
    downloaded remotely into a local path or loaded directly from the local path
    if already downloaded previously, as long as the format includes a 'sequence'
    column and an 'E' (detectability) column.
    """

    LOCAL = "local"
    REMOTE = "remote"
    NONE = "none"
    DOWNLOAD_OPTIONS = ["local", "remote", "none"]

    # Inner class for data generation
    class GenerateData:
        def __init__(
            self,
            aa_map,
            max_seq_length,
            species,
            base_url,
            data_path,
            verbose,
            download_option,
        ):

            # amino acid map
            self.aa_map = aa_map

            # dictionary of species
            self.species = species

            # max length for padding sequences
            self.max_seq_length = max_seq_length

            # flag for debugging
            self.verbose = verbose

            # URL for GPM data source
            self.base_url = base_url

            # data loading option, chosen from local storage, remote download, or none for empty
            self.download_option = download_option

            # path to saving downloaded data locally
            self.data_path = data_path

            # download data
            self.data = None
            if self.download_option != GPMDataset.NONE:
                self.vprint(f"Downloading dataset from {self.base_url}")
                if self.download_option == GPMDataset.REMOTE:
                    self.download_data()
                    self.vprint(f"Loading data from {self.data_path}")
                self.data = self.fetch_data_locally()
                self.vprint(f"Download completed. Data saved at {self.data_path}.")

                # map amino acids to IDs
                self.vprint(
                    f"Mapping amino acid sequences of length {self.max_seq_length} with map of size {len(self.aa_map)}"
                )
                self.data["sequence"] = self.data["sequence"].apply(self.encode_pep_seq)
                filter = self.data["sequence"].apply(lambda x: -1 not in x)
                self.data = self.data[filter]
                self.vprint(f"Mapped {len(self.data)} amino acid sequences.")

        def vprint(self, message, indent=0):
            # print message if verbose flag is set
            if self.verbose:
                print(indent * "\t", message)

        def encode_pep_seq(self, seq: str, error: int = -1):
            # translate a string of amino acid characters into a sequence of IDs
            # the error value is used in case characters outside the mapping are found
            encoded_seq = []
            for aa in seq:
                if aa in self.aa_map:
                    encoded_seq.append(self.aa_map[aa])
                else:
                    encoded_seq.append(error)
                    self.vprint(f"found {aa} which is not in map: {self.aa_map}")

            # pad sequences to max seq length
            while len(encoded_seq) < self.max_seq_length:
                encoded_seq.append(0)
            return encoded_seq

        def download_data(self):
            # download data from source and save to local folder
            with requests.Session() as session:
                os.makedirs(self.data_path, exist_ok=True)
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
                }
                for spec in itertools.chain.from_iterable(self.species.values()):
                    self.vprint(f"Downloading {spec} from {self.base_url}.", indent=1)
                    resp = None
                    try:
                        resp = session.get(
                            f"{self.base_url}?species={spec}", headers=headers
                        )
                        if resp.status_code == 200:

                            with open(
                                f"{os.path.join(self.data_path, spec)}.tsv",
                                "w",
                                encoding="utf-8",
                            ) as specie_file:
                                specie_file.write(resp.text)

                    except (
                        pd.errors.ParserError,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ChunkedEncodingError,
                        requests.exceptions.ReadTimeout,
                    ) as e:
                        print(f"Couldn't download {spec} from {resp.url}\n{e}")
                    self.vprint(f"Downloaded {resp.url}", indent=1)

        def fetch_data_locally(self):
            # load a dataframe from a local folder with .tsv files
            os.makedirs(self.data_path, exist_ok=True)
            p_filenames = os.listdir(self.data_path)
            p_dfs = []

            for p_filename in p_filenames:
                p_filepath = f"{os.path.join(self.data_path, p_filename)}"
                if p_filepath.endswith(".tsv"):
                    self.vprint(f"Loading from {p_filepath}", indent=1)
                    try:
                        with open(p_filepath, "r", encoding="utf-8") as p_file:
                            df = pd.read_csv(p_file, sep="\t")
                            p_dfs.append(df)

                    except pd.errors.ParserError as e:
                        print(f"Couldn't load from {p_filepath}")

            return pd.concat(p_dfs)

    def __init__(
        self,
        base_url=BASE_URL,
        data_path=DATA_PATH,
        aa_map=AMINO_ACID_DICT,
        max_seq_length=MAX_SEQ_LENGTH,
        species=SPECIES,
        verbose=False,
        dataset_name=None,
        download_option="remote",
    ):
        "Initialization"
        self.gen = self.GenerateData(
            aa_map,
            max_seq_length,
            species,
            base_url,
            data_path,
            verbose,
            download_option,
        )
        self.data = self.gen.data
        self.it = iter(self.generate_examples())
        self.aa_map = aa_map
        self.max_seq_length = max_seq_length
        self.data_path = data_path
        self.species = species
        self.base_url = base_url
        self.verbose = verbose

        if download_option not in self.DOWNLOAD_OPTIONS:
            raise ValueError(
                f"Invalid download option. Must be one of {self.DOWNLOAD_OPTIONS}"
            )
        self.download_option = download_option

        if dataset_name:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = self.data_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            ex = next(self.it)
        except StopIteration:
            self.it = iter(self.gen.generate_examples())
            ex = next(self.it)

        X = ex["sequence"]
        y = ex["E"]

        return X, y

    def train_test_split(
        self, train_frac, shuffle=True, train_dataset_name=None, test_dataset_name=None
    ):
        # split dataset into training and testing datasets, returning two GPMDatasets
        if not (train_frac > 0 and train_frac < 1):
            raise ValueError(
                f"Failed to generate datasets. Training fraction {train_frac} must be > 0 and < 1."
            )

        if shuffle:
            data = self.data.sample(frac=1)
        else:
            data = self.data

        train_dataset_name = (
            train_dataset_name if train_dataset_name else f"{self.dataset_name}_train"
        )
        test_dataset_name = (
            test_dataset_name if test_dataset_name else f"{self.dataset_name}_test"
        )

        train_data, test_data = train_test_split(data, test_size=1 - (train_frac))

        train_gpmdb = GPMDataset(
            base_url=self.base_url,
            data_path=self.data_path,
            aa_map=self.aa_map,
            max_seq_length=self.max_seq_length,
            species=self.species,
            verbose=self.verbose,
            dataset_name=train_dataset_name,
            download_option=self.NONE,
        )
        train_gpmdb.data = train_data

        test_gpmdb = GPMDataset(
            base_url=self.base_url,
            data_path=self.data_path,
            aa_map=self.aa_map,
            max_seq_length=self.max_seq_length,
            species=self.species,
            verbose=self.verbose,
            dataset_name=test_dataset_name,
            download_option=self.NONE,
        )
        test_gpmdb.data = test_data

        return train_gpmdb, test_gpmdb

    def generate_examples(self):
        # generator function that returns one data sample
        for id, row in self.data.iterrows():
            yield row

    def stat(self):
        lines = list()
        lines.append("")
        lines.append(f"Statistics of the Dataset:")
        lines.append(f"\t- Dataset Name: {self.dataset_name}")
        lines.append(f"\t- Number of samples: {len(self.data)}")
        lines.append(f"\t- Types of amino acids: {len(self.aa_map)}")
        lines.append(f"\t- Max sequence length: {self.max_seq_length}")
        lines.append(f"\t- Number of species: {len(self.species)}")
        for k, v in self.species.items():
            lines.append(f"\t- {k} samples included:  {v}")
        if self.data_path:
            lines.append(f"\t- Dataset saved at: {self.data_path}")
        lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    params = {"batch_size": 32, "num_workers": 1}

    # creating a custom dataset by specifiying species
    single_db = GPMDataset(
        dataset_name="single_gpm_dataset",
        species={"eukayrotes": ["Anopheles gambiae"]},
        verbose=True,
    )
    print(single_db.stat())

    # results of splitting dataset into training and test sets
    train_db, test_db = single_db.train_test_split(train_frac=0.8)
    train_db, val_db = single_db.train_test_split(
        train_frac=0.8,
        train_dataset_name=train_db.dataset_name,
        test_dataset_name=f"{single_db.dataset_name}_val",
    )
    print(train_db.stat())
    print(test_db.stat())
    print(val_db.stat())

    training_generator = DataLoader(train_db, **params)
    testing_generator = DataLoader(test_db, **params)
    validation_generator = DataLoader(val_db, **params)

    # creating a mini dataset by taking subset of all data
    # small_db = GPMDataset(dataset_name='small_gpm_dataset', data_path='small_gpm', species={k:v[:3] for k,v in SPECIES.items()})
    # print(small_db.stat())

    # complete dataset
    # gpmdb = GPMDataset()
    # print(gpmdb.stat())

    ### Output
    # Downloading dataset from https://gpmdb.thegpm.org/thegpm-cgi/peptides_by_species.pl
    #     Downloading Anopheles gambiae from https://gpmdb.thegpm.org/thegpm-cgi/peptides_by_species.pl.
    #     Downloaded https://gpmdb.thegpm.org/thegpm-cgi/peptides_by_species.pl?species=Anopheles%20gambiae
    # Loading data from single_gpm
    #     Loading from single_gpm/Anopheles gambiae.tsv
    # Download completed. Data saved at single_gpm.
    # Mapping amino acid sequences of length 81 with map of size 22
    # Mapped 45580 amino acid sequences.
    #
    # Statistics of the Dataset:
    # 	- Dataset Name: single_gpm_dataset
    # 	- Number of samples: 45580
    # 	- Types of amino acids: 22
    # 	- Max sequence length: 81
    # 	- Number of species: 1
    # 	- eukayrotes samples included:  ['Anopheles gambiae']
    # 	- Dataset saved at: single_gpm
    #
    #
    # Statistics of the Dataset:
    # 	- Dataset Name: single_gpm_dataset_train
    # 	- Number of samples: 29171
    # 	- Types of amino acids: 22
    # 	- Max sequence length: 81
    # 	- Number of species: 1
    # 	- eukayrotes samples included:  ['Anopheles gambiae']
    # 	- Dataset saved at: single_gpm
    #
    #
    # Statistics of the Dataset:
    # 	- Dataset Name: single_gpm_dataset_test
    # 	- Number of samples: 9116
    # 	- Types of amino acids: 22
    # 	- Max sequence length: 81
    # 	- Number of species: 1
    # 	- eukayrotes samples included:  ['Anopheles gambiae']
    # 	- Dataset saved at: single_gpm
    #
    #
    # Statistics of the Dataset:
    # 	- Dataset Name: single_gpm_dataset_val
    # 	- Number of samples: 7293
    # 	- Types of amino acids: 22
    # 	- Max sequence length: 81
    # 	- Number of species: 1
    # 	- eukayrotes samples included:  ['Anopheles gambiae']
    # 	- Dataset saved at: single_gpm
