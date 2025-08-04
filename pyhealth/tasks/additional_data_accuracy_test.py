import os
import gc
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.algorithms import get_algorithm_class
from pyhealth.datasets import get_dataset_class
from pyhealth.utils import FastDataLoader, seed_everything

class EmbeddingExtractionTask:
    """
    Task class for extracting embeddings from a trained model using a specified dataset and algorithm.
    Args:
        args: Namespace or argument object containing configuration parameters such as algorithm, dataset, model, output_dir, etc.
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). If None, automatically selects CUDA if available.
    Attributes:
        args: Configuration arguments.
        device: Computation device.
        hparams: Hyperparameters for the algorithm and dataset.
        dataset: Initialized dataset object.
        algorithm: Loaded and prepared model/algorithm for embedding extraction.
    Methods:
        _init_hparams():
            Initializes and returns hyperparameters for the algorithm and dataset using a random seed.
        _init_dataset():
            Instantiates and returns the dataset class with the initialized hyperparameters and arguments.
        _load_algorithm():
            Loads the algorithm/model class, restores its state from a checkpoint, and prepares it for evaluation.
        run_testing(split_name="test"):
            Runs embedding extraction and evaluation on the specified data split for all environments, saving results to CSV files.
        run():
            Seeds all random number generators, runs the testing procedure, and performs cleanup after extraction.
    """
    def __init__(self, args, device=None):
        self.args = args
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hparams = self._init_hparams()
        self.dataset = self._init_dataset()
        self.algorithm = self._load_algorithm()

    def _init_hparams(self):
        from pyhealth.hparams_registry import random_hparams
        from pyhealth.misc import seed_hash
        return random_hparams(
            self.args.algorithm,
            self.args.dataset,
            seed_hash(0, 0)
        )

    def _init_dataset(self):
        dataset_class = get_dataset_class(self.args.dataset)
        dataset_class.TRAIN_ENVS = ["MIMIC"]
        dataset_class.VAL_ENV = "MIMIC"
        dataset_class.TEST_ENV = "MIMIC"
        return dataset_class(self.hparams, self.args)

    def _load_algorithm(self):
        algorithm_class = get_algorithm_class(self.args.algorithm)
        algorithm = algorithm_class(
            input_shape=self.dataset.input_shape,
            num_classes=self.dataset.num_classes,
            num_domains=len(self.dataset.TRAIN_ENVS),
            hparams=self.hparams,
            dataset_name=self.args.dataset,
            dataset=self.dataset,
            model_name=self.args.model
        )
        checkpoint_path = os.path.join(self.args.output_dir, "model.pkl")
        algorithm.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        algorithm.to(self.device)
        algorithm.eval()
        return algorithm

    def run_testing(self, split_name="test"):
        print(f"Running testing on split: {split_name}")
        loaders = {
            env: FastDataLoader(
                dataset=self.dataset.get_torch_dataset([env], split_name, self.args),
                batch_size=self.hparams["batch_size"] * 4,
                num_workers=self.dataset.N_WORKERS,
                shuffle=False,
            ) for env in self.dataset.ENVIRONMENTS
        }

        for env, loader in loaders.items():
            print(f"Evaluating: {env}")
            meta_df = self.dataset.eval_metrics(
                self.algorithm, loader, env,
                device=self.device,
                weights=None,
                thresh=self.args.es_opt_thresh,
                emb_only=True
            )
            save_path = os.path.join(self.args.output_dir, f"emb-{split_name}-{env}.csv")
            meta_df.to_csv(save_path)
        
        del loaders
        gc.collect()

    def run(self):
        seed_everything(self.args.seed)
        self.run_testing()
        del self.algorithm
        del self.dataset
        gc.collect()
        print("Finished evaluation and extraction.")
