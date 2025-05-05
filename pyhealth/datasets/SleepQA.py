import pandas as pd
import os
from pyhealth.datasets import BaseDataset
from typing import Dict, Tuple, List

class SleepQADataset(BaseDataset):
    def __init__(self, root: str = "/content/SleepQA/data/training", dev: bool = False):
        # Load and prepare data
        corpus_path = os.path.join(root, "sleep-corpus.tsv")
        qa_path = os.path.join(root, "sleep-test.csv")

        corpus_df = pd.read_csv(corpus_path, sep="\t", header=None, names=["id", "passage", "title"])
        qa_df = pd.read_csv(qa_path, sep="\t", header=None, names=["question", "answer"])

        if dev:
            corpus_df = corpus_df.sample(n=500, random_state=42)
            qa_df = qa_df.sample(n=50, random_state=42)

        # Define tables dictionary for BaseDataset
        tables = {
            "corpus": corpus_df,
            "qa": qa_df
        }

        super().__init__(dataset_name="SleepQA", root=root, tables=tables, config_path="/content/sleepQA.yaml", dev=dev)

        # Store examples for task-level work later
        self.examples = [
            {"question": row["question"].strip(), "answer": row["answer"].strip()}
            for _, row in qa_df.iterrows()
        ]
        self.corpus = corpus_df

    def get_raw_samples(self) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        return self.corpus, self.examples

# Main function to instantiate and test the dataset class
def main():
    dataset = SleepQADataset(dev=True)  # Set dev=True for smaller data
    corpus, examples = dataset.get_raw_samples()

    print(f"Corpus:\n{corpus.head()}")
    print(f"Examples:\n{examples[:5]}")  # Print the first 5 examples

if __name__ == "__main__":
    main()
