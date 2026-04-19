"""T5 fine-tuning example for the NCBI Disease recognition task.

This example uses the PyHealth NCBI Disease dataset/task and derives a
text-to-text BIO tagging objective locally. That keeps the PyHealth
contribution focused on reproducible data loading while still matching the
T5-style setup used by the paper.
"""

import shutil
import tempfile
from pathlib import Path

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from pyhealth.datasets import NCBIDiseaseDataset
from pyhealth.tasks import NCBIDiseaseRecognition


class TokenizedTextDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def serialize_source(tokens, task_prefix: str):
    prefix = f"{task_prefix.strip()}: " if task_prefix.strip() else ""
    return prefix + " ".join(tokens)


def build_hf_dataset(
    sample_dataset,
    tokenizer,
    max_length: int = 256,
    task_prefix: str = "ncbi disease",
):
    rows = []
    for sample in sample_dataset:
        tokens, labels = NCBIDiseaseRecognition.entities_to_bio_tags(
            sample["text"], sample["entities"]
        )
        source_text = serialize_source(tokens, task_prefix)
        target_text = " ".join(labels)
        rows.append({"input_text": source_text, "target_text": target_text})

    tokenized_rows = []
    for row in rows:
        model_inputs = tokenizer(
            row["input_text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        labels = tokenizer(
            text_target=row["target_text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        tokenized_rows.append(model_inputs)

    return TokenizedTextDataset(tokenized_rows)


def copy_demo_root(source_root: str, target_root: str) -> None:
    source_path = Path(source_root)
    target_path = Path(target_root)
    for path in source_path.glob("NCBI*_corpus.txt"):
        shutil.copy(path, target_path / path.name)


if __name__ == "__main__":
    model_name = "google/flan-t5-base"
    dataset_root = "test-resources/ncbi_disease"

    with tempfile.TemporaryDirectory() as root:
        copy_demo_root(dataset_root, root)

        base_dataset = NCBIDiseaseDataset(root=root, dev=False)
        sample_dataset = base_dataset.set_task(NCBIDiseaseRecognition())

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        hf_dataset = build_hf_dataset(sample_dataset, tokenizer)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        args = Seq2SeqTrainingArguments(
            output_dir="./output/ncbi_disease_t5",
            per_device_train_batch_size=2,
            num_train_epochs=1,
            learning_rate=2e-5,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=hf_dataset,
            eval_dataset=hf_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
