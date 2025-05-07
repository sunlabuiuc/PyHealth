from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel

class TransformersforSequenceClassification(BaseModel):
    """AutoModelForSequenceClassification for Huggingface models.
    
    This class is used for sequence classification tasks where the 
    input is any free-text and the output can either be a single label 
    (multiclass), multiple labels (multilabel), or a regression
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
        max_length: int = 256
    ):
        super(TransformersforSequenceClassification, self).__init__(
            dataset=dataset,
        )
        self.model_name = model_name

        assert len(self.feature_keys) == 1, "Only one feature key is supported if Transformers is initialized"
        self.feature_key = self.feature_keys[0]
        assert len(self.label_keys) == 1, "Only one label key is supported if RNN is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        if self.mode == "multilabel":
            problem_type = "multi_label_classification"
            num_labels = self.get_output_size()
        elif self.mode == "multiclass":
            problem_type = "single_label_classification"
            num_labels = self.get_output_size()
        elif self.mode == "binary":
            problem_type = "single_label_classification"
            num_labels = 2
        elif self.mode == "regression":
            problem_type = "regression"
            num_labels = 1
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        x = kwargs[self.feature_key]
        x = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        x = x.to(self.device)

        outputs = self.model(**x)
        logits = outputs.logits
        loss = outputs.loss
        y_true = kwargs[self.label_key].to(self.device)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }
    
if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset, get_dataloader
    from pyhealth.tasks import MIMIC3ICD9Coding
    # Testing the model with the MIMIC3 medical coding task dataset
    root = "/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4"
    dataset = MIMIC3Dataset(
        root=root,
        dataset_name="mimic3",  
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "NOTEEVENTS"
        ],
        dev=True,
    )

    mimic3_coding = MIMIC3ICD9Coding()
    samples = dataset.set_task(mimic3_coding)

    train_loader = get_dataloader(samples, batch_size=4, shuffle=True)

    model = TransformersforSequenceClassification(
        dataset=samples,
        feature_keys=["text"],
        label_key="icd_codes",
        mode="multilabel",
        model_name="whaleloops/keptlongformer",
        max_length=2048
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()