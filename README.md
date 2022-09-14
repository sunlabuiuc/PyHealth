# PyHealth-OMOP
### The working ML pipeline in the package
> dataset instance -> task-specific dataset -> ML model -> trainer -> evaluator

### 1. Introduction
All healthcare tasks in our package follow this pipeline. We have implements one module for each of the steps in the package, and any task should call five modules in order to complete the pipeline by default. **We have tried hard to make these modules independent**, such that users can use part of the pipeline. We will explain it later after introducing these modules.
- ```pyhealth.datasets``` provides a clean structure for the dataset. Any instance from this module is independent from downstream tasks. We support the processing scripts for three open datasets, MIMIC-III, MIMIC-IV and eICU, as well as the standard OMOP-formatted data. The output instance contains a unified **dictionary-based structure**.
- ```pyhealth.tasks``` inputs the **<pyhealth.datasets>** object and further processes it (clean up irrelevant info.) and finally provides another structure for the specific task. This module bridges the dataset instance and the downstream models.
- ```pyhealth.models``` provides the state-of-the-art healthcare ML models
- We use ```from pytorch_lightning import Trainer``` as our model trainer
- ```pyhealth.evaluator``` lists the evaluator for each healthcare tasks with detailed metrics specification.

For use cases, users can either use our package for the entire ML pipeline or borrow part of it, such as (i) process your own OMOP data via ```pyhealth.datasets.omop```; (ii) using our pre-processed MIMIC-III, MIMIC-IV, eICU via ```pyhealth.datasets``` (for generating your own task dataset); (iii) using our pre-processed task-specific dataset  via ```pyhealth.datasets``` and ```pyhealth.tasks```; (iv) using the healthcare ML models only via ```pyhealth.models```.

### 2. Demo - drug recommendation
#### Step 1: Load the dataset instance
```python
from pyhealth.datasets import MIMIC3BaseDataset
base_ds = MIMIC3BaseDataset(root="...", files=['conditions', ...])
```
#### Step 2: Process for obtaining task-specific instance
```python
from pyhealth.tasks import DrugRecDataset
drug_rec_ds = DrugRecDataset(base_ds)

# task-specific artifacts for each downstream model
voc_size = drug_rec_dataset.voc_size
params = drug_rec_dataset.params
```
#### Step 3: load the healthcare predictive model
```python
from pyhealth.models import RETAIN
model = RETAIN(voc_size, params).train(drug_rec_ds)
```
#### Step 4: Model training
```python
# prepare for train / val / test
from pyhealth.data import split
from torch.utils.data import DataLoader
drug_rec_trainset, drug_rec_valset, drug_rec_testset = split.random_split(drug_rec_dataset, [0.8, 0.1, 0.1])
drug_rec_train_loader = DataLoader(drug_rec_trainset, batch_size=1, collate_fn=lambda x: x[0])
drug_rec_val_loader = DataLoader(drug_rec_valset, batch_size=1, collate_fn=lambda x: x[0])
drug_rec_test_loader = DataLoader(drug_rec_testset, batch_size=1, collate_fn=lambda x: x[0])

# training
from pytorch_lightning import Trainer
trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=5)
trainer.fit(model=model, train_dataloaders=drug_rec_train_loader, val_dataloaders=drug_rec_val_loader)
```
#### Step 5: Model evaluation
```python
from pyhealth.evaluator import DrugRecEvaluator
evaluator = DrugRecEvaluator(model)
evaluator.evaluate(drug_rec_test_loader)
```

### Step 6: Code mappings
```python
from pyhealth.codemap import InnerMap
ICD = InnerMap('icd-10')
ICD['I50. 9'] # heart failure
ICD.patient('I50. 9')
ICD.siblings('I50. 9')
ICD.children('I50. 9')

from pyhealth.codemap import CrossMap
NDC_to_RxNorm = CrossMap('NDC', 'RxNorm')
NDC_to_RxNorm['76413-153-06'] # AZITHROMYCIN tablet
>> ['68084027801', '59762306003']
```
