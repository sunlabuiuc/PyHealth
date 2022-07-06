# %% md

# 1. load/reload all dependency


from MedCode import CodeMapping

import pyhealth.datasets.datasets as datasets
from pyhealth.evaluator.evaluating_drug_recommendation import DrugRecommendationEvaluator
from pyhealth.models.modeling_retain import RETAIN
from pyhealth.trainer import Trainer

# 2. get MIMIC-III dataset object

# choose a target code: ATC4, NDC or RXCUI
target_code = 'ATC4'
tool = CodeMapping('RxNorm', target_code)
tool.load_mapping()

# load dataset by passing arguments
dataset = datasets.MIMIC_III(
    table_names=['med', 'prod', 'diag'],
    code_map=tool.RxNorm_to_ATC4,
)

# 3. model train and test

# initialize RETAIN model and build the dataloaders
model = RETAIN(
    voc_size=dataset.voc_size,
    emb_size=64,
    mode="multilabel",
    num_class=dataset.voc_size[2],
)
dataset.get_dataloader("RETAIN")

evaluator = DrugRecommendationEvaluator(model)

# set checkpoint and trainer
trainer = Trainer()

# train model
trainer.fit(
    model=model,
    train_dataloader=dataset.train_loader,
    evaluator=evaluator,
    eval_dataloader=dataset.val_loader,
    epochs=2,

)

# load the best model for test
evaluator.evasluate(dataset.test_loader)

# %%
