# -*- coding: utf-8 -*-
"""dsa_activity_classification.ipynb

# Activity Classification Using DSA Dataset

# Install PyHealth

You might need to restart kernel after running this section.
"""

!git clone https://github.com/ranyou/PyHealth.git
!cd PyHealth && pip install -e .

"""# Load Dataset"""

from pyhealth.datasets import DSADataset

dataset = DSADataset(download=True, root="./daily-and-sports-activities")
dataset.stats()

"""# Define Task"""

samples = dataset.set_task()

from pyhealth.datasets import get_dataloader, split_by_sample

train_dataset, val_dataset, test_dataset = split_by_sample(samples, [0.7, 0.1, 0.2])

train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)

"""# Define Model"""

from pyhealth.models import RNN

model = RNN(samples)

"""# Train Model"""

from pyhealth.trainer import Trainer

trainer = Trainer(model=model, metrics=["accuracy"])
trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=1)

"""# Evaluate Model"""

trainer.evaluate(test_loader)

