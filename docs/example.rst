Examples
========
**Demo for RETAIN on Drug Recommendation with GPU**\ :

Step 1: Create dataset
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # load Dataset and transform it to OMOP form
    from pyhealth.datasets import MIMIC3BaseDataset

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")


.. code-block:: javascript

   ----- Output Data Structure -----
   Dataset.patients: [
      {
          "patient_id": patient_id,
          "visits": [
              {
                  "visit_id": visit_id,
                  "patient_id": patient_id,
                  "conditions": [List],
                  "procedures": [List],
                  "drugs": [List],
                  "visit_info": <dict>
              }
              ...
          ]
      }
      ...
   ]


* **User can use this module for data processing**

* **[researchers from CS]** build their own model on top of it
* **[reserachers from medical area]** use the models in the package

.. code-block:: python

   # create task-specific dataset
   from pyhealth.tasks import DrugRecDataset
   drug_rec_dataset = DrugRecDataset(base_dataset)
   drug_rec_dataset.info()

.. code-block:: javascript

   ----- Output Data Structure -----
   >> drug_rec_dataloader[0]
   >> {
      "conditions": List[tensor],
      "procedures": List[tensor],
      "drugs": List[tensor]
   }


Step 2: Select healthcare predictive model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For this example, we select `RETAIN <https://arxiv.org/abs/1608.05745/>`_ as the healthcare predictive model.

.. code-block:: python

   voc_size = drug_rec_dataset.voc_size
   params = drug_rec_dataset.params

   from pyhealth.models import RETAIN
   model = RETAIN(voc_size, params)


Step 3: Create dataloader
^^^^^^^^^^^^^^^^^^^^^^^^^
Then, we create train/val/test dataloader in PyTorch format.

.. code-block:: python


    from pyhealth.data import split
    from torch.utils.data import DataLoader

    drug_rec_trainset, drug_rec_valset, drug_rec_testset = split.random_split(drug_rec_dataset, [0.8, 0.1, 0.1])
    drug_rec_train_loader = DataLoader(drug_rec_trainset, batch_size=1, collate_fn=lambda x: x[0])
    drug_rec_val_loader = DataLoader(drug_rec_valset, batch_size=1, collate_fn=lambda x: x[0])
    drug_rec_test_loader = DataLoader(drug_rec_testset, batch_size=1, collate_fn=lambda x: x[0])



Step 4: Train & Predict & Evaluate
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we **train** our model:

.. code-block:: python

    # train
    from pytorch_lightning import Trainer

    trainer = Trainer(
        gpus=1,
        max_epochs=3,
        progress_bar_refresh_rate=5,
    )

    trainer.fit(
        model=model,
        train_dataloaders=drug_rec_train_loader,
        val_dataloaders=drug_rec_val_loader,
    )

and make **drug recommendation (prediction)** with the trained model:

.. code-block:: python

    # evaluation
    from pyhealth.evaluator import DrugRecEvaluator
    evaluator = DrugRecEvaluator(model)
    evaluator.evaluate(drug_rec_test_loader)

**Evaluation**:

.. code-block:: javascript

    Jaccard: 0.132,  PRAUC: 0.3148, AVG_PRC: 0.5275, AVG_RECALL: 0.1575, AVG_F1: 0.2291, AVG_MED: 14.08