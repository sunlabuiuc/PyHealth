Examples
========================

---------DL Model---------
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Demo for RETAIN on Drug Recommendation with GPU**\ :

Step 1: Load Dataset
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # load Dataset and transform it to OMOP form
    from pyhealth.datasets import MIMIC3BaseDataset
    base_ds = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4", flag="prod")


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

Step 2: Build Task-specific Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cast the dataset into a task-specific one:

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
..

Create dataloaders from the task-specific dataset:

.. code-block:: python

    # create dataloaders
    from pyhealth.data.split import split_by_pat
    train_loader, val_loader, test_loader = split_by_pat(drug_rec_dataset, [2/3, 1/6, 1/6], batch_size=64)


Step 3: Select Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For this example, we select `RETAIN <https://arxiv.org/abs/1608.05745/>`_ as the healthcare predictive model.

.. code-block:: python

    from pyhealth.models import RETAIN
    model = RETAIN(
        task = "drug_recommendation",
        voc_size = drug_rec_dataset.voc_size,
        tokenizers = drug_rec_dataset.tokenizers,
        emb_dim = 64,
    )


Step 4: Training
^^^^^^^^^^^^^^^^


.. code-block:: python


    from pyhealth.trainer import Trainer
    from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel

    trainer = Trainer(enable_logging=True, output_path="../output")
    trainer.fit(model,
                train_loader=train_loader,
                epochs=50,
                evaluate_fn=evaluate_multilabel,
                eval_loader=val_loader,
                monitor="jaccard")

    # load the best model
    best_model = trainer.load_best(model)



Step 5: Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we **evaluate** our model:

.. code-block:: python

    from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel
    evaluate_multilabel(model, test_loader, 'cpu')


.. code-block:: javascript

    Jaccard: 0.132,  PRAUC: 0.3148, AVG_PRC: 0.5275, AVG_RECALL: 0.1575, AVG_F1: 0.2291, AVG_MED: 14.08
..

---------ML Model---------
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Demo for SVM on Drug Recommendation**\ :

Step 1: Load Dataset
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # load Dataset and transform it to OMOP form
    from pyhealth.datasets import MIMIC3BaseDataset
    base_ds = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4", flag="prod")


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

Step 2: Build Task-specific Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cast the dataset into a task-specific one:

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
..

Create dataloaders from the task-specific dataset:

.. code-block:: python

    # create dataloaders
    from pyhealth.data.split import split_by_pat
    train_loader, val_loader, test_loader = split_by_pat(drug_rec_dataset, [2/3, 1/6, 1/6], batch_size=64)


Step 3: Select Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For this example, we select SVM classifier as the healthcare predictive model.

.. code-block:: python

    from pyhealth.models import MLModel
    from sklearn.svm import SVC
    model = MLModel(
        task = "drug_recommendation",
        classifier = SVC(gamma='auto', verbose=1),
    #     classifier = LogisticRegression(random_state=0, n_jobs=-1, max_iter=10, verbose=1),
        voc_size = drugrec_ds.voc_size,
        tokenizers = drugrec_ds.tokenizers
    )


Step 4: Training
^^^^^^^^^^^^^^^^


.. code-block:: python

    model.fit(
        train_loader=train_loader,
        evaluate_fn=evaluate_multilabel,
        eval_loader=val_loader,
        monitor="jaccard",
    )


Step 5: Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we **evaluate** our model:

.. code-block:: python

    from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel
    evaluate_multilabel(model, test_loader, 'cpu')

..

.. toctree::
   :maxdepth: 4

   pyhealth.data