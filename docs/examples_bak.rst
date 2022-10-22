Examples
========================

**Hands on Five-stage Pipeline:** A demo on applying a deep learning model (RETAIN) and a classical machine learning model (logistic regression) on MIMIC-III drug recommendation task.

Step 1: Load Dataset
----------------------
We initialize the MIMIC-III general-purpose dataset object. Here, ``conditions`` and ``procedures`` are the two types of 
events that are used as features (users can add more, such as ``labs``, to argument ``files``) for predicting ``drugs``.

.. code-block:: python

    from pyhealth.datasets import MIMIC3BaseDataset
    base_ds = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4", 
        files=['conditions', 'procedures', 'drugs'],
    )
..

The **output** ``base_ds.patients`` is a dict-based object and indexed by ``patient_id`` and ``visit_id``. It contains visit 
status and multiple event sequences under one visit. It is a general purpose object. 

.. code-block:: javascript

    <dict>
        - patient_id <str>: <Patient>

        <Patient>
            - patient_id <str>
            - visits <dict>
                - visit_id <str>: <Visit>

            <Visit>
                - visit_id <str>
                - patient_id <str>
                - encounter_time <float>
                - duration <float>
                - mortality_status <bool>
                - conditions <list[Event]>
                - procedures <list[Event]>
                - drugs <list[Event]>

                <Event>
                    - code <str>
                    - time <float>
..

- [!!!] Researchers can use this data processing module alone for supporting their own tasks. 

Step 2: Task-specific Process
-------------------------------

Cast the general-purpose dataset by cleaning the structure for the specific task

.. code-block:: python

   from pyhealth.tasks import DrugRecDataset
   drug_rec_ds = DrugRecDataset(base_dataset)
..

The **output** ``drug_rec_ds`` is a ``torch.utils.data.Dataset`` object and can be used to create dataloader. Users can also use ``drug_rec_ds.patients`` which is also dict-based and has the form:

.. code-block:: javascript

   <dict>
        - patient_id <str>: <Patient>
        
        <Patient>
           - patient_id <str>
            - visits <dict>
                - visit_id <str>: <DrugRecVisit>
        
            <DrugRecVisit>
                - visit_id <str>
                - patient_id <str>
                - conditions <list>
                - procedures <list>
                - drugs <list>
..

Create data loaders:

.. code-block:: python

    from pyhealth.data.split import split_by_pat
    train_loader, val_loader, test_loader = split_by_pat(drug_rec_ds, 
                                                         ratios=[0.8, 0.1, 0.1], 
                                                         batch_size=64)
..

Step 3: Build Deep Learning Models
-----------------------------------

We choose `RETAIN <https://arxiv.org/abs/1608.05745/>`_ as the healthcare predictive model.

.. code-block:: python

    from pyhealth.models import RETAIN
    model = RETAIN(task="drug_recommendation",
                   voc_size=drug_rec_dataset.voc_size,
                   tokenizers=drug_rec_dataset.tokenizers,
                   emb_dim=64)
..

- [!!!] Researchers can call our model functions alone for your own prediction tasks. We have implemented more than 25 recent deep learning moddels published in top venues!

Step 4: Training
------------------

Call the Trainer and specify your own necessary configurations and the wait quietly. You can speficy to train the model on CPU or CUDA. By default, we select the best available CUDA in your enviorment.

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
..


Step 5: Evaluation
---------------------

The evaluation is as simple as calling ``sklearn.metrics``. Load the best model from the trainer and then call the task metric.

.. code-block:: python

    from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel

    # load the best model
    best_model = trainer.load_best(model)
    evaluate_multilabel(best_model, test_loader)

    # result
    {'ddi': 0.07266, 'jaccard': 0.4767, 'prauc': 0.7385, 'f1': 0.6366}
..

---------

Using Classical ML Models 
----------------------------

Starting from **Step 2**, we wrap the classical ML models from ``sklearn`` into the ``MLModel`` function and provide a unified interface for training and evaluation.

- Model initialization

.. code-block:: python

    from pyhealth.models import MLModel
    from sklearn.linear_model import LogisticRegression
    model = MLModel(output_path="../output",
                 task="drug_recommendation",
                 classifier=LogisticRegression(random_state=0, max_iter=10),
                 voc_size=drugrec_ds.voc_size,
                 tokenizers=drugrec_ds.tokenizers)
..

- Model training

.. code-block:: python

    model.fit(train_loader=train_loader,
              evaluate_fn=evaluate_multilabel,
              eval_loader=val_loader,
              monitor="jaccard")
..

- Model evaluation

.. code-block:: python

    model.load(path="../output/221002-170055/best.ckpt")
    evaluate_multilabel(model, test_loader)
..

----------