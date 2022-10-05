# APIs

```{eval-rst}
.. currentmodule:: pyhealth
```


## Data

```{eval-rst}
.. module:: pyhealth

```

```{eval-rst}
.. autosummary::
    :toctree: data
    :nosignatures:
    
    data.Event
    data.Visit
    data.Patient
    data.BaseDataset
    data.TaskDataset
```

## Datasets

```{eval-rst}
.. autosummary::
    :toctree: datasets
    :nosignatures:
    
    datasets.MIMIC3BaseDataset
    datasets.MIMIC4BaseDataset
    datasets.eICUBaseDataset
    datasets.OMOPBaseDataset
```

## Evaluator

```{eval-rst}
.. autosummary::
    :toctree: evaluator
    :nosignatures:
    
    evaluator.DrugRecEvaluator
    evaluator.metrics_multiclass
    evaluator.evaluate_multiclass
    evaluator.multi_label_metric
    evaluator.ddi_rate_score
    evaluator.evaluate_multilabel
```

## Models

```{eval-rst}
.. autosummary::
    :toctree: models
    :nosignatures:
    
    models.RNN
    models.RETAIN
    models.MICRON
    models.GAMENet
    models.SafeDrug
    models.MLModel
    models.Med2Vec
    models.Tokenizer
    models.Vocabulary
```

## Tasks

```{eval-rst}
.. autosummary::
    :toctree: tasks
    :nosignatures:
    
    tasks.DrugRecDataset
    tasks.Med2VecDataset
    tasks.LengthOfStayDataset
    tasks.MortalityDataset
    tasks.Readmission
```