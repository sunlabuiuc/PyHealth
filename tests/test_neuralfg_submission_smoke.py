"""Fast, deterministic NeuralFG PR checks using only tiny synthetic inputs.

Place under tests/test_neuralfg_submission_smoke.py and run from PyHealth root:
  python -m pytest tests/test_neuralfg_submission_smoke.py -q
"""
import pytest
import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.neural_fg import NeuralFG


def _dataset():
    samples = [
        {'patient_id': f'p{i}', 'features': [0.1*i, 0.2+i, 0.5-i*.1],
         'duration': [float(i+1)], 'event': [i%3]}
        for i in range(6)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={'features': 'tensor'},
        output_schema={'duration': 'tensor', 'event': 'tensor'},
        dataset_name='neuralfg_pr_smoke',
    )


def _model(ds):
    torch.manual_seed(42)
    return NeuralFG(dataset=ds, num_risks=2, hidden_dim=8, embedding_dim=8,
                    embedding_layers=1, monotonic_layers=1, dropout=0,
                    time_grid=torch.tensor([0.0,1.0,2.0,3.0]))


def test_neuralfg_forward_and_gradient():
    ds=_dataset(); model=_model(ds)
    batch=next(iter(get_dataloader(ds,batch_size=6,shuffle=False)))
    result=model(**batch)
    assert result['y_prob'].shape==(6,4,2)
    assert result['y_true'].shape==(6,2)
    assert result['loss'].ndim==0 and torch.isfinite(result['loss'])
    result['loss'].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_neuralfg_cif_monotonic_and_bounded():
    model=_model(_dataset())
    features=torch.tensor([[0.3,0.1,0.7],[-0.1,0.2,1.0]])
    times=torch.linspace(0,5,25)
    predicted=model.predict_cif(features,times)
    assert predicted.shape==(2,25,2)
    assert torch.all(predicted>=-1e-7)
    assert torch.all(predicted.sum(dim=-1)<=1.0+1e-6)
    assert torch.all(predicted[:,1:,:] >= predicted[:,:-1,:]-1e-6)
    assert torch.allclose(predicted[:,0,:],torch.zeros(2,2),atol=1e-6)


def test_neuralfg_invalid_event_rejected():
    model=_model(_dataset())
    with pytest.raises(ValueError,match='event labels cannot exceed'):
        model.network.competing_risk_nll(torch.zeros(2,3),torch.tensor([1.,2.]),
                                        torch.tensor([1,3]))
