import torch
from torch import nn
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
"""
Bailey Malis and Divy Sunderraj
NetIDS: bmalis2, diivyesh3
Custom LSTM model for PyHealth SampleDataset.

Implements a simple LSTM model.
    -LSTM: PyHealth LSTM basemodel subclass.
    -test_lstm: function to test the LSTM model with a SampleDataset.
    

LSTM model is used in reproducing results from:
Addressing Wearable Sleep Tracking Inequity: a New Dataset and Novel Methods
for a Population with Sleep Disorders

Link to paper: https://arxiv.org/abs/2306.14808

"""



class LSTM(BaseModel):
    """LSTM model for PyHealth SampleDataset."""
    def __init__(self, dataset, input_size=1, hidden_size=16, num_layers=1, dropout=0):
        super().__init__(dataset)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout 
        )

        self.fc = nn.Linear(hidden_size, self.get_output_size())

    def forward(self, x, y):
        """
        Forward pass of the LSTM model.
        """
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        loss = self.get_loss_function()(logits, y.to(self.device))
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y, "logit": logits}
    
def test_lstm():
    """
    Test the LSTM model with a SampleDataset.
    """
    samples = [
      {
          "patient_id": "patient-0",
          "visit_id": "visit-0",
          "procedures": [[1.0], [2.0], [3.5], [4.0]],
          "label": 0,
      },
      {
          "patient_id": "patient-1",
          "visit_id": "visit-1",
          "procedures": [[5.0], [2.0], [3.5], [4.0]],
          "label": 1,
      },
      ] 

    input_schema = {
      "procedures": "tensor",
    }
    output_schema = {"label": "binary"}  

    dataset = SampleDataset(samples, input_schema, output_schema, dataset_name="test")

    x = torch.stack([d["procedures"] for d in dataset]).float()
    y = torch.stack([d["label"] for d in dataset]).float()

    model = LSTM(dataset=dataset, input_size=1, hidden_size=16, num_layers=1)
    output = model(x, y)
    assert "loss" in output
    assert output["loss"].dim() == 0
    print(output)
    output["loss"].backward()  
        



if __name__ == "__main__":
    test_lstm()
