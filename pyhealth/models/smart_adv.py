import torch
from torch.nn import Module


class SmartAdversary(Module):
    
    
    
    """
    1. Name(s): Yeseong Jeon, Taoran Shen
    2. NetID(s): yeseong2, taorans2
    3. Paper title: An Adversarial Approach for the Robust Classification of Pneumonia from Chest Radiographs
    4. Paper link: https://arxiv.org/abs/2001.04051
    5. Description: This file implements SmartAdversary, an improved adversarial model for confounder removal
    in chest X-ray classification. The model extends the standard adversarial classifier by adding
    Batch Normalization and Dropout to improve stability and generalization. The model was tested
    on the NIH dataset for View Position confounder removal.

    Extension of the standard Adversarial classifier with added improvements:
    - Batch Normalization
    - Dropout

    This model is designed to improve stability and generalization by reducing overfitting and enhancing convergence during adversarial training. It can be used in settings where confounder removal (e.g., View Position) is required.

    Args:
        n_sensitive (int): Number of sensitive attributes to predict.
        n_hidden (int): Number of hidden units in each hidden layer. Default is 128.
        dropout_rate (float): Dropout rate applied after each hidden layer. Default is 0.3.

    Examples:
        >>> from pyhealth.models import SmartAdversary
        >>> adv = SmartAdversary(n_sensitive=1)
        >>> x = torch.randn(10, 1)
        >>> output = adv(x)
        >>> output.shape
        torch.Size([10, 1])
    """
    def __init__(self, n_sensitive, n_hidden=128, dropout_rate=0.3):
        super(SmartAdversary, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.Dropout(dropout_rate),

            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.Dropout(dropout_rate),

            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.Dropout(dropout_rate),

            torch.nn.Linear(n_hidden, n_sensitive)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SmartAdversary model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_sensitive).
        """
        return torch.sigmoid(self.network(x))
