"""MixLSTM model for clinical time-series prediction.

Implementation of the mixLSTM architecture from Oh et al. 2020,
"Relaxed Parameter Sharing: Effectively Modeling Time-Varying
Relationships in Clinical Time-Series" (https://arxiv.org/abs/1906.02898).

The key idea is to relax the parameter-sharing constraint of a standard
LSTM by maintaining K independent LSTM cells and combining their
parameters at each time step using learned mixing coefficients. This
allows the model to capture temporal conditional shift, where the
relationship between features and outcomes changes over time.
"""

import math
from abc import ABC
from collections import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class MLP(nn.Module):
    """A simple multi-layer perceptron used as a building block.

    Args:
        neuron_sizes: List of layer sizes, e.g. ``[in_dim, hidden, out_dim]``.
        activation: Activation function class (default: ``nn.LeakyReLU``).
        bias: Whether linear layers include a bias term.
    """

    def __init__(self, neuron_sizes, activation=nn.LeakyReLU, bias=True):
        super(MLP, self).__init__()
        self.neuron_sizes = neuron_sizes

        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])

        self.classifier = nn.Sequential(*layers[:-1])

    def eval_forward(self, x, y):
        """Run a forward pass in eval mode (ignores ``y``)."""
        self.eval()
        return self.forward(x)

    def forward(self, x):
        """Flatten the input and pass it through the MLP."""
        x = x.contiguous()
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)


############################ main models ##################################
class MoE(nn.Module):
    """Abstract base class for mixture-of-experts modules.

    Supports specifying a set of experts and a gating function. Subclasses
    must implement how experts are combined (see ``MoO`` and ``MoW``).

    Args:
        experts: The expert modules to be combined.
        gate: The gating function that produces mixing coefficients.
    """

    def __init__(self, experts, gate):
        super(MoE, self).__init__()
        self.experts = experts
        self.gate = gate


class MoO(MoE):
    """Mixture of Outputs.

    Each expert produces an output independently, and the outputs are
    combined via a weighted sum using coefficients from the gate.

    Args:
        experts: The expert modules.
        gate: The gating function.
        bs_dim: Batch-size dimension of expert outputs (default: 1).
        expert_dim: Expert dimension after stacking (default: 0).
    """

    def __init__(self, experts, gate, bs_dim=1, expert_dim=0):
        super(MoO, self).__init__(experts, gate)
        # this is for RNN architecture: bs_dim = 2 for RNN
        self.bs_dim = bs_dim
        self.expert_dim = expert_dim

    def combine(self, o, coef):
        """Combine expert outputs using the mixing coefficients."""
        if isinstance(o[0], abc.Sequence):  # account for multi_output setting
            return [self.combine(o_, coef) for o_ in zip(*o)]
        else:
            o = torch.stack(o)
            # reshape o to (_, bs, n_expert)  b/c coef is (bs, n_expert)
            o = o.transpose(self.expert_dim, -1)
            o = o.transpose(self.bs_dim, -2)

            # change back
            res = o * coef
            res = res.transpose(self.expert_dim, -1)
            res = res.transpose(self.bs_dim, -2)
            return res.sum(0)

    def forward(self, x, coef=None):
        """Compute each expert's output and combine them."""
        coef = self.gate(x, coef)  # (bs, n_expert) or n_expert
        self.last_coef = coef
        o = [expert(x) for expert in self.experts]
        return self.combine(o, coef)


class MoW(MoE):
    """Mixture of Weights.

    Instead of combining expert outputs, this module combines expert
    parameters before the forward pass, effectively producing a single
    assembled expert per time step.
    """

    def forward(self, x, coef=None):
        """Run the assembled expert on the input."""
        # assume experts has already been assembled
        coef = self.gate(x, coef)
        self.last_coef = coef
        return self.experts(x, coef)


################## sample gating functions for get_coefficients ###########
class Gate(ABC, nn.Module):
    """Abstract base class for gating functions."""

    def forward(self, x, coef=None):
        raise NotImplementedError()


class AdaptiveLSTMGate(Gate):
    """A gate that computes mixing coefficients from the LSTM hidden state.

    Args:
        input_size: Size of the hidden state used as input.
        num_experts: Number of experts in the mixture.
        normalize: If True, apply softmax to the coefficients.
    """

    def __init__(self, input_size, num_experts, normalize=False):
        super(self.__class__, self).__init__()
        self.forward_function = MLP([input_size, num_experts])
        self.normalize = normalize

    def forward(self, x, coef=None):
        """Produce mixing coefficients from the hidden state."""
        x, (h, c) = x  # h (_, bs, d)
        o = self.forward_function(h.transpose(0, 1))  # (bs, num_experts)
        if self.normalize:
            return nn.functional.softmax(o, 1)
        else:
            return o


class NonAdaptiveGate(Gate):
    """A gate with learnable (or fixed) coefficients that do not depend on x.

    Args:
        num_experts: Number of experts.
        coef: Optional initial coefficient tensor. If None, randomly init.
        fixed: If True, coefficients are not trainable.
        normalize: If True, apply softmax to the coefficients.
    """

    def __init__(self, num_experts, coef=None, fixed=False, normalize=False):
        super(self.__class__, self).__init__()
        self.normalize = normalize
        if coef is None:  # initialization
            coef = torch.ones(num_experts)
            nn.init.uniform_(coef)
        if fixed:
            coef = nn.Parameter(coef, requires_grad=False)
        else:
            coef = nn.Parameter(coef)

        self.coefficients = coef

    def forward(self, x, coef=None):
        """Return the (optionally normalized) mixing coefficients."""
        if self.normalize:
            return nn.functional.softmax(self.coefficients, 0)
        else:
            return self.coefficients


class IDGate(Gate):
    """Identity gate that passes through a previous coefficient unchanged."""

    def forward(self, x, coef):
        """Return the coefficient that was passed in."""
        return coef


################ time series example models ################
def moo_linear(in_features, out_features, num_experts, bs_dim=1, expert_dim=0):
    """Create a MoO over a set of linear layers with tied shape.

    Args:
        in_features: Input feature size.
        out_features: Output feature size.
        num_experts: Number of expert linear layers.
        bs_dim: Batch-size dimension (see ``MoO``).
        expert_dim: Expert dimension (see ``MoO``).

    Returns:
        A ``MoO`` module wrapping ``num_experts`` linear layers with an
        identity gate.
    """
    # repeat a linear model for self.num_experts times
    experts = nn.ModuleList()
    for _ in range(num_experts):
        experts.append(nn.Linear(in_features, out_features))

    # tie weights later
    return MoO(experts, IDGate(), bs_dim=bs_dim, expert_dim=expert_dim)


class mowLSTM_(nn.Module):
    """Internal helper implementing one layer of the mixture-of-weights LSTM.

    Applies a per-time-step mixture of LSTM cells by combining the input
    and hidden weight matrices across experts.

    Args:
        input_size: Input feature dimension.
        hidden_size: Hidden state dimension.
        num_experts: Number of expert cells to mix (K).
        batch_first: If True, expects input shape (batch, seq_len, dim).
    """

    def __init__(self, input_size, hidden_size, num_experts=2,
                 batch_first=False):

        super(mowLSTM_, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.batch_first = batch_first

        # build cell
        self.input_weights = moo_linear(input_size, 4 * hidden_size,
                                        self.num_experts, bs_dim=2)  # i,f,g,o
        self.hidden_weights = moo_linear(hidden_size, 4 * hidden_size,
                                         self.num_experts, bs_dim=2)
        # init same as pytorch version
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for m in self.input_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv)
        for m in self.hidden_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv)

        # maybe: layer normalization: see jeeheh's code
        # maybe: orthogonal initialization: see jeeheh's code
        # note: pytorch implementation does neither

    def rnn_step(self, x, hidden, coef):
        """Run a single LSTM step with mixed expert parameters."""
        bs = x.shape[1]
        h, c = hidden
        gates = self.input_weights(x, coef) + self.hidden_weights(h, coef)
        # maybe: layer normalization: see jeeheh's code

        ingate, forgetgate, cellgate, outgate = gates.view(bs, -1).chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c = forgetgate * c + ingate * cellgate
        h = outgate * torch.tanh(c)  # maybe use layer norm here as well
        return h, c

    def forward(self, x, hidden, coef):
        """Run the mixture LSTM over a full sequence."""
        if self.batch_first:  # change to seq_len first
            x = x.transpose(0, 1)

        seq_len = x.shape[0]
        output = []
        for t in range(seq_len):
            hidden = self.rnn_step(x[t].unsqueeze(0), hidden, coef)
            output.append(hidden[0])  # seq_len x (_, bs, d)

        output = torch.cat(output, 0)
        return output, hidden


class mowLSTM(nn.Module):
    """Stacked mixture-of-weights LSTM used internally by ``MixLSTM``.

    Handles multi-layer stacking, dropout, and the final output projection.

    Args:
        input_size: Input feature size.
        hidden_size: Hidden state size.
        num_classes: Output dimension of the final projection.
        num_experts: Number of expert cells to mix (K).
        num_layers: Number of stacked LSTM layers.
        batch_first: If True, expects input shape (batch, seq_len, dim).
        dropout: Dropout probability between layers.
        bidirectional: Whether to use a bidirectional LSTM.
        activation: Optional activation applied to the final output.
    """

    def __init__(self, input_size, hidden_size, num_classes, num_experts=2,
                 num_layers=1, batch_first=False, dropout=0,
                 bidirectional=False, activation=None):

        super(mowLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.dropouts = nn.ModuleList()

        self.h2o = moo_linear(self.num_directions * self.hidden_size,
                              self.num_classes, self.num_experts, bs_dim=2)

        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: x

        self.rnns = nn.ModuleList()
        for i in range(num_layers * self.num_directions):
            input_size = input_size if i == 0 else hidden_size
            self.rnns.append(mowLSTM_(input_size, hidden_size, num_experts,
                                      batch_first))
            self.dropouts.append(nn.Dropout(p=dropout))

    def forward(self, x, coef):
        """Forward pass through the stacked mixture LSTM."""
        x, hidden = x
        self.last_coef = coef

        h, c = hidden
        hs, cs = [], []
        for i in range(self.num_layers):
            if i != 0 and i != (self.num_layers - 1):
                x = self.dropouts[i](x)  # waste 1 dropout but no problem
            x, hidden = self.rnns[i](x, (h[i].unsqueeze(0),
                                          c[i].unsqueeze(0)), coef)
            hs.append(hidden[0])
            cs.append(hidden[1])

        h = torch.cat(hs, 0)
        c = torch.cat(cs, 0)
        o = x
        # run through prediction layer: o: (seq_len, bs, d)
        o = self.dropouts[0](o)
        o = self.h2o(o, coef)
        o = self.activation(o)

        return o, (h, c)


class ExampleMowLSTM(nn.Module):
    """Wrapper that instantiates a mixture LSTM with per-time-step gates.

    For each of the ``t`` time steps, a separate ``NonAdaptiveGate`` is
    created so that the mixing coefficients can vary over time. All gates
    share the same underlying experts.

    Args:
        input_size: Input feature size.
        hidden_size: Hidden state size.
        num_classes: Output dimension.
        num_layers: Number of stacked LSTM layers.
        num_directions: 1 (unidirectional) or 2 (bidirectional).
        dropout: Dropout probability.
        activation: Optional output activation.
    """

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleMowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t):
        """Configure the model for ``k`` experts and ``t`` time steps.

        Args:
            k: Number of expert cells to mix.
            t: Maximum number of time steps; one gate is created per step.
        """
        self.k = k
        self.T = t
        self.cells = nn.ModuleList()

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional=(self.num_directions == 2),
                          activation=self.activation)
        self.experts = experts

        for _ in range(t):
            gate = NonAdaptiveGate(self.k, normalize=True)
            # gate = AdaptiveLSTMGate(self.hidden_size *\
            #                         self.num_layers *\
            #                         self.num_directions,
            #                         self.k,
            #                         normalize=True)
            self.cells.append(MoW(experts, gate))

    def forward(self, x, hidden):
        """Run the mixture LSTM step-by-step using the per-step gates."""
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)

        o = torch.cat(o, 0)  # (seq_len, bs, d)
        return o, hidden


def orthogonal(shape):
    """Generate an orthogonal matrix of the given shape via SVD."""
    flat_shape = (int(shape[0]), int(np.prod(shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def lstm_ortho_initializer(shape, scale=1.0):
    """Initialize LSTM weights with orthogonal blocks for each of the 4 gates.

    Args:
        shape: Target shape where the second dimension must be divisible by 4.
        scale: Scalar to multiply the orthogonal matrices by.

    Returns:
        A numpy array of the requested shape.
    """
    size_x = shape[0]
    size_h = int(shape[1] / 4)  # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h]) * scale
    t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 2:size_h * 3] = orthogonal([size_x, size_h]) * scale
    t[:, size_h * 3:] = orthogonal([size_x, size_h]) * scale
    return t


class MixLSTM(BaseModel):
    """Mixture-of-LSTMs model for clinical time-series prediction.

    Implements the mixLSTM architecture from Oh et al. 2020 for handling
    temporal conditional shift: settings where the relationship between
    input features and the target changes over time. Instead of sharing a
    single set of LSTM parameters across all time steps, MixLSTM maintains
    ``num_experts`` independent LSTM cells and, at every time step,
    computes a learned convex combination of their parameters using mixing
    coefficients constrained to the simplex. This enables smooth
    transitions between different temporal dynamics without hard segment
    boundaries.

    The model inherits from PyHealth's ``BaseModel`` and infers the input
    dimension and sequence length from the ``SampleDataset`` passed at
    construction time, so it can be used with any existing PyHealth task
    whose input is a time-series tensor.

    The model supports two operating modes that are chosen automatically
    based on the dataset's output schema:

    * Standard classification (``binary``, ``multiclass``, ``multilabel``,
      or ``regression``): predictions are taken from the final time step of
      the sequence and the appropriate PyHealth loss function is applied.
    * Per-timestep regression (when the output schema is a raw ``tensor``):
      the model outputs a value at every time step and the MSE loss is
      computed over timesteps beginning at ``prev_used_timestamps``. This
      reproduces the synthetic copy-memory task described in Section 4.1
      of the paper.

    Paper:
        Oh et al. 2020, "Relaxed Parameter Sharing: Effectively Modeling
        Time-Varying Relationships in Clinical Time-Series."
        https://arxiv.org/abs/1906.02898

    Args:
        dataset: A ``SampleDataset`` used to infer input feature size and
            sequence length from the first sample.
        num_experts: Number of expert LSTM cells to mix (``K`` in the paper).
            Higher values give the model more flexibility to vary parameters
            over time at the cost of more parameters. Defaults to 2.
        hidden_size: Size of the LSTM hidden state. Defaults to 100.
        prev_used_timestamps: For the per-timestep regression mode, the
            index of the first time step at which the loss is computed.
            Earlier time steps are skipped because their targets are
            trivially defined in the synthetic task. Ignored in the
            standard classification mode. Defaults to 0.

    Attributes:
        input_size: Inferred input feature dimension.
        time_steps: Inferred sequence length.
        hidden_size: LSTM hidden state size.
        _per_timestep: ``True`` when the model is running in per-timestep
            regression mode, ``False`` when it is running in standard
            classification mode.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import MixLSTM
        >>> samples = [
        ...     {
        ...         "patient_id": f"p-{i}",
        ...         "visit_id": "v-0",
        ...         "series": torch.randn(48, 76).numpy().tolist(),
        ...         "label": int(i % 2),
        ...     }
        ...     for i in range(100)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"series": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="demo",
        ... )
        >>> model = MixLSTM(dataset=dataset, num_experts=2, hidden_size=64)
    """

    def __init__(self, dataset: SampleDataset, num_experts=2, hidden_size=100,
                 prev_used_timestamps=0):
        super(MixLSTM, self).__init__(dataset)

        # Identify primary input key and infer shape
        input_keys = list(dataset.input_processors.keys())
        self.input_key = input_keys[0]
        self.label_key = self.label_keys[0] if self.label_keys else None

        sample = dataset[0]
        val = sample[self.input_key]
        if isinstance(val, (list, tuple)):
            for item in val:
                if torch.is_tensor(item) or isinstance(
                        item, (list, tuple, np.ndarray)):
                    val = item
                    break
        if torch.is_tensor(val):
            input_dim = val.shape[-1] if val.dim() >= 2 else 1
            T = val.shape[0]
        else:
            arr = np.array(val)
            input_dim = arr.shape[-1] if arr.ndim >= 2 else 1
            T = len(val)

        self.input_size = int(input_dim)
        self.time_steps = int(T)
        self.prev_used_timestamps = prev_used_timestamps

        # Detect per-timestep regression: output target is a tensor, not a
        # standard label type.  In that case self.mode is None / unrecognised.
        self._per_timestep = (
            self.mode not in ("binary", "multiclass", "multilabel", "regression")
        )

        if self._per_timestep:
            num_classes = 1  # predict one scalar per timestep
        else:
            num_classes = int(self.get_output_size())

        self.model = ExampleMowLSTM(self.input_size, hidden_size,
                                    num_classes, num_layers=1,
                                    num_directions=1, dropout=0,
                                    activation=None)

        self.num_layers = 1
        self.num_directions = 1
        self.hidden_size = hidden_size
        self.model.setKT(num_experts, self.time_steps)

    def forward(self, **kwargs):
        """Run a forward pass.

        Expects the input tensor under ``kwargs[self.input_key]`` with
        shape ``(batch, seq_len, input_dim)``. If a label tensor is also
        provided under ``self.label_key``, the appropriate loss is
        computed and returned.

        Args:
            **kwargs: Batch dictionary, typically produced by a PyHealth
                DataLoader and passed by ``Trainer`` as ``model(**batch)``.

        Returns:
            A dictionary with the following keys, where shapes depend on
            which mode the model is operating in:

            * Classification mode (``_per_timestep = False``):
                - ``logit``: ``(batch, num_classes)`` from the final step.
                - ``y_prob``: Probabilities produced by ``prepare_y_prob``.
                - ``loss`` (optional): PyHealth's standard loss for the task.
                - ``y_true`` (optional): Ground-truth labels.

            * Per-timestep regression mode (``_per_timestep = True``):
                - ``logit``: ``(batch, seq_len, 1)`` — one prediction per
                    time step.
                - ``y_prob``: Same tensor as ``logit``.
                - ``loss`` (optional): MSE computed over time steps from
                    ``prev_used_timestamps`` onward.
                - ``y_true`` (optional): Ground-truth target tensor.
        """
        x = kwargs.get(self.input_key)

        # (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        batch_size = x.size(1)
        device = self.device
        h = torch.zeros(self.num_layers * self.num_directions,
                        batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers * self.num_directions,
                        batch_size, self.hidden_size, device=device)

        outputs, _ = self.model(x, (h, c))
        # (seq_len, bs, out) => (bs, seq_len, out)
        logits_seq = outputs.permute(1, 0, 2)

        if self._per_timestep:
            # --- Per-timestep regression (original MLHC2019 synthetic task)
            results = {"logit": logits_seq, "y_prob": logits_seq}
            if self.label_key and self.label_key in kwargs:
                y_true = kwargs[self.label_key].to(device)
                if y_true.dim() == 2:
                    y_true = y_true.unsqueeze(-1)
                l = self.prev_used_timestamps
                pred = logits_seq[:, l:, :].contiguous()
                target = y_true[:, l:, :].contiguous()
                loss = F.mse_loss(pred.view(-1, pred.size(-1)),
                                  target.view(-1, target.size(-1)))
                results["loss"] = loss
                results["y_true"] = y_true
            return results

        logits = logits_seq[:, -1, :]
        y_prob = self.prepare_y_prob(logits)
        results = {"logit": logits, "y_prob": y_prob}

        if self.label_key and self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def after_backward(self):
        """Hook called after backward(); no-op for this model."""
        pass
