from __future__ import annotations

import torch
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


__all__ = ["NeuralFG"]


def _value_tensor(
    value: Tensor | tuple[Tensor, ...],
) -> Tensor:
    """Extract the primary tensor emitted by a PyHealth processor."""

    if isinstance(value, tuple):
        return value[0]

    return value


def _infer_input_dim(
    dataset: SampleDataset,
    feature_key: str,
) -> int:
    """Infer dense feature dimension from one processed sample."""

    sample = dataset[0]

    if feature_key not in sample:
        raise ValueError(
            f"feature_key {feature_key!r} is not present in the dataset"
        )

    value = sample[feature_key]

    if isinstance(value, tuple):
        value = value[0]

    tensor = torch.as_tensor(value)

    if tensor.ndim != 1:
        raise ValueError(
            "NeuralFG expects each patient's features to be "
            "a 1D dense numeric vector."
        )

    return int(tensor.shape[0])


class PositiveLinear(nn.Module):
    """Linear layer with non-negative effective weights.

    Neural Fine-Gray constrains monotonic-network weights to be
    positive. The trainable raw weights remain unconstrained, while
    their squared values are used during the forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be positive")

        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features

        self.raw_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize trainable parameters."""

        nn.init.xavier_uniform_(self.raw_weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def positive_weight(self) -> Tensor:
        """Return non-negative effective weights."""

        return self.raw_weight.square()

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply positive-weight linear transformation."""

        return nn.functional.linear(
            x,
            self.positive_weight,
            self.bias,
        )


class MonotonicNetwork(nn.Module):
    """Positive monotonic network for one competing risk.

    The network receives time together with the shared patient
    representation. Positive weights and monotonic activation
    functions preserve monotonicity with respect to time.

    The final Softplus guarantees a positive output
    M_r(t, E(x)).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if num_hidden_layers <= 0:
            raise ValueError(
                "num_hidden_layers must be positive"
            )

        layers: list[nn.Module] = []

        width = input_dim

        for _ in range(num_hidden_layers):
            layers.append(
                PositiveLinear(
                    in_features=width,
                    out_features=hidden_dim,
                )
            )

            layers.append(nn.Tanh())

            width = hidden_dim

        layers.append(
            PositiveLinear(
                in_features=width,
                out_features=1,
            )
        )

        layers.append(nn.Softplus())

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return positive monotonic-network output."""

        return self.network(x)


class _NeuralFGNetwork(nn.Module):
    """Core Neural Fine-Gray network.

    Implements:

    1. Shared covariate embedding E(x)
    2. Balancing network B(E(x))
    3. Event-specific monotonic networks M_r(t, E(x))
    4. Event-specific cumulative incidence functions
    5. Automatic differentiation of CIFs with respect to time
    6. Competing-risk negative log-likelihood
    """

    def __init__(
        self,
        input_dim: int,
        num_risks: int,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        embedding_layers: int = 2,
        monotonic_layers: int = 2,
        dropout: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if num_risks <= 0:
            raise ValueError(
                "num_risks must be positive"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive"
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if embedding_layers <= 0:
            raise ValueError(
                "embedding_layers must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        self.input_dim = int(input_dim)
        self.num_risks = int(num_risks)
        self.embedding_dim = int(embedding_dim)
        self.eps = float(eps)

        # ---------------------------------------------------------
        # E(x): shared embedding network
        # ---------------------------------------------------------

        embedding_modules: list[nn.Module] = []

        width = input_dim

        for _ in range(embedding_layers):

            embedding_modules.append(
                nn.Linear(
                    width,
                    embedding_dim,
                )
            )

            embedding_modules.append(
                nn.Tanh()
            )

            if dropout > 0:
                embedding_modules.append(
                    nn.Dropout(dropout)
                )

            width = embedding_dim

        self.embedding_network = nn.Sequential(
            *embedding_modules
        )

        # ---------------------------------------------------------
        # B(E(x)): balancing network
        #
        # Keep logits separate from Softmax so PyHealth can expose
        # true raw outputs under "logit".
        # ---------------------------------------------------------

        self.balancing_hidden = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

        self.balancing_head = nn.Linear(
            hidden_dim,
            num_risks,
        )

        self.balancing_softmax = nn.Softmax(
            dim=-1
        )

        # ---------------------------------------------------------
        # M_r(t, E(x)): one network per competing risk
        # ---------------------------------------------------------

        monotonic_input_dim = (
            embedding_dim + 1
        )

        self.monotonic_networks = nn.ModuleList(
            [
                MonotonicNetwork(
                    input_dim=monotonic_input_dim,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=monotonic_layers,
                )
                for _ in range(num_risks)
            ]
        )

    def embed(
        self,
        features: Tensor,
    ) -> Tensor:
        """Return E(x), the shared patient representation."""

        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, input_dim]"
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {features.shape[1]}"
            )

        return self.embedding_network(
            features
        )

    def balancing_logits(
        self,
        embedding: Tensor,
    ) -> Tensor:
        """Return raw balancing-network logits."""

        hidden = self.balancing_hidden(
            embedding
        )

        return self.balancing_head(
            hidden
        )

    def balancing_probabilities(
        self,
        embedding: Tensor,
    ) -> Tensor:
        """Return B(E(x)).

        Shape:
            [batch, num_risks]

        Every row sums to one.
        """

        logits = self.balancing_logits(
            embedding
        )

        return self.balancing_softmax(
            logits
        )

    def monotonic_outputs(
        self,
        embedding: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Return M_r(t, E(x)) for all competing risks.

        Returns:
            Tensor with shape:
                [batch, num_risks]
        """

        time = time.reshape(-1, 1)

        if torch.any(time < 0):
            raise ValueError(
                "time must be non-negative"
            )

        if embedding.shape[0] != time.shape[0]:
            raise ValueError(
                "embedding and time must have "
                "the same batch size"
            )

        monotonic_input = torch.cat(
            [
                time,
                embedding,
            ],
            dim=-1,
        )

        outputs = [
            network(monotonic_input)
            for network in self.monotonic_networks
        ]

        return torch.cat(
            outputs,
            dim=-1,
        )

    def conditional_cdf(
        self,
        embedding: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Calculate conditional event-time CDFs.

        Computes:

            1 - exp(
                -t * M_r(t, E(x))
            )

        Returns:
            Tensor with shape:
                [batch, num_risks]
        """

        time = time.reshape(-1, 1)

        monotonic_value = (
            self.monotonic_outputs(
                embedding,
                time,
            )
        )

        return 1.0 - torch.exp(
            -time * monotonic_value
        )

    def cif(
        self,
        features: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Compute event-specific cumulative incidence functions.

        F_r(t | x) =
            B(E(x))_r *
            [1 - exp(-t * M_r(t, E(x)))]

        Returns:
            Tensor with shape:
                [batch, num_risks]
        """

        embedding = self.embed(
            features
        )

        balance = (
            self.balancing_probabilities(
                embedding
            )
        )

        conditional = (
            self.conditional_cdf(
                embedding,
                time,
            )
        )

        return balance * conditional

    def raw_outputs(
        self,
        features: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Return raw NeuralFG outputs.

        The raw output combines:

        1. Balancing-network logits
        2. Event-specific monotonic-network outputs

        Returns:
            Tensor with shape:
                [batch, 2 * num_risks]
        """

        embedding = self.embed(
            features
        )

        balance_logits = (
            self.balancing_logits(
                embedding
            )
        )

        monotonic_values = (
            self.monotonic_outputs(
                embedding,
                time,
            )
        )

        return torch.cat(
            [
                balance_logits,
                monotonic_values,
            ],
            dim=-1,
        )

    def cif_and_density(
        self,
        features: Tensor,
        time: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return CIF and dF_r(t|x)/dt.

        Automatic differentiation is used to obtain
        the derivative of every event-specific CIF
        with respect to time.

        Returns:
            cif:
                Tensor with shape:
                    [batch, num_risks]

            density:
                Tensor with shape:
                    [batch, num_risks]
        """

        time = time.reshape(-1, 1)

        if not time.requires_grad:
            time = (
                time.clone()
                .detach()
                .requires_grad_(True)
            )

        cif = self.cif(
            features=features,
            time=time,
        )

        densities = []

        for risk_idx in range(
            self.num_risks
        ):

            risk_cif = cif[
                :,
                risk_idx,
            ]

            derivative = torch.autograd.grad(
                outputs=risk_cif.sum(),
                inputs=time,
                create_graph=True,
                retain_graph=True,
            )[0]

            densities.append(
                derivative
            )

        density = torch.cat(
            densities,
            dim=-1,
        )

        return cif, density

    def competing_risk_nll(
        self,
        features: Tensor,
        time: Tensor,
        event: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute competing-risk negative log-likelihood.

        Event coding:

            0 = right censored
            1 = competing risk 1
            2 = competing risk 2
            ...
            R = competing risk R

        Observed event contribution:

            log(
                dF_r(t | x) / dt
            )

        Censored contribution:

            log(
                1 - sum_r F_r(t | x)
            )
        """

        event = (
            event.reshape(-1)
            .long()
        )

        time = time.reshape(
            -1,
            1,
        )

        batch_size = (
            features.shape[0]
        )

        if time.shape[0] != batch_size:
            raise ValueError(
                "features and time must have "
                "the same batch size"
            )

        if event.shape[0] != batch_size:
            raise ValueError(
                "features and event must have "
                "the same batch size"
            )

        if torch.any(event < 0):
            raise ValueError(
                "event labels cannot be negative"
            )

        if torch.any(
            event > self.num_risks
        ):
            raise ValueError(
                "event labels cannot exceed "
                "num_risks"
            )

        cif, density = (
            self.cif_and_density(
                features=features,
                time=time,
            )
        )

        # ---------------------------------------------------------
        # Censored contribution
        #
        # S(t|x) =
        #     1 - sum_r F_r(t|x)
        # ---------------------------------------------------------

        survival = (
            1.0 - cif.sum(dim=-1)
        ).clamp_min(
            self.eps
        )

        censored_log_likelihood = (
            torch.log(survival)
        )

        # ---------------------------------------------------------
        # Observed-event contribution
        # ---------------------------------------------------------

        risk_index = (
            event - 1
        ).clamp(
            min=0
        )

        event_density = (
            density.gather(
                dim=1,
                index=risk_index.unsqueeze(1),
            )
            .squeeze(1)
            .clamp_min(self.eps)
        )

        event_log_likelihood = (
            torch.log(
                event_density
            )
        )

        observed = event > 0

        log_likelihood = torch.where(
            observed,
            event_log_likelihood,
            censored_log_likelihood,
        )

        negative_log_likelihood = (
            -log_likelihood
        )

        if reduction == "none":
            return negative_log_likelihood

        if reduction == "sum":
            return (
                negative_log_likelihood.sum()
            )

        if reduction == "mean":
            return (
                negative_log_likelihood.mean()
            )

        raise ValueError(
            "reduction must be "
            "'none', 'sum', or 'mean'"
        )


class NeuralFG(BaseModel):
    """Neural Fine-Gray model for competing-risk survival analysis.

    NeuralFG directly models event-specific cumulative incidence
    functions using:

    1. A shared embedding network E(x)
    2. A balancing network B(E(x))
    3. Event-specific positive monotonic networks
    4. A competing-risk likelihood

    Event coding:

        0 = right censored
        1 = competing risk 1
        2 = competing risk 2
        ...
        R = competing risk R

    Args:
        dataset:
            Fitted PyHealth SampleDataset.

        feature_key:
            Input field containing dense patient covariates.

        duration_key:
            Output field containing observed event or
            censoring time.

        event_key:
            Output field containing censoring/event type.

        input_dim:
            Number of input covariates. If None, infer it
            from the dataset.

        num_risks:
            Number of competing event types excluding
            censoring.

        hidden_dim:
            Hidden dimension of the balancing and monotonic
            networks.

        embedding_dim:
            Dimension of the shared patient representation.

        embedding_layers:
            Number of layers in the embedding network.

        monotonic_layers:
            Number of hidden layers in each event-specific
            monotonic network.

        dropout:
            Dropout probability in the embedding network.

        time_grid:
            Time points at which forward() returns CIF
            predictions.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        num_risks: int = 2,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        embedding_layers: int = 2,
        monotonic_layers: int = 2,
        dropout: float = 0.25,
        time_grid: Tensor | None = None,
    ) -> None:
        super().__init__(dataset)

        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.num_risks = int(
            num_risks
        )

        # NeuralFG uses its own competing-risk likelihood
        # rather than a built-in classification/regression loss.
        # self.mode = None
        # BaseModel.mode is a read-only property.

        # ---------------------------------------------------------
        # Validate dataset fields
        # ---------------------------------------------------------

        if feature_key not in dataset.input_schema:
            raise ValueError(
                f"feature_key {feature_key!r} "
                "is not in dataset.input_schema"
            )

        if duration_key not in dataset.output_schema:
            raise ValueError(
                f"duration_key {duration_key!r} "
                "is not in dataset.output_schema"
            )

        if event_key not in dataset.output_schema:
            raise ValueError(
                f"event_key {event_key!r} "
                "is not in dataset.output_schema"
            )

        if num_risks <= 0:
            raise ValueError(
                "num_risks must be positive"
            )

        # ---------------------------------------------------------
        # Determine feature dimension
        # ---------------------------------------------------------

        if input_dim is None:
            input_dim = _infer_input_dim(
                dataset,
                feature_key,
            )

        self.input_dim = int(
            input_dim
        )

        # ---------------------------------------------------------
        # Core NeuralFG network
        # ---------------------------------------------------------

        self.network = _NeuralFGNetwork(
            input_dim=self.input_dim,
            num_risks=num_risks,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            embedding_layers=embedding_layers,
            monotonic_layers=monotonic_layers,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Default prediction grid
        #
        # The reproduction experiment should later provide
        # q25, q50, and q75 evaluation times explicitly.
        # ---------------------------------------------------------

        if time_grid is None:
            time_grid = torch.linspace(
                0.1,
                10.0,
                50,
            )

        time_grid = torch.as_tensor(
            time_grid,
            dtype=torch.float32,
        )

        if time_grid.ndim != 1:
            raise ValueError(
                "time_grid must be "
                "one-dimensional"
            )

        if torch.any(time_grid < 0):
            raise ValueError(
                "time_grid cannot contain "
                "negative values"
            )

        self.register_buffer(
            "time_grid",
            time_grid.reshape(-1),
        )

    def _predict_cif_grid(
        self,
        features: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict CIFs at multiple times.

        Args:
            features:
                Tensor with shape:
                    [batch, input_dim]

            times:
                Tensor with shape:
                    [num_times]

        Returns:
            Tensor with shape:
                [batch, num_times, num_risks]
        """

        batch_size = (
            features.shape[0]
        )

        num_times = (
            times.numel()
        )

        expanded_features = (
            features.unsqueeze(1)
            .expand(
                batch_size,
                num_times,
                self.input_dim,
            )
            .reshape(
                batch_size * num_times,
                self.input_dim,
            )
        )

        expanded_times = (
            times.reshape(1, -1)
            .expand(
                batch_size,
                num_times,
            )
            .reshape(
                batch_size * num_times,
                1,
            )
        )

        cif = self.network.cif(
            features=expanded_features,
            time=expanded_times,
        )

        return cif.reshape(
            batch_size,
            num_times,
            self.num_risks,
        )

    def forward(
        self,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run one NeuralFG forward pass.

        The batch must contain:

            features
            duration
            event

        Returns:
            Dictionary containing:

                loss:
                    Competing-risk negative log-likelihood.

                logit:
                    Raw NeuralFG network outputs.

                y_prob:
                    CIF predictions over self.time_grid.

                y_true:
                    Observed duration and event type.

                observed_cif:
                    CIF evaluated at each patient's
                    observed follow-up time.
        """

        required_keys = (
            self.feature_key,
            self.duration_key,
            self.event_key,
        )

        missing = [
            key
            for key in required_keys
            if key not in kwargs
        ]

        if missing:
            raise KeyError(
                f"Missing required batch fields: "
                f"{missing}"
            )

        # ---------------------------------------------------------
        # Extract PyHealth tensors
        # ---------------------------------------------------------

        features = _value_tensor(
            kwargs[self.feature_key]
        )

        duration = _value_tensor(
            kwargs[self.duration_key]
        )

        event = _value_tensor(
            kwargs[self.event_key]
        )

        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )

        duration = torch.as_tensor(
            duration,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        event = torch.as_tensor(
            event,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, input_dim]"
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {features.shape[1]}"
            )

        if features.shape[0] != duration.shape[0]:
            raise ValueError(
                "features and duration must have "
                "the same batch size"
            )

        if features.shape[0] != event.shape[0]:
            raise ValueError(
                "features and event must have "
                "the same batch size"
            )

        if not torch.isfinite(
            features
        ).all():
            raise ValueError(
                "features must contain only "
                "finite values"
            )

        if not torch.isfinite(
            duration
        ).all():
            raise ValueError(
                "duration must contain only "
                "finite values"
            )

        if torch.any(duration < 0):
            raise ValueError(
                "duration cannot contain "
                "negative values"
            )

        if torch.any(event < 0):
            raise ValueError(
                "event labels cannot be negative"
            )

        if torch.any(
            event > self.num_risks
        ):
            raise ValueError(
                "event labels cannot exceed "
                "num_risks"
            )

        # ---------------------------------------------------------
        # Training loss
        # ---------------------------------------------------------

        loss = (
            self.network.competing_risk_nll(
                features=features,
                time=duration,
                event=event,
            )
        )

        # ---------------------------------------------------------
        # CIF at observed patient times
        # ---------------------------------------------------------

        observed_cif = (
            self.network.cif(
                features=features,
                time=duration,
            )
        )

        # ---------------------------------------------------------
        # Raw network outputs
        # ---------------------------------------------------------

        logits = (
            self.network.raw_outputs(
                features=features,
                time=duration,
            )
        )

        # ---------------------------------------------------------
        # CIF curves over common evaluation grid
        # ---------------------------------------------------------

        y_prob = (
            self._predict_cif_grid(
                features=features,
                times=self.time_grid,
            )
        )

        # ---------------------------------------------------------
        # True survival outcomes
        #
        # column 0 = duration
        # column 1 = event type
        # ---------------------------------------------------------

        y_true = torch.stack(
            [
                duration,
                event.float(),
            ],
            dim=-1,
        )

        return {
            "loss": loss,
            "logit": logits,
            "y_prob": y_prob,
            "y_true": y_true,
            "observed_cif": observed_cif,
        }

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        **kwargs: Tensor | tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        """Run forward using dense features directly.

        NeuralFG operates directly on dense numeric features,
        so supplied embeddings can be passed as the feature tensor.
        """

        kwargs[self.feature_key] = (
            embeddings
        )

        return self.forward(
            **kwargs
        )

    @torch.no_grad()
    def predict_cif(
        self,
        features: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict cumulative incidence functions.

        Args:
            features:
                Patient covariates with shape:
                    [batch, input_dim]

            times:
                Evaluation times with shape:
                    [num_times]

        Returns:
            CIF tensor with shape:
                [batch, num_times, num_risks]
        """

        features = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        )

        times = torch.as_tensor(
            times,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)

        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, input_dim]"
            )

        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {features.shape[1]}"
            )

        if torch.any(times < 0):
            raise ValueError(
                "times cannot contain "
                "negative values"
            )

        previous_mode = self.training

        try:
            self.eval()

            return self._predict_cif_grid(
                features=features,
                times=times,
            )

        finally:
            self.train(
                previous_mode
            )
