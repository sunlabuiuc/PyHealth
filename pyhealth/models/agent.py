"""Dr. Agent model for PyHealth 2.0.

Author: Joshua Steier
Paper: Dr. Agent: Clinical predictive model via mimicked second opinions
Link: https://doi.org/10.1093/jamia/ocaa074
Description: Multi-agent reinforcement learning model with dynamic skip
    connections for clinical prediction tasks. Uses two policy gradient
    agents (primary and second-opinion) to capture long-term dependencies
    in patient EHR sequences.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit


class AgentLayer(nn.Module):
    """Dr. Agent layer with dual-agent dynamic skip connections.

    This layer implements the core mechanism from the Dr. Agent paper:
    two policy gradient agents select optimal historical hidden states
    to capture long-term dependencies in sequential EHR data.

    Args:
        input_dim: Input feature dimension.
        static_dim: Static feature dimension. If 0, no static features used.
        cell: RNN cell type, one of "gru" or "lstm". Default is "gru".
        use_baseline: Whether to use baseline for variance reduction in
            REINFORCE. Default is True.
        n_actions: Number of historical states to consider (K in paper).
            Default is 10.
        n_units: Hidden units in agent MLPs. Default is 64.
        n_hidden: Hidden units in RNN cell. Default is 128.
        dropout: Dropout rate applied to final output. Default is 0.5.
        lamda: Weight for combining agent-selected state with current state.
            h_combined = lamda * h_agent + (1 - lamda) * h_current.
            Default is 0.5.

    Examples:
        >>> from pyhealth.models.agent import AgentLayer
        >>> layer = AgentLayer(input_dim=64, static_dim=12)
        >>> x = torch.randn(32, 50, 64)  # [batch, seq_len, features]
        >>> static = torch.randn(32, 12)  # [batch, static_dim]
        >>> last_out, all_out = layer(x, static=static)
        >>> last_out.shape
        torch.Size([32, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        cell: str = "gru",
        use_baseline: bool = True,
        n_actions: int = 10,
        n_units: int = 64,
        n_hidden: int = 128,
        dropout: float = 0.5,
        lamda: float = 0.5,
    ):
        super(AgentLayer, self).__init__()

        if cell not in ["gru", "lstm"]:
            raise ValueError("cell must be 'gru' or 'lstm'")

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = n_hidden
        self.static_dim = static_dim

        # Agent state storage (reset each forward pass)
        self.agent1_action: List[torch.Tensor] = []
        self.agent1_prob: List[torch.Tensor] = []
        self.agent1_entropy: List[torch.Tensor] = []
        self.agent1_baseline: List[torch.Tensor] = []
        self.agent2_action: List[torch.Tensor] = []
        self.agent2_prob: List[torch.Tensor] = []
        self.agent2_entropy: List[torch.Tensor] = []
        self.agent2_baseline: List[torch.Tensor] = []

        # Agent 1 (history agent): observes mean of historical hidden states
        self.agent1_fc1 = nn.Linear(self.n_hidden + self.static_dim, self.n_units)
        self.agent1_fc2 = nn.Linear(self.n_units, self.n_actions)

        # Agent 2 (primary agent): observes current input
        self.agent2_fc1 = nn.Linear(self.input_dim + self.static_dim, self.n_units)
        self.agent2_fc2 = nn.Linear(self.n_units, self.n_actions)

        # Baseline networks for variance reduction
        if use_baseline:
            self.agent1_value = nn.Linear(self.n_units, 1)
            self.agent2_value = nn.Linear(self.n_units, 1)

        # RNN cell
        if self.cell == "lstm":
            self.rnn = nn.LSTMCell(self.input_dim, self.n_hidden)
        else:
            self.rnn = nn.GRUCell(self.input_dim, self.n_hidden)

        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        # Dropout layer
        if dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=dropout)

        # Static feature integration layers
        if self.static_dim > 0:
            self.init_h = nn.Linear(self.static_dim, self.n_hidden)
            self.init_c = nn.Linear(self.static_dim, self.n_hidden)
            self.fusion = nn.Linear(self.n_hidden + self.static_dim, self.fusion_dim)

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def _choose_action(
        self,
        observation: torch.Tensor,
        agent: int = 1,
    ) -> torch.Tensor:
        """Select action (history index) based on observation.

        Each agent observes its environment and samples an action from
        a categorical distribution over K historical states.

        Args:
            observation: Environment observation of shape [batch, obs_dim].
            agent: Agent identifier (1=history agent, 2=primary agent).

        Returns:
            Selected action indices of shape [batch, 1].
        """
        observation = observation.detach()

        if agent == 1:
            hidden = self.tanh(self.agent1_fc1(observation))
            logits = self.agent1_fc2(hidden)
            if self.use_baseline:
                self.agent1_baseline.append(self.agent1_value(hidden))
        else:
            hidden = self.tanh(self.agent2_fc1(observation))
            logits = self.agent2_fc2(hidden)
            if self.use_baseline:
                self.agent2_baseline.append(self.agent2_value(hidden))

        probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()

        if agent == 1:
            self.agent1_entropy.append(dist.entropy())
            self.agent1_action.append(actions.unsqueeze(-1))
            self.agent1_prob.append(dist.log_prob(actions))
        else:
            self.agent2_entropy.append(dist.entropy())
            self.agent2_action.append(actions.unsqueeze(-1))
            self.agent2_prob.append(dist.log_prob(actions))

        return actions.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation through the Dr. Agent layer.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim].
            static: Optional static features of shape [batch, static_dim].
            mask: Optional mask of shape [batch, seq_len] where True/1
                indicates valid timesteps.

        Returns:
            last_output: Final hidden state of shape [batch, fusion_dim].
            all_outputs: All hidden states of shape [batch, seq_len, fusion_dim].
        """
        batch_size = x.size(0)
        time_step = x.size(1)

        # Reset agent state
        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        # Initialize hidden state
        if self.static_dim > 0 and static is not None:
            cur_h = self.init_h(static)
            if self.cell == "lstm":
                cur_c = self.init_c(static)
        else:
            cur_h = torch.zeros(
                batch_size, self.n_hidden, dtype=torch.float32, device=x.device
            )
            if self.cell == "lstm":
                cur_c = torch.zeros(
                    batch_size, self.n_hidden, dtype=torch.float32, device=x.device
                )

        h_list = []
        for t in range(time_step):
            cur_input = x[:, t, :]

            if t == 0:
                # First timestep: initialize history buffer
                obs_1 = cur_h
                obs_2 = cur_input

                if self.static_dim > 0 and static is not None:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                self._choose_action(obs_1, agent=1)
                self._choose_action(obs_2, agent=2)

                # Initialize history buffer with zeros
                observed_h = (
                    torch.zeros_like(cur_h)
                    .view(-1)
                    .repeat(self.n_actions)
                    .view(self.n_actions, batch_size, self.n_hidden)
                )
                action_h = cur_h

                if self.cell == "lstm":
                    observed_c = (
                        torch.zeros_like(cur_c)
                        .view(-1)
                        .repeat(self.n_actions)
                        .view(self.n_actions, batch_size, self.n_hidden)
                    )
                    action_c = cur_c
            else:
                # Update history buffer (sliding window)
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), dim=0)

                # Agent observations
                obs_1 = observed_h.mean(dim=0)
                obs_2 = cur_input

                if self.static_dim > 0 and static is not None:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                # Select actions
                act_idx1 = self._choose_action(obs_1, agent=1).long()
                act_idx2 = self._choose_action(obs_2, agent=2).long()

                # Gather selected hidden states
                batch_idx = torch.arange(
                    batch_size, dtype=torch.long, device=x.device
                ).unsqueeze(-1)
                action_h1 = observed_h[act_idx1, batch_idx, :].squeeze(1)
                action_h2 = observed_h[act_idx2, batch_idx, :].squeeze(1)
                action_h = (action_h1 + action_h2) / 2

                if self.cell == "lstm":
                    observed_c = torch.cat(
                        (observed_c[1:], cur_c.unsqueeze(0)), dim=0
                    )
                    action_c1 = observed_c[act_idx1, batch_idx, :].squeeze(1)
                    action_c2 = observed_c[act_idx2, batch_idx, :].squeeze(1)
                    action_c = (action_c1 + action_c2) / 2

            # Combine agent-selected state with current state
            weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h

            if self.cell == "lstm":
                weighted_c = self.lamda * action_c + (1 - self.lamda) * cur_c
                cur_h, cur_c = self.rnn(cur_input, (weighted_h, weighted_c))
            else:
                cur_h = self.rnn(cur_input, weighted_h)

            h_list.append(cur_h)

        # Stack all hidden states
        all_outputs = torch.stack(h_list, dim=1)

        # Fuse with static features if available
        if self.static_dim > 0 and static is not None:
            static_expanded = static.unsqueeze(1).expand(-1, time_step, -1)
            all_outputs = torch.cat((all_outputs, static_expanded), dim=2)
            all_outputs = self.fusion(all_outputs)

        # Get last valid output
        last_output = get_last_visit(all_outputs, mask)

        if self.dropout > 0.0:
            last_output = self.nn_dropout(last_output)

        return last_output, all_outputs


class Agent(BaseModel):
    """Dr. Agent model for clinical prediction tasks.

    This model uses two reinforcement learning agents with dynamic skip
    connections to capture long-term dependencies in patient EHR sequences.
    The primary agent focuses on current health status while the second-opinion
    agent considers historical context.

    Paper: Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked
        second opinions. JAMIA 2020.

    Args:
        dataset: SampleDataset with fitted input/output processors.
        embedding_dim: Embedding dimension for input features. Default is 128.
        hidden_dim: Hidden dimension for RNN and output. Default is 128.
        static_key: Key for static features (e.g., demographics). These are
            passed directly to AgentLayer, not through EmbeddingModel.
            Default is None.
        use_baseline: Whether to use baseline for RL variance reduction.
            Default is True.
        **kwargs: Additional arguments passed to AgentLayer (e.g., n_actions,
            n_units, dropout, lamda, cell).

    Example:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.models import Agent
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": [["A01", "A02"], ["B01"]],
        ...         "procedures": [["P1"], ["P2", "P3"]],
        ...         "demographic": [65.0, 1.0, 25.5],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "conditions": [["C01"]],
        ...         "procedures": [["P4"]],
        ...         "demographic": [45.0, 0.0, 22.1],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = Agent(dataset, static_key="demographic")
        >>> # Forward pass with a batch
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> output["loss"].backward()
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        static_key: Optional[str] = None,
        use_baseline: bool = True,
        **kwargs,
    ):
        super(Agent, self).__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.static_key = static_key
        self.use_baseline = use_baseline

        # Validate kwargs
        if "input_dim" in kwargs:
            raise ValueError("input_dim is determined by embedding_dim")
        if "n_hidden" in kwargs:
            raise ValueError("n_hidden is determined by hidden_dim")

        # Single label key required
        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        # Determine static dimension
        self.static_dim = 0
        if self.static_key is not None:
            for sample in self.dataset.samples:
                if self.static_key in sample:
                    self.static_dim = len(sample[self.static_key])
                    break

        # Sequence feature keys (exclude static)
        self.seq_feature_keys = [
            k for k in self.feature_keys if k != self.static_key
        ]

        # Embedding model for sequence features
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Agent layer for each sequence feature
        self.agent = nn.ModuleDict()
        for feature_key in self.seq_feature_keys:
            self.agent[feature_key] = AgentLayer(
                input_dim=embedding_dim,
                static_dim=self.static_dim,
                n_hidden=hidden_dim,
                use_baseline=use_baseline,
                **kwargs,
            )

        # Output layer
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.seq_feature_keys) * hidden_dim, output_size)

    def _compute_rl_loss(
        self,
        agent_layer: AgentLayer,
        pred: torch.Tensor,
        true: torch.Tensor,
        mask: torch.Tensor,
        gamma: float = 0.9,
        entropy_term: float = 0.01,
    ) -> torch.Tensor:
        """Compute REINFORCE loss for agent optimization.

        Args:
            agent_layer: AgentLayer instance with stored action probabilities.
            pred: Predicted logits of shape [batch, output_size].
            true: Ground truth labels.
            mask: Valid timestep mask of shape [batch, seq_len].
            gamma: Discount factor for long-term rewards. Default is 0.9.
            entropy_term: Entropy bonus coefficient. Default is 0.01.

        Returns:
            Combined RL loss (policy loss + value loss if using baseline).
        """
        # Compute rewards based on prediction accuracy
        if self.mode == "binary":
            pred_prob = torch.sigmoid(pred)
            rewards = ((pred_prob - 0.5) * 2 * true).squeeze()
        elif self.mode == "multiclass":
            pred_prob = torch.softmax(pred, dim=-1)
            y_onehot = torch.zeros_like(pred_prob).scatter(1, true.unsqueeze(1), 1)
            rewards = (pred_prob * y_onehot).sum(-1).squeeze()
        elif self.mode == "multilabel":
            pred_prob = torch.sigmoid(pred)
            rewards = (
                ((pred_prob - 0.5) * 2 * true).sum(dim=-1)
                / (true.sum(dim=-1) + 1e-7)
            ).squeeze()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Stack agent log probabilities and entropy
        act_prob1 = torch.stack(agent_layer.agent1_prob).permute(1, 0)
        act_prob1 = act_prob1.to(self.device) * mask.float()
        act_entropy1 = torch.stack(agent_layer.agent1_entropy).permute(1, 0)
        act_entropy1 = act_entropy1.to(self.device) * mask.float()

        act_prob2 = torch.stack(agent_layer.agent2_prob).permute(1, 0)
        act_prob2 = act_prob2.to(self.device) * mask.float()
        act_entropy2 = torch.stack(agent_layer.agent2_entropy).permute(1, 0)
        act_entropy2 = act_entropy2.to(self.device) * mask.float()

        if self.use_baseline:
            act_baseline1 = (
                torch.stack(agent_layer.agent1_baseline)
                .squeeze(-1)
                .permute(1, 0)
                .to(self.device)
            )
            act_baseline1 = act_baseline1 * mask.float()
            act_baseline2 = (
                torch.stack(agent_layer.agent2_baseline)
                .squeeze(-1)
                .permute(1, 0)
                .to(self.device)
            )
            act_baseline2 = act_baseline2 * mask.float()

        # Compute discounted cumulative rewards
        seq_len = act_prob1.size(1)
        running_rewards = []
        discounted_reward = torch.zeros_like(rewards)
        for i in reversed(range(seq_len)):
            if i == seq_len - 1:
                discounted_reward = rewards + gamma * discounted_reward
            else:
                discounted_reward = gamma * discounted_reward
            running_rewards.insert(0, discounted_reward)
        rewards_tensor = torch.stack(running_rewards).permute(1, 0).detach()

        # Compute losses
        mask_sum = torch.sum(mask.float(), dim=1).clamp(min=1.0)

        if self.use_baseline:
            # Value function loss
            loss_value1 = torch.mean(
                torch.sum((rewards_tensor - act_baseline1) ** 2, dim=1) / mask_sum
            )
            loss_value2 = torch.mean(
                torch.sum((rewards_tensor - act_baseline2) ** 2, dim=1) / mask_sum
            )

            # Policy gradient loss with baseline
            advantage1 = rewards_tensor - act_baseline1
            advantage2 = rewards_tensor - act_baseline2
            loss_rl1 = torch.mean(
                -torch.sum(
                    act_prob1 * advantage1 + entropy_term * act_entropy1, dim=1
                )
                / mask_sum
            )
            loss_rl2 = torch.mean(
                -torch.sum(
                    act_prob2 * advantage2 + entropy_term * act_entropy2, dim=1
                )
                / mask_sum
            )
            return loss_rl1 + loss_rl2 + loss_value1 + loss_value2
        else:
            # Policy gradient loss without baseline
            loss_rl1 = torch.mean(
                -torch.sum(
                    act_prob1 * rewards_tensor + entropy_term * act_entropy1, dim=1
                )
                / mask_sum
            )
            loss_rl2 = torch.mean(
                -torch.sum(
                    act_prob2 * rewards_tensor + entropy_term * act_entropy2, dim=1
                )
                / mask_sum
            )
            return loss_rl1 + loss_rl2

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Keyword arguments containing input features and labels.
                Must include all keys from input_schema and output_schema.

        Returns:
            Dictionary containing:
                - loss: Combined task loss and RL loss.
                - y_prob: Predicted probabilities.
                - y_true: Ground truth labels.
                - logit: Raw logits.
                - embed (optional): Patient embeddings if embed=True in kwargs.
        """
        patient_emb = []
        mask_dict = {}

        # Get static features
        static = None
        if self.static_key is not None and self.static_key in kwargs:
            static_data = kwargs[self.static_key]
            if isinstance(static_data, torch.Tensor):
                static = static_data.float().to(self.device)
            else:
                static = torch.tensor(
                    static_data, dtype=torch.float, device=self.device
                )

        # Get embeddings for sequence features
        embedded = self.embedding_model(kwargs)

        # Process each sequence feature through its agent
        for feature_key in self.seq_feature_keys:
            x = embedded[feature_key]
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float, device=self.device)
            x = x.float().to(self.device)

            # Compute mask from embeddings (non-zero entries are valid)
            mask = x.sum(dim=-1) != 0
            mask_dict[feature_key] = mask

            # Forward through agent layer
            out, _ = self.agent[feature_key](x, static=static, mask=mask)
            patient_emb.append(out)

        # Concatenate embeddings and predict
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        # Compute task loss
        y_true = kwargs[self.label_key]
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, device=self.device)
        y_true = y_true.to(self.device)
        loss_task = self.get_loss_function()(logits, y_true)

        # Compute RL loss for each agent
        loss_rl = torch.tensor(0.0, device=self.device)
        for feature_key in self.seq_feature_keys:
            loss_rl = loss_rl + self._compute_rl_loss(
                self.agent[feature_key],
                logits,
                y_true,
                mask_dict[feature_key],
            )

        loss = loss_task + loss_rl
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    # Example usage with synthetic data
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
            "procedures": [["P1", "P2"], ["P3"]],
            "demographic": [1.0, 2.0, 1.3],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "conditions": [["A04A", "B035", "C129"]],
            "procedures": [["P4", "P5"]],
            "demographic": [1.0, 2.0, 1.3],
            "label": 0,
        },
    ]

    dataset = SampleDataset(
        samples=samples,
        input_schema={"conditions": "sequence", "procedures": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = Agent(
        dataset=dataset,
        static_key="demographic",
        embedding_dim=128,
        hidden_dim=128,
    )

    data_batch = next(iter(train_loader))
    ret = model(**data_batch)
    print(ret)
    ret["loss"].backward()