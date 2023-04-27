from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
import torch.nn.functional as F


class AgentLayer(nn.Module):
    """Dr. Agent layer.

    Paper: Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked second opinions. JAMIA.

    This layer is used in the Dr. Agent model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: dynamic feature size.
        static_dim: static feature size, if 0, then no static feature is used.
        cell: rnn cell type. Default is "gru".
        use_baseline: whether to use baseline for the RL agent. Default is True.
        n_actions: number of historical visits to choose. Default is 10.
        n_units: number of hidden units in each agent. Default is 64.
        fusion_dim: number of hidden units in the final representation. Default is 128.
        n_hidden: number of hidden units in the rnn. Default is 128.
        dropout: dropout rate. Default is 0.5.
        lamda: weight for the agent selected hidden state and the current hidden state. Default is 0.5.

    Examples:
        >>> from pyhealth.models import AgentLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = AgentLayer(64)
        >>> c, _ = layer(input)
        >>> c.shape
        torch.Size([3, 128])
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
        dropout: int = 0.5,
        lamda: int = 0.5,
    ):
        super(AgentLayer, self).__init__()

        if cell not in ["gru", "lstm"]:
            raise ValueError("Only gru and lstm are supported for cell.")

        self.cell = cell
        self.use_baseline = use_baseline
        self.n_actions = n_actions
        self.n_units = n_units
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        # self.n_output = n_output
        self.dropout = dropout
        self.lamda = lamda
        self.fusion_dim = n_hidden
        self.static_dim = static_dim

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        self.agent1_fc1 = nn.Linear(self.n_hidden + self.static_dim, self.n_units)
        self.agent2_fc1 = nn.Linear(self.input_dim + self.static_dim, self.n_units)
        self.agent1_fc2 = nn.Linear(self.n_units, self.n_actions)
        self.agent2_fc2 = nn.Linear(self.n_units, self.n_actions)
        if use_baseline == True:
            self.agent1_value = nn.Linear(self.n_units, 1)
            self.agent2_value = nn.Linear(self.n_units, 1)

        if self.cell == "lstm":
            self.rnn = nn.LSTMCell(self.input_dim, self.n_hidden)
        else:
            self.rnn = nn.GRUCell(self.input_dim, self.n_hidden)

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        if dropout > 0.0:
            self.nn_dropout = nn.Dropout(p=dropout)
        if self.static_dim > 0:
            self.init_h = nn.Linear(self.static_dim, self.n_hidden)
            self.init_c = nn.Linear(self.static_dim, self.n_hidden)
            self.fusion = nn.Linear(self.n_hidden + self.static_dim, self.fusion_dim)
        # self.output = nn.Linear(self.fusion_dim, self.n_output)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def choose_action(self, observation, agent=1):
        observation = observation.detach()

        if agent == 1:
            result_fc1 = self.agent1_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent1_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent1_value(result_fc1)
                self.agent1_baseline.append(result_value)
        else:
            result_fc1 = self.agent2_fc1(observation)
            result_fc1 = self.tanh(result_fc1)
            result_fc2 = self.agent2_fc2(result_fc1)
            if self.use_baseline == True:
                result_value = self.agent2_value(result_fc1)
                self.agent2_baseline.append(result_value)

        probs = self.softmax(result_fc2)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()

        if agent == 1:
            self.agent1_entropy.append(m.entropy())
            self.agent1_action.append(actions.unsqueeze(-1))
            self.agent1_prob.append(m.log_prob(actions))
        else:
            self.agent2_entropy.append(m.entropy())
            self.agent2_action.append(actions.unsqueeze(-1))
            self.agent2_prob.append(m.log_prob(actions))

        return actions.unsqueeze(-1)

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            last_output: a tensor of shape [batch size, n_hidden] representing the
                patient embedding.
            output: a tensor of shape [batch size, sequence len, n_hidden] representing the patient embedding at each time step.
        """
        # rnn will only apply dropout between layers

        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        self.agent1_action = []
        self.agent1_prob = []
        self.agent1_entropy = []
        self.agent1_baseline = []
        self.agent2_action = []
        self.agent2_prob = []
        self.agent2_entropy = []
        self.agent2_baseline = []

        if self.static_dim > 0:
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

        h = []
        for cur_time in range(time_step):
            cur_input = x[:, cur_time, :]

            if cur_time == 0:
                obs_1 = cur_h
                obs_2 = cur_input

                if self.static_dim > 0:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                self.choose_action(obs_1, 1).long()
                self.choose_action(obs_2, 2).long()

                observed_h = (
                    torch.zeros_like(cur_h, dtype=torch.float32)
                    .view(-1)
                    .repeat(self.n_actions)
                    .view(self.n_actions, batch_size, self.n_hidden)
                )
                action_h = cur_h
                if self.cell == "lstm":
                    observed_c = (
                        torch.zeros_like(cur_c, dtype=torch.float32)
                        .view(-1)
                        .repeat(self.n_actions)
                        .view(self.n_actions, batch_size, self.n_hidden)
                    )
                    action_c = cur_c

            else:
                observed_h = torch.cat((observed_h[1:], cur_h.unsqueeze(0)), 0)

                obs_1 = observed_h.mean(dim=0)
                obs_2 = cur_input

                if self.static_dim > 0:
                    obs_1 = torch.cat((obs_1, static), dim=1)
                    obs_2 = torch.cat((obs_2, static), dim=1)

                act_idx1 = self.choose_action(obs_1, 1).long()
                act_idx2 = self.choose_action(obs_2, 2).long()
                batch_idx = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)
                action_h1 = observed_h[act_idx1, batch_idx, :].squeeze(1)
                action_h2 = observed_h[act_idx2, batch_idx, :].squeeze(1)
                action_h = (action_h1 + action_h2) / 2
                if self.cell == "lstm":
                    observed_c = torch.cat((observed_c[1:], cur_c.unsqueeze(0)), 0)
                    action_c1 = observed_c[act_idx1, batch_idx, :].squeeze(1)
                    action_c2 = observed_c[act_idx2, batch_idx, :].squeeze(1)
                    action_c = (action_c1 + action_c2) / 2

            if self.cell == "lstm":
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                weighted_c = self.lamda * action_c + (1 - self.lamda) * cur_c
                rnn_state = (weighted_h, weighted_c)
                cur_h, cur_c = self.rnn(cur_input, rnn_state)
            else:
                weighted_h = self.lamda * action_h + (1 - self.lamda) * cur_h
                cur_h = self.rnn(cur_input, weighted_h)
            h.append(cur_h)

        h = torch.stack(h, dim=1)

        if self.static_dim > 0:
            static = static.unsqueeze(1).repeat(1, time_step, 1)
            h = torch.cat((h, static), dim=2)
            h = self.fusion(h)

        last_out = get_last_visit(h, mask)

        if self.dropout > 0.0:
            last_out = self.nn_dropout(last_out)
        return last_out, h


class Agent(BaseModel):
    """Dr. Agent model.

    Paper: Junyi Gao et al. Dr. Agent: Clinical predictive model via mimicked second opinions. JAMIA.

    Note:
        We use separate Dr. Agent layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        We follow the current convention for the Dr. Agent model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply Dr. Agent on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                Dr. Agent one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Dr. Agent directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run Dr. Agent directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        static_keys: the key in samples to use as static features, e.g. "demographics". Default is None.
                     we only support numerical static features.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension of the RNN in the Dr. Agent layer. Default is 128.
        use_baseline: whether to use the baseline value to calculate the RL loss. Default is True.
        **kwargs: other parameters for the Dr. Agent layer.


    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import Agent
        >>> model = Agent(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         static_key="demographic",
        ...         mode="binary"
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(1.4059, grad_fn=<AddBackward0>),
            'y_prob': tensor([[0.4861], [0.5348]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[0.], [1.]]),
            'logit': tensor([[-0.0556], [0.1392]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        use_baseline: bool = True,
        **kwargs,
    ):
        super(Agent, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # validate kwargs for Dr. Agent layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.static_key = static_key
        self.use_baseline = use_baseline
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        self.static_dim = 0
        if self.static_key is not None:
            self.static_dim = self.dataset.input_info[self.static_key]["len"]

        self.agent = nn.ModuleDict()
        # add feature Dr. Agent layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "Dr. Agent only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "Dr. Agent only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "Dr. Agent only supports 2-dim or 3-dim float and int as input types"
                )

            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)
            self.agent[feature_key] = AgentLayer(
                input_dim=embedding_dim,
                static_dim=self.static_dim,
                n_hidden=hidden_dim,
                **kwargs,
            )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def get_loss(self, model, pred, true, mask, gamma=0.9, entropy_term=0.01):

        if self.mode == "binary":
            pred = torch.sigmoid(pred)
            rewards = ((pred - 0.5) * 2 * true).squeeze()
        elif self.mode == "multiclass":
            pred = torch.softmax(pred, dim=-1)
            y_onehot = torch.zeros_like(pred).scatter(1, true.unsqueeze(1), 1)
            rewards = (pred * y_onehot).sum(-1).squeeze()
        elif self.mode == "multilabel":
            pred = torch.sigmoid(pred)
            rewards = (
                ((pred - 0.5) * 2 * true).sum(dim=-1) / (true.sum(dim=-1) + 1e-7)
            ).squeeze()
        elif self.mode == "regression":
            rewards = (1 / torch.abs(pred - true)).squeeze()  # b*t
            rewards = torch.clamp(rewards, min=0, max=5)
        else:
            raise ValueError(
                "mode should be binary, multiclass, multilabel or regression"
            )

        act_prob1 = model.agent1_prob
        act_prob1 = torch.stack(act_prob1).permute(1, 0).to(self.device)
        act_prob1 = act_prob1 * mask.view(act_prob1.size(0), act_prob1.size(1))
        act_entropy1 = model.agent1_entropy
        act_entropy1 = torch.stack(act_entropy1).permute(1, 0).to(self.device)
        act_entropy1 = act_entropy1 * mask.view(
            act_entropy1.size(0), act_entropy1.size(1)
        )
        if self.use_baseline == True:
            act_baseline1 = model.agent1_baseline
            act_baseline1 = (
                torch.stack(act_baseline1).squeeze(-1).permute(1, 0).to(self.device)
            )
            act_baseline1 = act_baseline1 * mask.view(
                act_baseline1.size(0), act_baseline1.size(1)
            )

        act_prob2 = model.agent2_prob
        act_prob2 = torch.stack(act_prob2).permute(1, 0).to(self.device)
        act_prob2 = act_prob2 * mask.view(act_prob2.size(0), act_prob2.size(1))
        act_entropy2 = model.agent2_entropy
        act_entropy2 = torch.stack(act_entropy2).permute(1, 0).to(self.device)
        act_entropy2 = act_entropy2 * mask.view(
            act_entropy2.size(0), act_entropy2.size(1)
        )
        if self.use_baseline == True:
            act_baseline2 = model.agent2_baseline
            act_baseline2 = (
                torch.stack(act_baseline2).squeeze(-1).permute(1, 0).to(self.device)
            )
            act_baseline2 = act_baseline2 * mask.view(
                act_baseline2.size(0), act_baseline2.size(1)
            )

        running_rewards = []
        discounted_rewards = 0
        for i in reversed(range(act_prob1.size(1))):
            if i == act_prob1.size(1) - 1:
                discounted_rewards = rewards + gamma * discounted_rewards
            else:
                discounted_rewards = (
                    torch.zeros_like(rewards) + gamma * discounted_rewards
                )
            running_rewards.insert(0, discounted_rewards)
        rewards = torch.stack(running_rewards).permute(1, 0)
        # rewards = (rewards - rewards.mean(dim=1).unsqueeze(-1)) / (
        #     rewards.std(dim=1) + 1e-7
        # ).unsqueeze(-1)
        rewards = rewards.detach()

        if self.use_baseline == True:
            loss_value1 = torch.sum((rewards - act_baseline1) ** 2, dim=1) / torch.sum(
                mask, dim=1
            )
            loss_value1 = torch.mean(loss_value1)
            loss_value2 = torch.sum((rewards - act_baseline2) ** 2, dim=1) / torch.sum(
                mask, dim=1
            )
            loss_value2 = torch.mean(loss_value2)
            loss_value = loss_value1 + loss_value2
            loss_RL1 = -torch.sum(
                act_prob1 * (rewards - act_baseline1) + entropy_term * act_entropy1,
                dim=1,
            ) / torch.sum(mask, dim=1)
            loss_RL1 = torch.mean(loss_RL1)
            loss_RL2 = -torch.sum(
                act_prob2 * (rewards - act_baseline2) + entropy_term * act_entropy2,
                dim=1,
            ) / torch.sum(mask, dim=1)
            loss_RL2 = torch.mean(loss_RL2)
            loss_RL = loss_RL1 + loss_RL2
            loss = loss_RL + loss_value
        else:
            loss_RL1 = -torch.sum(
                act_prob1 * rewards + entropy_term * act_entropy1, dim=1
            ) / torch.sum(mask, dim=1)
            loss_RL1 = torch.mean(loss_RL1)
            loss_RL2 = -torch.sum(
                act_prob2 * rewards + entropy_term * act_entropy2, dim=1
            ) / torch.sum(mask, dim=1)
            loss_RL2 = torch.mean(loss_RL2)
            loss_RL = loss_RL1 + loss_RL2
            loss = loss_RL

        return loss

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                loss_task: a scalar tensor representing the task loss.
                loss_RL: a scalar tensor representing the RL loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        mask_dict = {}
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x !=0, dim=2)
                mask_dict[feature_key] = mask

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x !=0, dim=2)
                mask_dict[feature_key] = mask

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)
                mask_dict[feature_key] = mask

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
                mask_dict[feature_key] = mask

            else:
                raise NotImplementedError

            if self.static_dim > 0:
                static = torch.tensor(
                    kwargs[self.static_key], dtype=torch.float, device=self.device
                )
                x, _ = self.agent[feature_key](x, static=static, mask=mask)
            else:
                x, _ = self.agent[feature_key](x, mask=mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss_task = self.get_loss_function()(logits, y_true)

        loss_rl = 0
        for feature_key in self.feature_keys:
            cur_loss = self.get_loss(
                self.agent[feature_key], logits, y_true, mask_dict[feature_key]
            )
            loss_rl += cur_loss
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
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            # "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
            "demographic": [1.0, 2.0, 1.3],
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            # "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
            "demographic": [1.0, 2.0, 1.3],
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = Agent(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            # "list_list_vectors",
        ],
        static_key="demographic",
        label_key="label",
        mode="binary",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
