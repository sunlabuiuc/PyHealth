import torch
import torch.nn.functional as F

from pyhealth.models import Transformer

from .base_interpreter import BaseInterpreter


def apply_self_attention_rules(R_ss, cam_ss):
    """Apply Chefer's self-attention rules for relevance propagation.

    This function propagates relevance scores through an attention layer by
    multiplying the current relevance matrix with the attention weights.

    Args:
        R_ss (torch.Tensor): Relevance matrix of shape ``[batch, seq_len, seq_len]``
            representing token-to-token relevance from previous layers.
        cam_ss (torch.Tensor): Attention weight matrix of shape ``[batch, seq_len, seq_len]``
            representing the current layer's attention scores.

    Returns:
        torch.Tensor: Updated relevance matrix of shape ``[batch, seq_len, seq_len]``
            after propagating through the attention layer.
    """
    return torch.matmul(cam_ss, R_ss)


def avg_heads(cam, grad):
    """Average attention scores weighted by their gradients across multiple heads.

    This function computes gradient-weighted attention scores and averages them
    across attention heads. The gradients indicate how much each attention weight
    contributed to the final prediction, providing a measure of importance.

    Args:
        cam (torch.Tensor): Attention weights. Shape ``[batch, heads, seq_len, seq_len]``
            for multi-head attention or ``[batch, seq_len, seq_len]`` for single-head.
        grad (torch.Tensor): Gradients of the loss with respect to attention weights.
            Same shape as ``cam``.

    Returns:
        torch.Tensor: Gradient-weighted attention scores, averaged across heads.
            Shape ``[batch, seq_len, seq_len]``. Negative values are clamped to zero.

    Note:
        If input tensors have fewer than 4 dimensions (single-head case), no
        averaging is performed and the element-wise product is returned directly.
    """
    # force shapes of cam and grad to be the same order
    if (
        len(cam.size()) < 4 and len(grad.size()) < 4
    ):  # check if no averaging needed. i.e single head
        return (grad * cam).clamp(min=0)
    cam = grad * cam  # elementwise mult
    cam = cam.clamp(min=0).mean(dim=1)  # average across heads
    return cam.clone()


class CheferRelevance(BaseInterpreter):
    """Transformer Self Attention Token Relevance Computation using Chefer's Method.

    This class computes the relevance of each token in the input sequence for a given
    class prediction. The relevance is computed using Chefer's Self Attention Rules,
    which provide interpretability for transformer models by propagating relevance
    scores through attention layers.

    The method is based on the paper:
        Generic Attention-model Explainability for Interpreting Bi-Modal and
        Encoder-Decoder Transformers
        Hila Chefer, Shir Gur, Lior Wolf
        https://arxiv.org/abs/2103.15679
        Implementation based on https://github.com/hila-chefer/Transformer-Explainability

    Args:
        model (Transformer): A trained PyHealth Transformer model to interpret.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import SampleDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import CheferRelevance
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Define sample data
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["D001", "D002", "D003"],
        ...         "procedures": ["P001", "P002"],
        ...         "drugs": ["M001", "M002"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["D004", "D005"],
        ...         "procedures": ["P003"],
        ...         "drugs": ["M003"],
        ...         "label": 0,
        ...     },
        ...     # ... more samples
        ... ]
        >>>
        >>> # Create dataset with schema
        >>> input_schema = {
        ...     "conditions": "sequence",
        ...     "procedures": "sequence",
        ...     "drugs": "sequence"
        ... }
        >>> output_schema = {"label": "binary"}
        >>>
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema=input_schema,
        ...     output_schema=output_schema,
        ...     dataset_name="example"
        ... )
        >>>
        >>> # Initialize Transformer model
        >>> model = Transformer(
        ...     dataset=dataset,
        ...     embedding_dim=128,
        ...     heads=2,
        ...     dropout=0.3,
        ...     num_layers=2
        ... )
        >>>
        >>> # Split data and create dataloaders
        >>> train_data, val_data, test_data = split_by_patient(dataset, [0.7, 0.15, 0.15])
        >>> train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
        >>> val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
        >>> test_loader = get_dataloader(test_data, batch_size=1, shuffle=False)
        >>>
        >>> # Train model
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     epochs=10,
        ...     monitor="roc_auc"
        ... )
        >>>
        >>> # Compute relevance scores for test samples
        >>> relevance = CheferRelevance(model)
        >>> data_batch = next(iter(test_loader))
        >>>
        >>> # Option 1: Specify target class explicitly
        >>> data_batch['class_index'] = 0
        >>> scores = relevance.get_relevance_matrix(**data_batch)
        >>> print(scores)
        {'conditions': tensor([[1.2210]], device='cuda:0'),
         'procedures': tensor([[1.0865]], device='cuda:0'),
         'drugs': tensor([[1.0000]], device='cuda:0')}
        >>>
        >>> # Option 2: Use predicted class (omit class_index)
        >>> scores = relevance.get_relevance_matrix(
        ...     conditions=data_batch['conditions'],
        ...     procedures=data_batch['procedures'],
        ...     drugs=data_batch['drugs'],
        ...     label=data_batch['label']
        ... )
    """

    def __init__(self, model: Transformer):
        """Initialize Chefer relevance interpreter.

        Args:
            model: A trained PyHealth Transformer model to interpret.
                Must be an instance of pyhealth.models.Transformer.

        Raises:
            AssertionError: If model is not a Transformer instance.
        """
        super().__init__(model)
        assert isinstance(model, Transformer), (
            f"CheferRelevance only works with Transformer models, "
            f"got {type(model).__name__}"
        )

    def attribute(self, **data):
        """Compute relevance scores for each token in the input features.

        This method performs a forward pass through the model and computes
        gradient-based relevance scores for each input token across all feature
        modalities (e.g., conditions, procedures, drugs). The relevance scores
        indicate the importance of each token for the predicted class. Higher
        relevance scores suggest that the token contributed more to the model's
        prediction.

        Args:
            **data: Input data dictionary from a dataloader batch containing:
                - Feature keys (e.g., 'conditions', 'procedures', 'drugs'):
                  Input tensors or sequences for each modality
                - 'label': Ground truth label tensor
                - 'class_index' (optional): Integer specifying target class for
                  relevance computation. If not provided, uses the predicted
                  class (argmax of model output).

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping each feature key to its
                relevance score tensor. Each tensor has shape ``[batch_size,
                num_tokens]`` where higher values indicate greater relevance for
                the prediction. Scores are non-negative due to the clamping
                operation in the relevance propagation algorithm.

        Note:
            - This method requires gradients, so it should not be called within
              a ``torch.no_grad()`` context.
            - The method modifies model state temporarily (registers hooks) but
              restores it after computation.
            - For batch processing, it's recommended to use batch_size=1 to get
              per-sample interpretability.

        Examples:
            >>> from pyhealth.interpret.methods import CheferRelevance
            >>>
            >>> # Assuming you have a trained transformer model and test data
            >>> relevance = CheferRelevance(trained_model)
            >>> test_batch = next(iter(test_loader))
            >>>
            >>> # Compute relevance for predicted class
            >>> scores = relevance.attribute(**test_batch)
            >>> print(f"Feature relevance: {scores.keys()}")
            >>> print(f"Condition relevance shape: {scores['conditions'].shape}")
            >>>
            >>> # Compute relevance for specific class (e.g., positive class)
            >>> test_batch['class_index'] = 1
            >>> scores_positive = relevance.attribute(**test_batch)
            >>>
            >>> # Analyze which tokens are most relevant
            >>> condition_scores = scores['conditions'][0]  # First sample
            >>> top_k_indices = torch.topk(condition_scores, k=5).indices
            >>> print(f"Most relevant condition tokens: {top_k_indices}")
        """
        return self.get_relevance_matrix(**data)

    def get_relevance_matrix(self, **data):
        """Compute relevance scores for each token in the input features.

        This method performs a forward pass through the model and computes gradient-based
        relevance scores for each input token across all feature modalities (e.g.,
        conditions, procedures, drugs). The relevance scores indicate the importance
        of each token for the predicted class. Higher relevance scores suggest that
        the token contributed more to the model's prediction.

        Args:
            **data: Input data dictionary from a dataloader batch containing:
                - Feature keys (e.g., 'conditions', 'procedures', 'drugs'):
                  Input tensors or sequences for each modality
                - 'label': Ground truth label tensor
                - 'class_index' (optional): Integer specifying target class for
                  relevance computation. If not provided, uses the predicted class
                  (argmax of model output).

        Returns:
            dict: Dictionary mapping each feature key to its relevance score tensor.
                Each tensor has shape ``[batch_size, num_tokens]`` where higher values
                indicate greater relevance for the prediction. Scores are non-negative
                due to the clamping operation in the relevance propagation algorithm.

        Note:
            - This method requires gradients, so it should not be called within a
              ``torch.no_grad()`` context.
            - The method modifies model state temporarily (registers hooks) but
              restores it after computation.
            - For batch processing, it's recommended to use batch_size=1 to get
              per-sample interpretability.

        Examples:
            >>> from pyhealth.interpret.methods import CheferRelevance
            >>>
            >>> # Assuming you have a trained transformer model and test data
            >>> relevance = CheferRelevance(trained_model)
            >>> test_batch = next(iter(test_loader))
            >>>
            >>> # Compute relevance for predicted class
            >>> scores = relevance.get_relevance_matrix(**test_batch)
            >>> print(f"Feature relevance: {scores.keys()}")
            >>> print(f"Condition relevance shape: {scores['conditions'].shape}")
            >>>
            >>> # Compute relevance for specific class (e.g., positive class in binary)
            >>> test_batch['class_index'] = 1
            >>> scores_positive = relevance.get_relevance_matrix(**test_batch)
            >>>
            >>> # Analyze which tokens are most relevant
            >>> condition_scores = scores['conditions'][0]  # First sample
            >>> top_k_indices = torch.topk(condition_scores, k=5).indices
            >>> print(f"Most relevant condition tokens: {top_k_indices}")
        """
        input = data
        input["register_hook"] = True
        index = data.get("class_index")

        logits = self.model(**input)["logit"]
        if index == None:
            index = torch.argmax(logits, dim=-1)

        # create one_hot matrix of n x c, one_hot vecs, for graph computation
        one_hot = F.one_hot(torch.tensor(index), logits.size()[1]).float()
        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(logits.device) * logits)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        feature_keys = self.model.feature_keys

        # get how many tokens we see per modality
        num_tokens = {}
        for key in feature_keys:
            feature_transformer = self.model.transformer[key].transformer
            for block in feature_transformer:
                num_tokens[key] = block.attention.get_attn_map().shape[-1]

        attn = {}
        for key in feature_keys:
            R = (
                torch.eye(num_tokens[key])
                .unsqueeze(0)
                .repeat(len(input[key]), 1, 1)
                .to(logits.device)
            )  # initialize identity matrix, but batched
            for blk in self.model.transformer[key].transformer:
                grad = blk.attention.get_attn_grad()
                cam = blk.attention.get_attn_map()
                cam = avg_heads(cam, grad)
                R += apply_self_attention_rules(R, cam).detach()

            attn[key] = R[:, 0]  # get CLS Token

        # return Rs for each feature_key
        return attn  # Assume CLS token is first row of attention score matrix
