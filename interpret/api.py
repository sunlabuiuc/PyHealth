from abc import ABC, abstractmethod
import torch
from torch import nn

class Interpretable(ABC):
    """Abstract interface for models supporting interpretability methods.
    
    This class defines the contract that models must fulfill to be compatible
    with PyHealth's interpretability module. It enables gradient-based 
    attribution methods and embedding-level perturbation methods to work with
    your model.
    
    The interface separates the embedding stage (which generates learned 
    representations from raw features) from the prediction stage 
    (which generates outputs from embeddings). This separation allows 
    interpretability methods to either:
    
    1. Use gradients flowing through embeddings
    2. Perturb embeddings and pass them through prediction head
    3. Directly access and analyze the learned representations
    
    Methods
    -------
    forward
        Standard forward pass of the model.
    forward_from_embedding
        Perform forward pass starting from embeddings.
    get_embedding_model
        Get the embedding/feature extraction stage if applicable.
    
    Assumptions
    -----------
    Models implementing this interface must adhere to the following assumptions:
    
    1. **Optional label handling**: The ``forward_from_embedding()`` method must 
       accept label keys (as specified in ``self.label_keys``) as optional keyword 
       arguments. The method should handle cases where labels are missing without 
       raising exceptions. When labels are absent, the method should skip loss 
       computation and omit 'loss' and 'y_true' from the return dictionary.
    
    2. **Non-linearity as nn.Module**: All non-linear activation functions 
       (ReLU, Sigmoid, Softmax, Tanh, etc.) must be defined as nn.Module instances 
       in the model's ``__init__`` method and called as instance methods 
       (e.g., ``self.relu(x)``). Do NOT use functional variants like ``F.relu(x)``, 
       ``F.sigmoid(x)``, or ``F.softmax(x)``. This is critical for 
       gradient-based interpretability methods (e.g., DeepLIFT) that require 
       hooks to be registered on non-linearities.
    
    Examples of correct activation usage::
    
        class GoodModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()    # Correct
                self.sigmoid = nn.Sigmoid()  # Correct
                
            def forward(self, x):
                x = self.relu(x)         # Correct
                x = self.sigmoid(x)      # Correct
                return x
    
    Examples of incorrect activation usage::
    
        class BadModel(nn.Module):
            def forward(self, x):
                x = F.relu(x)            # WRONG - functional variant
                x = F.sigmoid(x)         # WRONG - functional variant
                return x
    """

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        This is the standard entry point for running the model on a batch
        of data.  It accepts the raw feature tensors (as produced by the
        dataloader) and returns predictions.

        Parameters
        ----------
        **kwargs : torch.Tensor or tuple[torch.Tensor, ...]
            Keyword arguments keyed by the model's ``feature_keys`` and
            ``label_keys``.  Each value is either a single tensor or a
            tuple of tensors (e.g. ``(value, mask)``).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing at least:

            - **logit** (torch.Tensor): Raw model predictions / logits of
              shape ``(batch_size, num_classes)``.
            - **y_prob** (torch.Tensor): Predicted probabilities.
            - **loss** (torch.Tensor, optional): Scalar loss, present only
              when label keys are included in ``kwargs``.
            - **y_true** (torch.Tensor, optional): Ground-truth labels,
              present only when label keys are included in ``kwargs``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError

    def forward_from_embedding(
        self, 
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model starting from embeddings.
        
        This method enables interpretability methods to pass embeddings directly
        into the model's prediction head, bypassing the embedding stage. This is
        useful for:
        
        - **Gradient-based attribution** (DeepLIFT, Integrated Gradients): 
          Allows gradients to be computed with respect to embeddings
        - **Embedding perturbation** (LIME, SHAP): Allows perturbing embeddings 
          instead of raw features
        - **Intermediate representation analysis**: Enables inspection of learned 
          representations at the embedding layer
        
        Kwargs keys typically mirror the model's feature keys (from the dataset's
        input_schema), but represent embeddings instead of raw features.
        
        Parameters
        ----------
        **kwargs : torch.Tensor or tuple[torch.Tensor, ...]
            Variable keyword arguments representing input embeddings and optional labels.
            
            **Embedding arguments** (required): Should include all feature keys that 
            your model expects. Examples:
            
            - 'conditions': (batch_size, seq_length, embedding_dim)
            - 'procedures': (batch_size, seq_length, embedding_dim)
            - 'image': (batch_size, embedding_dim, height, width)
            
            **Label arguments** (optional): May include any label keys defined in 
            ``self.label_keys``. If label keys are present, the method should compute 
            loss and include 'loss' and 'y_true' in the return dictionary. If label 
            keys are absent, the method must not crash; simply omit 'loss' and 'y_true' 
            from the return dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing model outputs with the following keys:
            
            - **logit** (torch.Tensor): Raw model predictions/logits of shape
              (batch_size, num_classes) for classification tasks.
            
            - **y_prob** (torch.Tensor): Predicted probabilities of shape
              (batch_size, num_classes). For binary classification, often 
              shape (batch_size, 1).
            
            - **loss** (torch.Tensor, optional): Scalar loss value if 
              any of ``self.label_keys`` are present in kwargs. Returned only when 
              ground truth labels are provided. Should not be included if labels are 
              unavailable.
            
            - **y_true** (torch.Tensor, optional): True labels if present in 
              kwargs. Useful for consistency checking during attribution. Should not 
              be included if labels are unavailable.
            
            Additional keys may be returned depending on the model's task type
            (e.g., 'risks' for survival analysis, 'seq_output' for sequence models).
                
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
            
        Notes
        -----
        The implementation must gracefully handle missing label keys. Interpretability 
        methods may invoke this method with only embedding inputs (no labels), expecting 
        forward passes for attribution computation. The method should compute predictions 
        successfully in both scenarios.
            
        Examples
        --------
        For an EHR model with embedding dimension 64:
        
        >>> model = MyEHRModel(...)
        >>> batch_embeddings = {
        ...     'conditions': torch.randn(32, 100, 64),  # 32 samples, 100 time steps
        ...     'procedures': torch.randn(32, 100, 64),
        ... }
        >>> output = model.forward_from_embedding(**batch_embeddings)
        >>> logits = output['logit']  # Shape: (32, num_classes)
        >>> y_prob = output['y_prob']  # Shape: (32, num_classes)
        
        With optional labels:
        
        >>> batch_embeddings['mortality'] = torch.tensor([0, 1, 0, ...])  # Add labels
        >>> output = model.forward_from_embedding(**batch_embeddings)
        >>> loss = output['loss']  # Now included
        >>> y_true = output['y_true']  # Now included
        
        For an image model with spatial embeddings:
        
        >>> model = MyImageModel(...)
        >>> batch_embeddings = {
        ...     'image': torch.randn(16, 768, 14, 14),  # Vision Transformer embeddings
        ... }
        >>> output = model.forward_from_embedding(**batch_embeddings)
        """
        raise NotImplementedError

    def get_embedding_model(self) -> nn.Module | None:
        """Get the embedding/feature extraction stage of the model.
        
        This method provides access to the model's embedding stage, which 
        transforms raw input features into learned vector representations.
        This is used by interpretability methods to:
        
        - Generate embeddings from raw features before attribution
        - Identify the boundary between feature processing and prediction
        - Apply embedding-level analysis separately from prediction
        
        Returns
        -------
        nn.Module or None
            The embedding model/stage as an nn.Module if applicable, or None 
            if the model does not have a separable embedding stage.
            
            When returning a model, it should:
            
            - Accept the same input signature as the parent model (raw features)
            - Produce embeddings that are compatible with forward_from_embedding()
            - Be in the same device as the parent model
            
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        
        Examples
        --------
        For a model with explicit embedding and prediction stages:
        
        >>> class MyModel(InterpretableModelInterface):
        ...     def __init__(self):
        ...         self.embedding_layer = EmbeddingBlock(...)
        ...         self.prediction_head = PredictionBlock(...)
        ...         
        ...     def get_embedding_model(self):
        ...         return self.embedding_layer
        
        For models without a clear separable embedding stage, return None:
        
        >>> def get_embedding_model(self):
        ...     return None  # Embeddings are not separately accessible
        """
        raise NotImplementedError


class CheferInterpretable(Interpretable):
    """Abstract interface for models supporting Chefer relevance attribution.

    This is a subclass of :class:`Interpretable` and therefore
    inherits the embedding-level interface (``forward_from_embedding``,
    ``get_embedding_model``).  Models that implement this interface
    automatically satisfy the general interpretability contract **and** the
    Chefer-specific contract, so they work with both embedding-perturbation
    methods (DeepLIFT, LIME, …) and gradient-weighted attention methods
    (Chefer).

    The Chefer algorithm works as follows:

    1. **Forward + hook registration** — run the model while capturing
       attention weight tensors and registering backward hooks so their
       gradients are stored.
    2. **Backward** — back-propagate from a one-hot target class through
       the logits.
    3. **Relevance propagation** — for every feature key, iterate over
       attention layers, compute gradient-weighted attention
       (``clamp(attn * grad, min=0)``), and accumulate into a relevance
       matrix ``R`` via ``R += cam @ R``.
    4. **Attribution extraction** — extract the final per-token
       attribution from ``R`` (e.g. read the CLS row, or the
       last-valid-timestep row, possibly with reshaping).

    Steps 1, 3-b and 4 are model-specific; the rest is generic.  This
    interface captures exactly those model-specific pieces.

    Inherited from ``InterpretableModelInterface``
    -----------------------------------------------
    forward_from_embedding(**kwargs) -> dict[str, Tensor]
        Forward pass starting from pre-computed embeddings.
    get_embedding_model() -> nn.Module | None
        Access the embedding / feature-extraction stage.

    Additional (Chefer-specific) methods
    -------------------------------------
    set_attention_hooks(enabled) -> None
        Toggle attention map capture and gradient hook registration.
    get_attention_layers() -> dict[str, list[tuple[Tensor, Tensor]]]
        Paired (attn_map, attn_grad) for each attention layer, keyed by
        feature key.
    get_relevance_vector(R, **data) -> dict[str, Tensor]
        Reduce relevance matrices to per-token attribution vectors.

    Attributes
    ----------
    feature_keys : list[str]
        The feature keys from the task's ``input_schema`` (e.g.
        ``["conditions", "procedures"]``).  Already provided by
        :class:`~pyhealth.models.base_model.BaseModel`.

    Notes
    -----
    *  ``set_attention_hooks(True)`` must be called **before** the forward
       pass, and ``get_attention_layers`` must be called **after** the
       forward + backward passes, because attention maps are populated
       during forward and gradients during backward.
    *  The interface intentionally does **not** prescribe how hooks are
       registered internally — ``nn.MultiheadAttention`` with
       ``register_hook``, manual ``save_attn_grad`` callbacks, or explicit
       QKV computation all work as long as the getter methods return the
       right tensors.

    Examples
    --------
    Minimal skeleton for a new model:

    >>> class MyAttentionModel(BaseModel, CheferInterpretableModelInterface):
    ...     # feature_keys is inherited from BaseModel
    ...
    ...     def forward_from_embedding(self, **kwargs):
    ...         # ... prediction head from pre-computed embeddings ...
    ...
    ...     def get_embedding_model(self):
    ...         return self.embedding_layer
    ...
    ...     def set_attention_hooks(self, enabled):
    ...         self._register_hooks = enabled
    ...
    ...     def get_attention_layers(self):
    ...         result = {}
    ...         for key in self.feature_keys:
    ...             result[key] = [
    ...                 (blk.attention.get_attn_map(),
    ...                  blk.attention.get_attn_grad())
    ...                 for blk in self.encoder[key].blocks
    ...             ]
    ...         return result
    ...
    ...     def get_relevance_vector(self, R, **data):
    ...         return {key: r[:, 0] for key, r in R.items()}
    """

    @abstractmethod
    def set_attention_hooks(self, enabled: bool) -> None:
        """Toggle attention hook registration for subsequent forward passes.

        When ``enabled=True``, the next call to ``forward()`` (or
        ``forward_from_embedding()``) must:

        1. Store attention weight tensors so they are retrievable via
           :meth:`get_attention_layers`.
        2. Register backward hooks on those tensors so that after
           ``.backward()`` the corresponding gradients are also stored.

        When ``enabled=False``, subsequent forward passes should **not**
        capture attention maps or register gradient hooks, restoring the
        model to its normal (faster) execution mode.

        Parameters
        ----------
        enabled : bool
            ``True`` to start capturing attention maps and registering
            gradient hooks; ``False`` to stop.

        Typical implementations set an internal flag that the model's
        forward method checks::

            def set_attention_hooks(self, enabled):
                self._attention_hooks_enabled = enabled

        And inside the forward / encoder logic::

            if self._attention_hooks_enabled:
                attn.register_hook(self.save_attn_grad)
        """
        ...

    @abstractmethod
    def get_attention_layers(
        self,
    ) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Return (attention_map, attention_gradient) pairs for all feature keys.

        Must be called **after** ``set_attention_hooks(True)``,
        a ``forward()`` call, and a subsequent ``backward()`` call so
        that both attention maps and their gradients are populated.

        Returns
        -------
        dict[str, list[tuple[torch.Tensor, torch.Tensor]]]
            A dictionary keyed by ``feature_keys``.  Each value is a list
            with one ``(attn_map, attn_grad)`` tuple per attention layer,
            ordered from the first (closest to input) to the last
            (closest to output).

            Each tensor may have shape:

            * ``[batch, heads, seq, seq]`` — multi-head (will be
              gradient-weighted-averaged across heads by Chefer).
            * ``[batch, seq, seq]`` — already head-averaged.

            ``attn_map`` and ``attn_grad`` in the same tuple must have
            the same shape.

        Examples
        --------
        A model with stacked ``TransformerBlock`` layers per feature key:

        >>> def get_attention_layers(self):
        ...     return {
        ...         key: [
        ...             (blk.attention.get_attn_map(),
        ...              blk.attention.get_attn_grad())
        ...             for blk in self.transformer[key].transformer
        ...         ]
        ...         for key in self.feature_keys
        ...     }

        A model with a single MHA layer per feature key:

        >>> def get_attention_layers(self):
        ...     return {
        ...         key: [(self.stagenet[key].get_attn_map(),
        ...                self.stagenet[key].get_attn_grad())]
        ...         for key in self.feature_keys
        ...     }
        """
        ...

    @abstractmethod
    def get_relevance_tensor(
        self,
        R: dict[str, torch.Tensor],
        **data: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Reduce relevance matrices to per-token attribution vectors.

        The Chefer algorithm builds a relevance matrix of shape
        ``[batch, seq_len, seq_len]`` for each feature key.  This method
        reduces each matrix to a ``[batch, seq_len]`` vector by selecting
        the row corresponding to the classification position — giving the
        model full control over how the selection is done.

        Parameters
        ----------
        R : dict[str, torch.Tensor]
            Relevance matrices keyed by ``feature_keys``.  Each tensor
            has shape ``[batch, seq_len, seq_len]`` (seq_len may differ
            across keys).
        **data : torch.Tensor or tuple[torch.Tensor, ...]
            The original input data (same kwargs passed to
            ``forward()``).  Available for context when the selection
            logic is data-dependent (e.g. last valid timestep depends on
            mask).

        Returns
        -------
        dict[str, torch.Tensor]
            Attribution vectors keyed by ``feature_keys``.  Each tensor
            has shape ``[batch, seq_len]``.

        Examples
        --------
        CLS-token model (e.g. Transformer) — row 0 for all keys:

        >>> def get_relevance_vector(self, R, **data):
        ...     return {key: r[:, 0] for key, r in R.items()}

        Last-valid-timestep model (e.g. StageAttentionNet):

        >>> def get_relevance_vector(self, R, **data):
        ...     result = {}
        ...     for key, r in R.items():
        ...         mask = self._get_mask(key, **data)
        ...         last_idx = mask.sum(dim=1) - 1
        ...         batch_idx = torch.arange(r.shape[0], device=r.device)
        ...         result[key] = r[batch_idx, last_idx]
        ...     return result
        """
        ...

    # TODO: Add postprocess_attribution() when ViT support is ready.
    # ViT models need to strip the CLS column, reshape the patch vector
    # into a spatial [batch, 1, H, W] map, and optionally interpolate to
    # the original image size.  For EHR models this is a no-op.  We can
    # either fold this into extract_attribution() or add it as a separate
    # optional method.