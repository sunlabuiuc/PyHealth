from abc import ABC, abstractmethod
import torch
from torch import nn

class InterpretableModelInterface(ABC):
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