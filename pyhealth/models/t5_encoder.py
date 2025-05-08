import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os

class LOINCEncoder(tf.keras.Model):
    """
    LOINCEncoder model using T5 backbone from the paper 
    "Automated LOINC Standardization Using Pre-trained Large Language Models"
    
    This model uses a frozen Sentence-T5 encoder backbone and a trainable projection layer
    to create embeddings for LOINC codes and source text strings.
    """
    
    def __init__(self, 
                 embedding_dim=128, 
                 dropout_rate=0.0, 
                 model_url="https://tfhub.dev/google/sentence-t5/st5-base/1"):
        """
        Initialize the LOINCEncoder model
        
        Args:
            embedding_dim: Dimension of the final embedding vector (default: 128)
            dropout_rate: Dropout rate for regularization (default: 0.0, no dropout)
            model_url: TFHub URL for the Sentence-T5 model (default: ST5-base)
        """
        super(LOINCEncoder, self).__init__()
        
        # Load the ST5 encoder from TFHub
        with tf.device('/CPU:0'):  # Force text operations on CPU for compatibility
            self.t5_encoder = hub.KerasLayer(model_url, trainable=False)
        
        # Add dropout layer (only used in Stage 2)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        
        # Add projection layer to reduce embedding dimension
        self.projection_layer = tf.keras.layers.Dense(
            embedding_dim, 
            activation=None, 
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name='projection_layer'
        )
        
        # Add L2 normalization layer
        self.normalize = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name='l2_normalization'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass of the LOINCEncoder model
        
        Args:
            inputs: Text inputs (string or list of strings)
            training: Whether the model is in training mode
            
        Returns:
            embeddings: L2-normalized embeddings of shape (batch_size, embedding_dim)
        """
        # Handle input correctly
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            # Convert list of strings to tensor
            input_tensor = tf.convert_to_tensor(inputs)
        elif isinstance(inputs, str):
            # Convert single string to tensor with batch size 1
            input_tensor = tf.convert_to_tensor([inputs])
        else:
            # Assume it's already a tensor
            input_tensor = inputs
        
        # Get embeddings from T5 encoder
        # Force text operations on CPU for compatibility
        with tf.device('/CPU:0'):
            t5_outputs = self.t5_encoder(input_tensor)
        
        # Extract the embeddings from the T5 outputs
        # The ST5 model can return embeddings in different formats
        # (tensor, list, or dict with 'default' key), so handle all cases
        if isinstance(t5_outputs, dict) and 'default' in t5_outputs:
            embeddings = t5_outputs['default']
        elif isinstance(t5_outputs, list):
            embeddings = t5_outputs[0]
        else:
            # Assume it's a tensor
            embeddings = t5_outputs
        
        # Apply dropout if needed (only in Stage 2 and during training)
        if hasattr(self, 'dropout') and training:
            embeddings = self.dropout(embeddings, training=training)
        
        # Project to lower dimension
        embeddings = self.projection_layer(embeddings)
        
        # Apply L2 normalization
        embeddings = self.normalize(embeddings)
        
        return embeddings 