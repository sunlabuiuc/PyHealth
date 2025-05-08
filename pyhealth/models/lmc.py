import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        
        return self.out_proj(context)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.layers(x))

class LMC(nn.Module):
    def __init__(self, vocab_size, metadata_size, embedding_dim=100, hidden_dim=64, dropout=0.3, num_heads=4):
        super(LMC, self).__init__()
        
        # Model network parameters
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.metadata_embedding = nn.Embedding(metadata_size, embedding_dim)
        
        # Position embeddings for context
        self.position_embedding = nn.Parameter(torch.zeros(1, 100, embedding_dim))  # Max sequence length of 100
        
        # Model network (p_theta) with residual connections
        self.model_net = nn.Sequential(
            ResidualBlock(embedding_dim * 2, dropout),
            ResidualBlock(embedding_dim * 2, dropout),
            nn.Linear(embedding_dim * 2, embedding_dim * 2)
        )
        
        # Variational network (q_phi)
        self.context_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head attention for context encoding
        self.multi_head_attention = MultiHeadAttention(hidden_dim * 2, num_heads, dropout)
        
        # Variational network with residual connections
        self.variational_net = nn.Sequential(
            ResidualBlock(embedding_dim * 2 + hidden_dim * 2, dropout),
            ResidualBlock(embedding_dim * 2 + hidden_dim * 2, dropout),
            nn.Linear(embedding_dim * 2 + hidden_dim * 2, embedding_dim * 2)
        )
        
        # Output projection with residual connections
        self.output_proj = nn.Sequential(
            ResidualBlock(embedding_dim, dropout),
            ResidualBlock(embedding_dim, dropout),
            nn.Linear(embedding_dim, vocab_size)
        )
        
        # Temperature parameter for sampling
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_context(self, context_words):
        # context_words: [batch_size, seq_len]
        context_emb = self.word_embedding(context_words)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional embeddings
        seq_len = context_emb.size(1)
        context_emb = context_emb + self.position_embedding[:, :seq_len, :]
        
        # BiLSTM encoding
        lstm_out, _ = self.context_encoder(context_emb)  # [batch_size, seq_len, hidden_dim*2]
        
        # Multi-head attention
        context_encoding = self.multi_head_attention(lstm_out)  # [batch_size, seq_len, hidden_dim*2]
        context_encoding = torch.mean(context_encoding, dim=1)  # [batch_size, hidden_dim*2]
        
        return context_encoding
    
    def model_network(self, word_idx, metadata_idx):
        # Get embeddings
        word_emb = self.word_embedding(word_idx)  # [batch_size, embedding_dim]
        metadata_emb = self.metadata_embedding(metadata_idx)  # [batch_size, embedding_dim]
        
        # Concatenate
        combined = torch.cat([word_emb, metadata_emb], dim=1)  # [batch_size, embedding_dim*2]
        
        # Get distribution parameters
        params = self.model_net(combined)  # [batch_size, embedding_dim*2]
        mu_p, log_sigma_p = torch.chunk(params, 2, dim=1)
        sigma_p = torch.exp(log_sigma_p)
        
        return mu_p, sigma_p
    
    def variational_network(self, word_idx, metadata_idx, context_words):
        # Get word and metadata embeddings
        word_emb = self.word_embedding(word_idx)  # [batch_size, embedding_dim]
        metadata_emb = self.metadata_embedding(metadata_idx)  # [batch_size, embedding_dim]
        
        # Encode context
        context_encoding = self.encode_context(context_words)  # [batch_size, hidden_dim*2]
        
        # Combine all inputs
        combined = torch.cat([word_emb, metadata_emb, context_encoding], dim=1)
        
        # Get distribution parameters
        params = self.variational_net(combined)
        mu_q, log_sigma_q = torch.chunk(params, 2, dim=1)
        sigma_q = torch.exp(log_sigma_q)
        
        return mu_q, sigma_q
    
    def forward(self, word_idx, metadata_idx, context_words, target_words):
        # Get distributions
        mu_p, sigma_p = self.model_network(word_idx, metadata_idx)
        mu_q, sigma_q = self.variational_network(word_idx, metadata_idx, context_words)
        
        # Sample from variational posterior with temperature
        eps = torch.randn_like(mu_q)
        z = mu_q + sigma_q * eps * self.temperature
        
        # Compute KL divergence with annealing
        p_dist = Normal(mu_p, sigma_p)
        q_dist = Normal(mu_q, sigma_q)
        kl_div = kl_divergence(q_dist, p_dist).mean()
        
        # Reconstruct context words
        logits = self.output_proj(z)  # [batch_size, vocab_size]
        
        # Reshape logits and target_words for cross entropy
        batch_size = logits.size(0)
        logits = logits.unsqueeze(1).expand(-1, target_words.size(1), -1)  # [batch_size, seq_len, vocab_size]
        logits = logits.reshape(-1, logits.size(-1))  # [batch_size*seq_len, vocab_size]
        target_words = target_words.reshape(-1)  # [batch_size*seq_len]
        
        # Compute class weights based on target distribution
        class_counts = torch.bincount(target_words, minlength=logits.size(-1))
        class_weights = 1.0 / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Use weighted cross entropy loss
        reconstruction_loss = F.cross_entropy(logits, target_words, weight=class_weights)
        
        # Total loss with KL annealing and temperature scaling
        kl_weight = min(1.0, self.current_epoch / 10.0)  # Gradually increase KL weight
        loss = reconstruction_loss + kl_weight * kl_div
        
        return {
            'loss': loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_div': kl_div,
            'z': z,
            'temperature': self.temperature.item()
        }
    
    def expand_acronym(self, acronym_idx, metadata_idx, context_words, candidate_expansions):
        # Get variational posterior
        mu_q, sigma_q = self.variational_network(acronym_idx, metadata_idx, context_words)
        
        # Get model distribution for each candidate
        expansion_scores = []
        for expansion_idx in candidate_expansions:
            mu_p, sigma_p = self.model_network(expansion_idx, metadata_idx)
            
            # Compute KL divergence
            p_dist = Normal(mu_p, sigma_p)
            q_dist = Normal(mu_q, sigma_q)
            kl_div = kl_divergence(q_dist, p_dist).mean()
            
            expansion_scores.append(kl_div)
        
        # Return best expansion (lowest KL divergence)
        best_idx = torch.argmin(torch.stack(expansion_scores))
        return candidate_expansions[best_idx] 