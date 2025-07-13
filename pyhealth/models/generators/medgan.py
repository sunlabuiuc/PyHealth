import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

from pyhealth.models import BaseModel


class MedGANAutoencoder(nn.Module):
    """simple autoencoder for pretraining"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)


class MedGANGenerator(nn.Module):
    """generator with residual connections"""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()
    
    def forward(self, x):
        # residual block 1
        residual = x
        out = self.activation1(self.bn1(self.linear1(x)))
        out1 = out + residual
        
        # residual block 2
        residual = out1
        out = self.activation2(self.bn2(self.linear2(out1)))
        out2 = out + residual
        
        return out2


class MedGANDiscriminator(nn.Module):
    """discriminator with minibatch averaging"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, minibatch_averaging: bool = True):
        super().__init__()
        self.minibatch_averaging = minibatch_averaging
        model_input_dim = input_dim * 2 if minibatch_averaging else input_dim
        
        self.model = nn.Sequential(
            nn.Linear(model_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if self.minibatch_averaging:
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)
        return self.model(x)


class MedGAN(BaseModel):
    """MedGAN for binary matrix generation"""
    
    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "generation",
        latent_dim: int = 128,
        hidden_dim: int = 128,
        autoencoder_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        minibatch_averaging: bool = True,
        **kwargs
    ):
        # dummy wrapper for BaseModel compatibility
        class DummyWrapper:
            def __init__(self, dataset, feature_keys, label_key):
                self.dataset = dataset
                self.input_schema = {key: "multilabel" for key in feature_keys}
                self.output_schema = {label_key: "multilabel"}
                self.input_processors = {}
                self.output_processors = {}
        
        wrapped_dataset = DummyWrapper(dataset, feature_keys, label_key)
        super().__init__(dataset=wrapped_dataset)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.minibatch_averaging = minibatch_averaging
        
        # build vocab (simplified)
        self.global_vocab = self._build_global_vocab(dataset, feature_keys)
        self.input_dim = len(self.global_vocab)
        
        # init components
        self.autoencoder = MedGANAutoencoder(input_dim=self.input_dim, hidden_dim=autoencoder_hidden_dim)
        self.generator = MedGANGenerator(latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim)
        self.discriminator = MedGANDiscriminator(
            input_dim=self.input_dim, 
            hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging
        )
        
        self._init_weights()
    
    @classmethod
    def from_binary_matrix(
        cls,
        binary_matrix: np.ndarray,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        autoencoder_hidden_dim: int = 128,
        discriminator_hidden_dim: int = 256,
        minibatch_averaging: bool = True,
        **kwargs
    ):
        """create MedGAN model from binary matrix (ICD-9, etc.)"""
        class MatrixWrapper:
            def __init__(self, matrix):
                self.matrix = matrix
                self.input_processors = {}
                self.output_processors = {}
            
            def __len__(self):
                return self.matrix.shape[0]
            
            def __getitem__(self, idx):
                return {"binary_vector": torch.tensor(self.matrix[idx], dtype=torch.float32)}
            
            def iter_patients(self):
                """iterate over patients"""
                for i in range(len(self)):
                    yield type('Patient', (), {
                        'binary_vector': self.matrix[i],
                        'patient_id': f'patient_{i}'
                    })()
        
        dummy_dataset = MatrixWrapper(binary_matrix)
        
        model = cls(
            dataset=dummy_dataset,
            feature_keys=["binary_vector"],
            label_key="binary_vector",
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            autoencoder_hidden_dim=autoencoder_hidden_dim,
            discriminator_hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging,
            **kwargs
        )
        
        # override input dimension
        model.input_dim = binary_matrix.shape[1]
        
        # reinitialize components with correct dimensions
        model.autoencoder = MedGANAutoencoder(input_dim=model.input_dim, hidden_dim=autoencoder_hidden_dim)
        model.generator = MedGANGenerator(latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim)
        model.discriminator = MedGANDiscriminator(
            input_dim=model.input_dim, 
            hidden_dim=discriminator_hidden_dim,
            minibatch_averaging=minibatch_averaging
        )
        
        # Move all components to the same device as the model
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        model.autoencoder = model.autoencoder.to(device)
        model.generator = model.generator.to(device)
        model.discriminator = model.discriminator.to(device)
        
        # override feature extraction
        def extract_features(batch_data, device):
            return batch_data["binary_vector"].to(device)
        
        model._extract_features_from_batch = extract_features
        
        return model
    
    def _build_global_vocab(self, dataset, feature_keys: List[str]) -> List[str]:
        """build vocab from dataset (simplified)"""
        vocab = set()
        for patient in dataset.iter_patients():
            for feature_key in feature_keys:
                if hasattr(patient, feature_key):
                    feature_values = getattr(patient, feature_key)
                    if isinstance(feature_values, list):
                        vocab.update(feature_values)
                    elif isinstance(feature_values, str):
                        vocab.add(feature_values)
        return sorted(list(vocab))
    
    def _init_weights(self):
        """init weights"""
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.autoencoder.apply(weights_init)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """forward pass"""
        features = self._extract_features_from_batch(kwargs, self.device)
        noise = torch.randn(features.shape[0], self.latent_dim, device=self.device)
        fake_samples = self.generator(noise)
        return {"real_features": features, "fake_samples": fake_samples}
    
    def generate(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """generate synthetic samples"""
        if device is None:
            device = self.device
        
        self.generator.eval()
        self.autoencoder.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=device)
            generated = self.generator(noise)
            # use autoencoder decoder to get final output
            generated = self.autoencoder.decode(generated)
        
        return generated
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """discriminate real vs fake"""
        return self.discriminator(x)
    
    def pretrain_autoencoder(self, dataloader: DataLoader, epochs: int = 100, lr: float = 0.001, device: torch.device = None):
        """pretrain autoencoder with detailed loss tracking"""
        if device is None:
            device = self.device
        
        # Ensure autoencoder is on the correct device
        self.autoencoder = self.autoencoder.to(device)
        
        print("Pretraining Autoencoder...")
        print("="*50)
        print("Epoch | A_loss | Progress")
        print("="*50)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Track losses for plotting
        a_losses = []
        
        self.autoencoder.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                # handle both tensor and dict inputs
                if isinstance(batch, torch.Tensor):
                    features = batch.to(device)
                else:
                    features = self._extract_features_from_batch(batch, device)
                
                reconstructed = self.autoencoder(features)
                loss = criterion(reconstructed, features)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            a_losses.append(avg_loss)
            
            # Print progress every epoch for shorter training runs, every 10 for longer runs
            print_freq = 1 if epochs <= 50 else 10
            if (epoch + 1) % print_freq == 0 or epoch == 0 or epoch == epochs - 1:
                progress = (epoch + 1) / epochs * 100
                print(f"{epoch+1:5d} | {avg_loss:.4f} | {progress:5.1f}%")
        
        print("="*50)
        print("Autoencoder Pretraining Completed!")
        print(f"Final A_loss: {a_losses[-1]:.4f}")
        
        return a_losses
    
    def _extract_features_from_batch(self, batch_data, device: torch.device) -> torch.Tensor:
        """extract features from batch"""
        features = []
        for feature_key in self.feature_keys:
            if feature_key in batch_data:
                features.append(batch_data[feature_key])
        
        if len(features) == 1:
            return features[0].to(device)
        else:
            return torch.cat(features, dim=1).to(device)
    
    def sample_transform(self, samples: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """convert to binary using threshold"""
        return (samples > threshold).float()
    
    def train_step(self, batch, optimizer_g, optimizer_d, optimizer_ae=None):
        """single training step"""
        real_features = self._extract_features_from_batch(batch, self.device)
        
        # train discriminator
        optimizer_d.zero_grad()
        noise = torch.randn(real_features.shape[0], self.latent_dim, device=self.device)
        fake_samples = self.generator(noise)
        
        real_predictions = self.discriminator(real_features)
        fake_predictions = self.discriminator(fake_samples.detach())
        
        d_loss = F.binary_cross_entropy(real_predictions, torch.ones_like(real_predictions)) + \
                 F.binary_cross_entropy(fake_predictions, torch.zeros_like(fake_predictions))
        d_loss.backward()
        optimizer_d.step()
        
        # train generator
        optimizer_g.zero_grad()
        fake_predictions = self.discriminator(fake_samples)
        g_loss = F.binary_cross_entropy(fake_predictions, torch.ones_like(fake_predictions))
        g_loss.backward()
        optimizer_g.step()
        
        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}