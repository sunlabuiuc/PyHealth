import functools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import time

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer


class CorGANDataset(Dataset):
    """Dataset wrapper for CorGAN training"""
    
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data.astype(np.float32)
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
            pass

        return torch.from_numpy(sample)


class CorGANAutoencoder(nn.Module):
    """Autoencoder for CorGAN - uses 1D convolutions to capture correlations"""
    
    def __init__(self, feature_size: int):
        super(CorGANAutoencoder, self).__init__()
        n_channels_base = 4
        
        # calculate the size after convolutions
        # input: (batch, 1, feature_size)
        # conv1: kernel=5, stride=2 -> (batch, 4, (feature_size-4)//2)
        # conv2: kernel=5, stride=2 -> (batch, 8, ((feature_size-4)//2-4)//2)
        # conv3: kernel=5, stride=3 -> (batch, 16, (((feature_size-4)//2-4)//2-4)//3)
        # conv4: kernel=5, stride=3 -> (batch, 32, ((((feature_size-4)//2-4)//2-4)//3-4)//3)
        # conv5: kernel=5, stride=3 -> (batch, 64, (((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3)
        # conv6: kernel=8, stride=1 -> (batch, 128, ((((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3-7))
        
        # rough estimate for latent size
        latent_size = max(1, feature_size // 100)  # ensure at least 1
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=8, stride=1,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        # decoder - reverse of encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=8, stride=1,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=3,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=5, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=5, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, x):
        return self.decoder(x)


class CorGANGenerator(nn.Module):
    """Generator for CorGAN - MLP that generates latent representations"""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super(CorGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Layer 1
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        
        # Layer 2  
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.ReLU()
        
        # Layer 3
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation3 = nn.Tanh()

    def forward(self, x):
        # Layer 1
        out = self.activation1(self.bn1(self.linear1(x)))
        
        # Layer 2
        out = self.activation2(self.bn2(self.linear2(out)))
        
        # Layer 3
        out = self.activation3(self.bn3(self.linear3(out)))
        
        return out


class CorGANDiscriminator(nn.Module):
    """Discriminator for CorGAN - MLP with minibatch averaging"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, minibatch_averaging: bool = True):
        super(CorGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.minibatch_averaging = minibatch_averaging
        
        # adjust input dimension for minibatch averaging
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
            # minibatch averaging: concatenate batch mean to each sample
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)
        
        output = self.model(x)
        return output


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def autoencoder_loss(x_output, y_target):
    """Autoencoder reconstruction loss"""
    return F.binary_cross_entropy(x_output, y_target, reduction='mean')


def discriminator_accuracy(predicted, y_true):
    """Calculate discriminator accuracy"""
    predicted = (predicted >= 0.5).float()
    accuracy = (predicted == y_true).float().mean()
    return accuracy.item()


class CorGAN(BaseModel):
    """
    CorGAN: Correlation-capturing Generative Adversarial Network
    
    Uses CNNs to capture correlations between adjacent medical features by combining
    Convolutional GANs with Convolutional Autoencoders.
    
    Args:
        dataset: PyHealth dataset object
        feature_keys: List of feature keys to use
        label_key: Label key (not used in unsupervised generation)
        mode: Training mode (not used in GAN context)
        latent_dim: Dimensionality of latent space
        hidden_dim: Hidden dimension for networks
        batch_size: Training batch size
        n_epochs: Number of training epochs
        n_epochs_pretrain: Number of autoencoder pretraining epochs
        lr: Learning rate
        weight_decay: Weight decay for optimization
        b1: Beta1 for Adam optimizer
        b2: Beta2 for Adam optimizer
        n_iter_D: Number of discriminator iterations per generator iteration
        clamp_lower: Lower bound for weight clipping
        clamp_upper: Upper bound for weight clipping
        minibatch_averaging: Whether to use minibatch averaging in discriminator
        **kwargs: Additional arguments
    
    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(...)
        >>> model = CorGAN(dataset=dataset, feature_keys=["conditions"])
        >>> model.fit()
        >>> synthetic_data = model.generate(n_samples=50000)
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "generation",
        latent_dim: int = 128,
        hidden_dim: int = 128,
        batch_size: int = 512,
        n_epochs: int = 1000,
        n_epochs_pretrain: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        b1: float = 0.9,
        b2: float = 0.999,
        n_iter_D: int = 5,
        clamp_lower: float = -0.01,
        clamp_upper: float = 0.01,
        minibatch_averaging: bool = True,
        **kwargs
    ):
        super(CorGAN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            **kwargs
        )
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_epochs_pretrain = n_epochs_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = b1
        self.b2 = b2
        self.n_iter_D = n_iter_D
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        self.minibatch_averaging = minibatch_averaging
        
        # build unified vocabulary for all feature keys
        self.global_vocab = self._build_global_vocab(dataset, feature_keys)
        self.input_dim = len(self.global_vocab)
        self.tokenizer = Tokenizer(tokens=self.global_vocab, special_tokens=[])
        
        # initialize components
        self.autoencoder = CorGANAutoencoder(feature_size=self.input_dim)
        self.autoencoder_decoder = self.autoencoder.decoder  # separate decoder for generator
        
        self.generator = CorGANGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.discriminator = CorGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim * 2,
            minibatch_averaging=minibatch_averaging
        )
        
        # apply custom weight initialization
        self._init_weights()
        
        # setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # setup optimizers
        g_params = [
            {'params': self.generator.parameters()},
            {'params': self.autoencoder_decoder.parameters(), 'lr': 1e-4}
        ]
        self.optimizer_G = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        
        # setup tensors for training
        self.one = torch.tensor(1.0, device=self.device)
        self.mone = torch.tensor(-1.0, device=self.device)
    
    def _build_global_vocab(self, dataset: BaseDataset, feature_keys: List[str]) -> List[str]:
        """Build unified vocabulary across all feature keys"""
        global_vocab = set()
        
        # collect all unique codes from all patients and feature keys
        for patient_id in dataset.patients:
            patient = dataset.patients[patient_id]
            for feature_key in feature_keys:
                if feature_key in patient:
                    for visit in patient[feature_key]:
                        if isinstance(visit, list):
                            global_vocab.update(visit)
                        else:
                            global_vocab.add(visit)
        
        return sorted(list(global_vocab))
    
    def _encode_patient_record(self, record: Dict) -> torch.Tensor:
        """Encode a patient record to binary vector"""
        # create binary vector
        binary_vector = np.zeros(self.input_dim, dtype=np.float32)
        
        for feature_key in self.feature_keys:
            if feature_key in record:
                for visit in record[feature_key]:
                    if isinstance(visit, list):
                        for code in visit:
                            if code in self.global_vocab:
                                idx = self.global_vocab.index(code)
                                binary_vector[idx] = 1.0
                    else:
                        if visit in self.global_vocab:
                            idx = self.global_vocab.index(visit)
                            binary_vector[idx] = 1.0
        
        return torch.from_numpy(binary_vector)
    
    def _init_weights(self):
        """Initialize network weights"""
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.autoencoder.apply(weights_init)
    
    def _extract_features_from_batch(self, batch_data, device: torch.device) -> torch.Tensor:
        """Extract features from batch data"""
        features = []
        for patient_id in batch_data:
            patient = self.dataset.patients[patient_id]
            feature_vector = self._encode_patient_record(patient)
            features.append(feature_vector)
        
        return torch.stack(features).to(device)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass - not used in GAN context"""
        raise NotImplementedError("Forward pass not implemented for GAN models")
    
    def fit(self, train_dataloader: Optional[DataLoader] = None):
        """Train the CorGAN model"""
        print("Starting CorGAN training...")
        
        # create dataset and dataloader
        if train_dataloader is None:
            # create binary matrix from dataset
            data_matrix = []
            for patient_id in self.dataset.patients:
                patient = self.dataset.patients[patient_id]
                feature_vector = self._encode_patient_record(patient)
                data_matrix.append(feature_vector.numpy())
            
            data_matrix = np.array(data_matrix)
            dataset = CorGANDataset(data=data_matrix)
            
            sampler = torch.utils.data.sampler.RandomSampler(
                data_source=dataset, replacement=True
            )
            train_dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                shuffle=False, 
                num_workers=0, 
                drop_last=True, 
                sampler=sampler
            )
        
        # pretrain autoencoder
        print(f"Pretraining autoencoder for {self.n_epochs_pretrain} epochs...")
        for epoch_pre in range(self.n_epochs_pretrain):
            for i, samples in enumerate(train_dataloader):
                # configure input
                real_samples = samples.to(self.device)
                
                # generate a batch of images
                recons_samples = self.autoencoder(real_samples)
                
                # loss measures autoencoder's ability to reconstruct
                a_loss = autoencoder_loss(recons_samples, real_samples)
                
                # reset gradients
                self.optimizer_A.zero_grad()
                a_loss.backward()
                self.optimizer_A.step()
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch_pre + 1}/{self.n_epochs_pretrain}] [Batch {i}/{len(train_dataloader)}] [A loss: {a_loss.item():.3f}]")
        
        # adversarial training
        print(f"Starting adversarial training for {self.n_epochs} epochs...")
        gen_iterations = 0
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            for i, samples in enumerate(train_dataloader):
                # adversarial ground truths
                valid = torch.ones(samples.shape[0], device=self.device)
                fake = torch.zeros(samples.shape[0], device=self.device)
                
                # configure input
                real_samples = samples.to(self.device)
                
                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                
                # train the discriminator n_iter_D times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    n_iter_D = 100
                else:
                    n_iter_D = self.n_iter_D
                
                j = 0
                while j < n_iter_D:
                    j += 1
                    
                    # clamp parameters to a cube
                    for p in self.discriminator.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)
                    
                    # reset gradients of discriminator
                    self.optimizer_D.zero_grad()
                    
                    errD_real = torch.mean(self.discriminator(real_samples), dim=0)
                    errD_real.backward(self.one)
                    
                    # sample noise as generator input
                    z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                    
                    # generate a batch of images
                    fake_samples = self.generator(z)
                    fake_samples = torch.squeeze(self.autoencoder_decoder(fake_samples.unsqueeze(dim=2)))
                    
                    errD_fake = torch.mean(self.discriminator(fake_samples.detach()), dim=0)
                    errD_fake.backward(self.mone)
                    errD = errD_real - errD_fake
                    
                    # optimizer step
                    self.optimizer_D.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                
                # zero grads
                self.optimizer_G.zero_grad()
                
                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)
                
                # generate a batch of images
                fake_samples = self.generator(z)
                fake_samples = torch.squeeze(self.autoencoder_decoder(fake_samples.unsqueeze(dim=2)))
                
                # loss measures generator's ability to fool the discriminator
                errG = torch.mean(self.discriminator(fake_samples), dim=0)
                errG.backward(self.one)
                
                # optimizer step
                self.optimizer_G.step()
                gen_iterations += 1
            
            # end of epoch
            epoch_end = time.time()
            print(f"[Epoch {epoch + 1}/{self.n_epochs}] [Batch {i}/{len(train_dataloader)}] "
                  f"Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} "
                  f"Loss_D_real: {errD_real.item():.3f} Loss_D_fake: {errD_fake.item():.3f}")
            print(f"Epoch time: {epoch_end - epoch_start:.2f} seconds")
        
        print("Training completed!")
    
    def generate(self, n_samples: int, device: torch.device = None) -> torch.Tensor:
        """Generate synthetic data"""
        if device is None:
            device = self.device
        
        # set models to eval mode
        self.generator.eval()
        self.autoencoder_decoder.eval()
        
        # generate samples
        gen_samples = np.zeros((n_samples, self.input_dim), dtype=np.float32)
        n_batches = int(n_samples / self.batch_size)
        
        with torch.no_grad():
            for i in range(n_batches):
                # sample noise as generator input
                z = torch.randn(self.batch_size, self.latent_dim, device=device)
                gen_samples_tensor = self.generator(z)
                gen_samples_decoded = torch.squeeze(self.autoencoder_decoder(gen_samples_tensor.unsqueeze(dim=2)))
                gen_samples[i * self.batch_size:(i + 1) * self.batch_size, :] = gen_samples_decoded.cpu().data.numpy()
        
        # handle remaining samples
        remaining = n_samples % self.batch_size
        if remaining > 0:
            z = torch.randn(remaining, self.latent_dim, device=device)
            gen_samples_tensor = self.generator(z)
            gen_samples_decoded = torch.squeeze(self.autoencoder_decoder(gen_samples_tensor.unsqueeze(dim=2)))
            gen_samples[n_batches * self.batch_size:, :] = gen_samples_decoded.cpu().data.numpy()
        
        # binarize output
        gen_samples[gen_samples >= 0.5] = 1.0
        gen_samples[gen_samples < 0.5] = 0.0
        
        return torch.from_numpy(gen_samples)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'autoencoder_decoder_state_dict': self.autoencoder_decoder.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_A_state_dict': self.optimizer_A.state_dict(),
            'global_vocab': self.global_vocab,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.autoencoder_decoder.load_state_dict(checkpoint['autoencoder_decoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])
        
        self.global_vocab = checkpoint['global_vocab']
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim'] 