import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel, ResBlock2D
from pyhealth.processors import ImageProcessor, MultiHotProcessor


class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)
    
class GAN(BaseModel):
    """GAN model for synthetic patient code sets or small images.

    Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
    Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets.

    A generator turns random noise into a fake sample, and a discriminator
    learns to tell fake samples from real ones; each network trains against
    the other. ``dataset`` must have exactly one input feature, and that
    feature's processor picks the mode:

    * **Code-set mode** (``multi_hot``): each sample is one multi-hot vector
      over the code vocabulary, e.g. a MIMIC-III patient's pooled diagnosis
      codes from :class:`~pyhealth.tasks.EHRCodeSetGenerationMIMIC3`. Both
      networks are MLPs, and :meth:`generate` decodes synthetic patients.
    * **Image mode** (``image``): CNN networks for 32x32, 64x64 or 128x128
      images, sized from the processor's ``image_size`` and the images'
      channels. :meth:`generate` returns a tensor of images.

    The PyHealth ``Trainer`` does not apply: the two networks are updated by
    separate optimizers with opposing objectives, which :meth:`train_model`
    handles.

    Args:
        dataset: a ``SampleDataset`` with exactly one input feature, backed by
            a ``MultiHotProcessor`` or an ``ImageProcessor``.
        hidden_dim: the noise dimension, also the width of the hidden layers.
            Default is 128.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import GAN
        >>> samples = [
        ...     {"patient_id": "p1", "visits": ["A", "B", "C"]},
        ...     {"patient_id": "p2", "visits": ["A", "C", "D"]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"visits": "multi_hot"},
        ...     output_schema={},
        ... )
        >>> model = GAN(dataset=dataset, hidden_dim=16)
        >>> history = model.train_model(dataset, epochs=1, batch_size=2)
        >>> len(model.generate(num_samples=3))
        3
    """

    def __init__(self, dataset=None, hidden_dim: int = 128, **kwargs):
        # Before PyHealth 2.0, GAN was built from input_channel and input_size.
        if (
            dataset is None
            or isinstance(dataset, int)
            or "input_channel" in kwargs
            or "input_size" in kwargs
        ):
            raise TypeError(
                "GAN(input_channel, input_size, ...) is deprecated: GAN now "
                "takes a dataset whose single input feature is 'multi_hot' "
                "(code sets) or 'image' (images), e.g. "
                "GAN(dataset=samples, hidden_dim=128)."
            )
        super().__init__(dataset)
        self.hidden_dim = hidden_dim

        if len(self.feature_keys) != 1:
            raise ValueError(
                "GAN expects exactly one input feature, got "
                f"{self.feature_keys}."
            )
        self.feature_key = self.feature_keys[0]
        processor = dataset.input_processors[self.feature_key]
        if isinstance(processor, MultiHotProcessor):
            self.gan_mode = "code_set"
            self._init_code_set_mode(processor)
        elif isinstance(processor, ImageProcessor):
            self.gan_mode = "image"
            self._init_image_mode(dataset, processor)
        else:
            raise TypeError(
                f"GAN expects '{self.feature_key}' to use a MultiHotProcessor "
                "(input_schema 'multi_hot') or an ImageProcessor ('image'), "
                f"got {type(processor).__name__}."
            )

    def _init_code_set_mode(self, processor: MultiHotProcessor) -> None:
        """Builds MLP networks over the multi-hot code vocabulary."""
        self.input_dim = processor.size()
        self._idx_to_code: list[str | None] = [None] * self.input_dim
        for code, idx in processor.label_vocab.items():
            self._idx_to_code[idx] = code

        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

    def _init_image_mode(self, dataset, processor: ImageProcessor) -> None:
        """Builds CNN networks sized to the dataset's images."""
        input_size = processor.image_size
        if input_size not in (32, 64, 128):
            raise ValueError(
                "GAN's image mode supports an image_size of 32, 64 or 128, got "
                f"{input_size}; set it in the input_schema, e.g. "
                "(\"image\", {\"image_size\": 64})."
            )
        if not processor.to_tensor or processor.normalize:
            raise ValueError(
                "GAN's image mode needs images as tensors in [0, 1], the "
                "generator's output range: use to_tensor=True and "
                "normalize=False."
            )
        input_channel = dataset[0][self.feature_key].shape[0]

        # encoder part
        if input_size == 128:
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
            
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
            
        elif input_size == 64:
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )

            self.generator = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 128, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
    
        elif input_size == 32:
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                Flatten(),
                nn.Linear(64 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
            
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
            
    def discriminate(self, x) -> torch.Tensor:
        y = self.discriminator(x)
        return y

    def sampling(self, n_samples, device) -> torch.Tensor:
        if self.gan_mode == "code_set":
            return torch.randn(n_samples, self.hidden_dim, device=device)
        eps = torch.randn(n_samples, self.hidden_dim, 1, 1).to(device)
        return eps

    def generate_fake(self, n_samples, device) -> torch.Tensor:
        eps = self.sampling(n_samples, device)
        fake_images = self.generator(eps)
        return fake_images

    @staticmethod
    def _bce(prob: torch.Tensor, target: float) -> torch.Tensor:
        """Binary cross-entropy of discriminator outputs against one label."""
        return F.binary_cross_entropy(prob, torch.full_like(prob, target))

    def _discriminator_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Rewards calling real samples real (1) and fakes fake (0).

        ``fake`` is detached so this loss only reaches the discriminator.
        """
        real_loss = self._bce(self.discriminator(real), 1.0)
        fake_loss = self._bce(self.discriminator(fake.detach()), 0.0)
        return (real_loss + fake_loss) / 2

    def _generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """Rewards fakes that the discriminator calls real (1)."""
        return self._bce(self.discriminator(fake), 1.0)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Scores a batch of real samples against as many fresh fakes.

        Only computes the losses; :meth:`train_model` runs the optimization.

        Args:
            **kwargs: a batch from ``get_dataloader``, holding the input
                feature as a ``(batch_size, input_dim)`` multi-hot tensor in
                code-set mode, or ``(batch_size, channels, size, size)``
                images in image mode.

        Returns:
            A dict with the scalar ``d_loss`` and ``g_loss``, the
            discriminator's ``real_prob`` and ``fake_prob`` (each
            ``(batch_size, 1)``), and the generated ``fake`` samples, shaped
            like the real batch.
        """
        real = kwargs[self.feature_key].to(self.device).float()
        fake = self.generate_fake(real.shape[0], self.device)
        return {
            "d_loss": self._discriminator_loss(real, fake),
            "g_loss": self._generator_loss(fake),
            "real_prob": self.discriminator(real),
            "fake_prob": self.discriminator(fake),
            "fake": fake,
        }

    def train_model(
        self,
        train_dataset,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 2e-4,
        device=None,
    ) -> dict[str, list[float]]:
        """Trains the GAN, alternating the two networks on every batch.

        Each batch takes one discriminator step (tell the real batch from a
        fresh batch of fakes) and then one generator step (make those fakes
        pass as real). Named ``train_model`` so it does not shadow
        ``nn.Module.train()``. Call ``torch.manual_seed`` first for a
        reproducible run.

        Args:
            train_dataset: a ``SampleDataset`` with the same input feature
                as the one the model was built from, e.g. its training split.
            epochs: number of passes over ``train_dataset``. Default is 50.
            batch_size: samples per step. Default is 64.
            lr: Adam learning rate for both networks. Default is 2e-4.
            device: device to train on. Default is CUDA when available,
                otherwise CPU.

        Returns:
            The mean loss of each epoch, as ``{"d_loss": [...],
            "g_loss": [...]}``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        loader = get_dataloader(train_dataset, batch_size, shuffle=True)

        # beta1 = 0.5 is the usual Adam setting for GANs (Radford et al., 2016).
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        history: dict[str, list[float]] = {"d_loss": [], "g_loss": []}
        for _ in tqdm(range(epochs), desc="GAN training"):
            d_total, g_total, n_batches = 0.0, 0.0, 0
            for batch in loader:
                real = batch[self.feature_key].to(self.device).float()
                fake = self.generate_fake(real.shape[0], self.device)

                opt_d.zero_grad()
                d_loss = self._discriminator_loss(real, fake)
                d_loss.backward()
                opt_d.step()

                opt_g.zero_grad()
                g_loss = self._generator_loss(fake)
                g_loss.backward()
                opt_g.step()

                d_total += d_loss.item()
                g_total += g_loss.item()
                n_batches += 1
            history["d_loss"].append(d_total / max(n_batches, 1))
            history["g_loss"].append(g_total / max(n_batches, 1))
        return history

    def generate(
        self, num_samples: int, random_sampling: bool = False
    ) -> list[dict] | torch.Tensor:
        """Generates synthetic samples.

        In code-set mode the generator gives each code a probability; a
        patient has the codes at or above 0.5, or, with ``random_sampling``,
        a coin flip per code weighted by its probability.

        Args:
            num_samples: number of synthetic samples.
            random_sampling: code-set mode only; sample each code instead of
                thresholding. Default is False.

        Returns:
            In code-set mode, one
            ``{"patient_id": "synthetic_i", "visits": [[code, ...]]}`` dict
            per patient, the format of :meth:`pyhealth.models.MedGAN.generate`:
            a single visit holding the patient's codes, possibly none. In
            image mode, a ``(num_samples, channels, size, size)`` tensor of
            images in ``[0, 1]``.
        """
        with torch.no_grad():
            probs = self.generate_fake(num_samples, self.device)
            if self.gan_mode == "image":
                if random_sampling:
                    raise ValueError(
                        "random_sampling only applies to code-set mode."
                    )
                return probs.cpu()
            if random_sampling:
                rows = torch.bernoulli(probs)
            else:
                rows = (probs >= 0.5).float()

        results = []
        for i, row in enumerate(rows.cpu()):
            codes = [
                self._idx_to_code[idx]
                for idx in row.nonzero(as_tuple=True)[0].tolist()
                if self._idx_to_code[idx] not in (None, "<pad>", "<unk>")
            ]
            results.append({"patient_id": f"synthetic_{i}", "visits": [codes]})
        return results
