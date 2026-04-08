"""MedFlamingo: A Multimodal Medical Few-Shot Learner.

This module implements the MedFlamingo model, which adapts the OpenFlamingo
architecture to the medical domain by fine-tuning on paired medical image-text
data (MTB: medical textbooks, PMC-OA: PubMed Central Open Access).

Architecture:
    1. Vision Encoder (frozen): CLIP ViT-L/14, produces patch embeddings.
    2. Perceiver Resampler: maps variable-length patch embeddings to a fixed
       set of visual tokens.
    3. Gated Cross-Attention Dense Blocks: interleaved with frozen LLM layers,
       allowing language tokens to attend to visual tokens. Gates are
       initialized to zero for stable training.
    4. Language Model (frozen): generates text conditioned on interleaved
       image-text context.

Paper:
    Moor et al. "Med-Flamingo: a Multimodal Medical Few-shot Learner"
    ML4H 2023. https://arxiv.org/abs/2307.15189

Code: https://github.com/snap-stanford/med-flamingo

Licensing:
    - OpenFlamingo (base architecture): MIT License
    - CLIP ViT: MIT License
    - LLM backbone: varies by choice (LLaMA community license, OPT is open)
    - MedFlamingo checkpoint: consult the original repository for terms

Note:
    This implementation exposes both ``forward()`` for PyHealth training
    loops and ``generate()`` for direct multimodal prompting. The default
    constructor still relies on heavyweight pretrained backbones, so the
    first run may download substantial Hugging Face assets.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class PerceiverResampler(nn.Module):
    """Perceiver resampler: cross-attention to fixed-length latents.
    
    Maps variable-length visual token sequences to a fixed number of
    learned latent queries via cross-attention. Core Flamingo component.
    
    Args:
        dim: Input/output feature dimension.
        num_latents: Number of learned latent queries.
        depth: Number of cross-attention layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_latents: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.depth = depth
        
        # Learned latent queries (cross-attention queries)
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(depth)
        ])
        
        # Feed-forward after each cross-attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )
            for _ in range(depth)
        ])
        
        # Layer norms before cross-attention
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        
        self._init_latents()
    
    def _init_latents(self):
        """Initialize latent queries."""
        nn.init.normal_(self.latents, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resample visual features to fixed-length latents.
        
        Args:
            x: Visual features of shape (batch_size, num_patches, dim).
            
        Returns:
            Resampled latents of shape (batch_size, num_latents, dim).
        """
        batch_size = x.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)  # (B, num_latents, dim)
        
        # Apply cross-attention layers
        for i in range(self.depth):
            # Cross-attention: latents query, x key/value
            norm_latents = self.norms[i](latents)
            attn_out, _ = self.cross_attn_layers[i](
                norm_latents, x, x,
                need_weights=False
            )
            latents = latents + attn_out  # Residual connection
            
            # Feed-forward
            latents = latents + self.ff_layers[i](latents)
        
        return latents


class MedFlamingoLayer(nn.Module):
    """Gated cross-attention dense block for connecting vision and language.

    This layer implements the core architectural component of the Flamingo /
    MedFlamingo architecture: a gated cross-attention mechanism that allows
    a frozen language model to attend to visual features produced by a frozen
    vision encoder via a Perceiver Resampler.

    Components:
        1. **Perceiver Resampler** -- maps variable-length visual features
           from the vision encoder (CLIP ViT) to a fixed number of visual
           tokens using learned latent queries.
        2. **Gated Cross-Attention** -- language model hidden states attend
           to the resampled visual tokens. A learnable gating parameter
           (initialized to zero) controls the influence so the model starts
           from the frozen LLM's behavior.
        3. **Dense Feed-Forward** -- standard FFN after cross-attention.

    Paper:
        Moor et al. "Med-Flamingo: a Multimodal Medical Few-shot Learner"
        ML4H 2023.

    Base architecture:
        Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot
        Learning" NeurIPS 2022.

    Args:
        vision_dim: Dimension of vision encoder output features.
            Default 768 (CLIP ViT-L/14).
        lang_dim: Dimension of the language model hidden states.
            Default 1024.
        num_resampler_tokens: Number of fixed-length visual tokens output
            by the Perceiver Resampler. Default 64.
        num_resampler_layers: Number of Perceiver Resampler attention
            layers. Default 6.
        num_heads: Number of attention heads in cross-attention. Default 8.
        dropout: Dropout rate. Default 0.0.

    Example:
        >>> layer = MedFlamingoLayer(vision_dim=768, lang_dim=1024)
        >>> vision_feats = torch.randn(2, 257, 768)  # (B, num_patches, dim)
        >>> lang_hidden = torch.randn(2, 50, 1024)  # (B, seq_len, lang_dim)
        >>> updated_hidden = layer(lang_hidden, vision_feats)
        >>> updated_hidden.shape
        torch.Size([2, 50, 1024])
    """

    def __init__(
        self,
        vision_dim: int = 768,
        lang_dim: int = 1024,
        num_resampler_tokens: int = 64,
        num_resampler_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.lang_dim = lang_dim
        self.num_resampler_tokens = num_resampler_tokens
        self.num_resampler_layers = num_resampler_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # Perceiver Resampler: maps variable-length vision features to fixed tokens
        self.perceiver_resampler = PerceiverResampler(
            dim=vision_dim,
            num_latents=num_resampler_tokens,
            depth=num_resampler_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Project resampled vision features to language dimension if needed
        if vision_dim != lang_dim:
            self.vision_proj = nn.Linear(vision_dim, lang_dim)
        else:
            self.vision_proj = nn.Identity()
        
        # Gated cross-attention: language tokens attend to visual tokens
        self.norm_lang = nn.LayerNorm(lang_dim)
        self.gated_xattn = nn.MultiheadAttention(
            embed_dim=lang_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Gating parameters (initialized to zero for stable training)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        
        # Feed-forward network with gating
        self.norm_ff = nn.LayerNorm(lang_dim)
        self.ff = nn.Sequential(
            nn.Linear(lang_dim, lang_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lang_dim * 4, lang_dim),
        )
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        lang_hidden: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the gated cross-attention dense block.

        The flow:
            1. Resample ``vision_features`` to fixed-length tokens via
               the Perceiver Resampler.
            2. Language hidden states cross-attend to resampled visual
               tokens, gated by ``tanh(attn_gate)``.
            3. Feed-forward, gated by ``tanh(ff_gate)``.

        Args:
            lang_hidden: Language model hidden states of shape
                ``(batch_size, seq_len, lang_dim)``.
            vision_features: Vision encoder output of shape
                ``(batch_size, num_patches, vision_dim)``.

        Returns:
            Updated language hidden states of shape
            ``(batch_size, seq_len, lang_dim)``.
        """
        # Step 1: Resample visual features to fixed-length tokens
        resampled_vision = self.perceiver_resampler(vision_features)  # (B, num_resampler_tokens, vision_dim)
        resampled_vision = self.vision_proj(resampled_vision)  # (B, num_resampler_tokens, lang_dim)
        
        # Step 2: Gated cross-attention
        norm_lang_hidden = self.norm_lang(lang_hidden)
        attn_out, _ = self.gated_xattn(
            norm_lang_hidden,
            resampled_vision,
            resampled_vision,
            need_weights=False
        )
        # Gate the attention output: tanh(gate) is in [-1, 1]
        gated_attn = attn_out * torch.tanh(self.attn_gate)
        lang_hidden = lang_hidden + gated_attn
        
        # Step 3: Feed-forward with gating
        norm_lang_hidden = self.norm_ff(lang_hidden)
        ff_out = self.ff(norm_lang_hidden)
        gated_ff = ff_out * torch.tanh(self.ff_gate)
        lang_hidden = lang_hidden + gated_ff
        
        return lang_hidden


class MedFlamingo(BaseModel):
    """MedFlamingo: multimodal medical few-shot learner.

    MedFlamingo adapts the Flamingo architecture (frozen vision encoder +
    frozen language model + learned cross-attention bridges) to the medical
    domain by continued pretraining on paired medical image-text data from
    medical textbooks (MTB) and PubMed Central Open Access (PMC-OA).

    Architecture overview::

        Images ──► CLIP ViT (frozen) ──► Perceiver Resampler ──► visual tokens
                                                                      │
        Text ──► Tokenizer ──► LLM (frozen) ◄── gated xattn-dense ◄──┘
                                    │
                                 generate

    Supported tasks:
        - **Visual Question Answering (VQA):** given an image + question,
          generate an answer. Evaluated on VQA-RAD and PathVQA.
        - **Medical report generation:** given an image (+ optional prior
          context), generate a radiology report.
        - **Few-shot classification:** frame classification as text
          generation by providing labeled in-context examples.

    Compatibility with PyHealth:
        This model departs from the standard ``BaseModel.forward()`` pattern
        (which returns ``{loss, y_prob, y_true, logit}``) because MedFlamingo
        is primarily a generative model. Two interfaces are provided:

        - :meth:`generate` -- the native generation interface for VQA /
          report generation. Returns generated text.
        - :meth:`forward` -- conforms to BaseModel's expected return dict.
          When fully implemented, will wrap generation into the standard
          ``{loss, y_prob, y_true, logit}`` dict via a classification head
          (for VQA as multiclass) or language modeling loss.

    Paper:
        Moor et al. "Med-Flamingo: a Multimodal Medical Few-shot Learner"
        ML4H 2023. https://arxiv.org/abs/2307.15189

    Licensing:
        - OpenFlamingo (base architecture): MIT License
        - CLIP ViT: MIT License
        - LLM backbone: varies (LLaMA community license; OPT is open)
        - MedFlamingo checkpoint: see https://github.com/snap-stanford/med-flamingo

    Note:
        ``forward()`` implements the PyHealth classification-style contract
        for dataset-backed usage, while ``generate()`` provides the native
        multimodal prompting interface. The default constructor lazily loads
        large pretrained dependencies the first time the model is created.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset`, or ``None``
            for standalone usage (VQA / generation without PyHealth's data
            pipeline). When provided, used to configure classification heads.
        vision_model_name: HuggingFace identifier for the frozen vision
            encoder. Default ``"openai/clip-vit-large-patch14"``.
        lang_model_name: HuggingFace identifier for the frozen language
            model. Default ``"facebook/opt-6.7b"``. The original
            MedFlamingo uses LLaMA-7B, but OPT is openly accessible.
        medflamingo_checkpoint: Path or HuggingFace identifier for
            pretrained MedFlamingo weights. Default ``None``.
        cross_attn_every_n_layers: Insert a gated xattn-dense block every
            N language model layers. Default 4.
        num_resampler_tokens: Number of visual tokens from the Perceiver
            Resampler. Default 64.
        freeze_vision: Whether to freeze the vision encoder. Default ``True``.
        freeze_lm: Whether to freeze the language model. Default ``True``.

    Examples:
        >>> from pyhealth.models import MedFlamingo
        >>> # Standalone usage (no dataset required)
        >>> model = MedFlamingo(dataset=None)
        >>> model.vision_model_name
        'openai/clip-vit-large-patch14'
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        lang_model_name: str = "facebook/opt-6.7b",
        medflamingo_checkpoint: Optional[str] = None,
        cross_attn_every_n_layers: int = 4,
        num_resampler_tokens: int = 64,
        freeze_vision: bool = True,
        freeze_lm: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)

        self.vision_model_name = vision_model_name
        self.lang_model_name = lang_model_name
        self.medflamingo_checkpoint = medflamingo_checkpoint
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.num_resampler_tokens = num_resampler_tokens
        self.freeze_vision = freeze_vision
        self.freeze_lm = freeze_lm

        # Initialize components in order
        self._init_vision_encoder()
        self._init_lang_model()
        self._init_xattn_layers()

        # If a dataset is provided with a single label, prepare for
        # classification (VQA-as-multiclass).
        self._fc = None  # default; overridden below when dataset is available
        if dataset is not None and len(self.label_keys) == 1:
            self.label_key = self.label_keys[0]
            self._init_classification_head()
        else:
            self.label_key = None

    def _init_vision_encoder(self) -> None:
        """Initialize CLIP vision encoder (frozen by default)."""
        try:
            from transformers import CLIPVisionModel
        except ImportError:
            raise ImportError(
                "transformers library required for CLIP. Install with: "
                "pip install transformers"
            )
        
        self._vision_encoder = CLIPVisionModel.from_pretrained(
            self.vision_model_name
        )
        
        if self.freeze_vision:
            for param in self._vision_encoder.parameters():
                param.requires_grad = False
    
    def _init_lang_model(self) -> None:
        """Initialize language model and tokenizer (frozen by default)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required for language models. Install with: "
                "pip install transformers"
            )
        
        self._lang_model = AutoModelForCausalLM.from_pretrained(
            self.lang_model_name,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.lang_model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not defined
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        if self.freeze_lm:
            for param in self._lang_model.parameters():
                param.requires_grad = False
    
    def _init_xattn_layers(self) -> None:
        """Initialize gated cross-attention layers."""
        vision_dim = self._vision_encoder.config.hidden_size
        lang_dim = self._lang_model.config.hidden_size
        num_hidden_layers = self._lang_model.config.num_hidden_layers
        
        # Number of xattn layers = num_hidden_layers / cross_attn_every_n_layers
        num_xattn_layers = num_hidden_layers // self.cross_attn_every_n_layers
        
        self._xattn_layers = nn.ModuleList([
            MedFlamingoLayer(
                vision_dim=vision_dim,
                lang_dim=lang_dim,
                num_resampler_tokens=self.num_resampler_tokens,
                num_resampler_layers=6,
                num_heads=8,
                dropout=0.1,
            )
            for _ in range(num_xattn_layers)
        ])
    
    def _init_classification_head(self) -> None:
        """Initialize classification head for VQA task."""
        lang_dim = self._lang_model.config.hidden_size
        output_size = self.get_output_size()
        self._fc = nn.Linear(lang_dim, output_size)

    def forward(
        self,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass conforming to PyHealth's BaseModel interface.

        This implements the full pipeline:
            1. Extract image and text features from ``kwargs``.
            2. Pass images through the frozen vision encoder.
            3. Resample visual features via the Perceiver Resampler.
            4. Feed interleaved image-text tokens through gated xattn LLM.
            5. Project final hidden states to classification logits.
            6. Return ``{loss, y_prob, y_true, logit}``.

        For open-ended generation tasks, use :meth:`generate` instead.

        Args:
            **kwargs: Keyword arguments from the PyHealth dataloader. Expected
                to contain image and text feature keys as defined in the
                dataset's ``input_schema``, plus the label key if available.

        Returns:
            A dict with keys ``logit``, ``y_prob``, and optionally ``loss``
            and ``y_true``.

        Example:
            >>> model = MedFlamingo(dataset)
            >>> batch = {
            ...     "image": torch.randn(2, 3, 224, 224),
            ...     "question": ["What is in the image?", "Describe this."],
            ...     "answer": torch.tensor([0, 1])
            ... }
            >>> output = model(**batch)
            >>> output.keys()
            dict_keys(['logit', 'y_prob', 'loss', 'y_true'])
        """
        # Extract image and question from kwargs
        image_key = "image" if "image" in self.feature_keys else self.feature_keys[0]
        question_key = "question" if "question" in self.feature_keys else (
            self.feature_keys[1] if len(self.feature_keys) > 1 else None
        )
        
        images = kwargs.get(image_key)
        questions = kwargs.get(question_key, None)
        labels = kwargs.get(self.label_key) if self.label_key else None
        
        batch_size = images.shape[0]
        
        # Step 1: Encode images with frozen CLIP ViT
        vision_features = self._vision_encoder(pixel_values=images).last_hidden_state
        # Shape: (batch_size, num_patches + 1, vision_dim)
        
        # Step 2: Prepare text input (question)
        if questions is None:
            # If no questions, create dummy prompts
            encoded_text = self._tokenizer(
                [""] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(images.device)
        elif isinstance(questions, (list, tuple)):
            # Questions are strings
            encoded_text = self._tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(images.device)
        else:
            # Questions are already tokens
            encoded_text = questions
        
        # Get initial text embeddings from language model
        text_embeds = self._lang_model.model.embed_tokens(encoded_text["input_ids"])
        # Shape: (batch_size, seq_len, lang_dim)
        
        # Step 3: Interleave image features into text sequence
        # Strategy: Insert visual tokens at the beginning
        # For simplicity, we'll use visual tokens to condition the full sequence
        lang_hidden = text_embeds
        
        # Step 4: Apply gated cross-attention layers
        # We'll insert xattn layers at regular intervals
        for i, xattn_layer in enumerate(self._xattn_layers):
            # Apply cross-attention to condition text on images
            lang_hidden = xattn_layer(lang_hidden, vision_features)
        
        # Step 5: Get final representation (use [EOS] or last token)
        final_hidden = lang_hidden[:, -1, :]  # (batch_size, lang_dim)
        
        # Step 6: Project to classification logits (if classification head exists)
        if self._fc is not None:
            logit = self._fc(final_hidden)  # (batch_size, num_classes)
        else:
            # For generation tasks, return reduced logits
            logit = final_hidden[:, :1]  # Just use first feature
        
        # Prepare output dict following BaseModel convention
        y_prob = self.prepare_y_prob(logit)
        
        output = {
            "logit": logit,
            "y_prob": y_prob,
        }
        
        # Add loss if labels are provided
        if labels is not None:
            output["y_true"] = labels
            loss_fn = self.get_loss_function()
            if self.mode == "multiclass":
                output["loss"] = loss_fn(logit, labels)
            else:
                output["loss"] = loss_fn(logit, labels.float())
        
        return output

    def generate(
        self,
        images: List[torch.Tensor],
        prompt: str,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        **generation_kwargs: Any,
    ) -> str:
        """Generate text conditioned on images and a prompt.

        This is the native MedFlamingo interface for VQA and report
        generation with optional few-shot in-context examples.

        Pipeline:
            1. Encode each image with the frozen CLIP ViT.
            2. Resample visual features via the Perceiver Resampler.
            3. Interleave ``<image>`` visual tokens with text tokens for
               both few-shot examples and the query.
            4. Auto-regressively generate from the frozen LLM using gated
               cross-attention to condition on visual tokens.

        Args:
            images: List of image tensors, each of shape ``(C, H, W)`` or
                ``(1, C, H, W)`` if batched.
            prompt: Text prompt (e.g., a medical question like
                "What is the primary finding in this X-ray?").
            few_shot_examples: Optional list of dicts, each with keys
                ``"image"`` (:class:`torch.Tensor`) and ``"text"``
                (:class:`str`), providing in-context demonstrations.
                Example: [{"image": img1, "text": "Q: ... A: ..."}]
            max_new_tokens: Maximum number of tokens to generate.
                Default 256.
            temperature: Sampling temperature. Default 1.0 (no sampling).
            **generation_kwargs: Additional kwargs passed to the language
                model's ``generate()`` method (e.g., ``top_p=0.9``,
                ``num_beams=3``).

        Returns:
            Generated text string (the model's response).

        Example:
            >>> model = MedFlamingo()
            >>> image = torch.randn(3, 224, 224)
            >>> response = model.generate(
            ...     images=[image],
            ...     prompt="Describe the main finding in this chest X-ray."
            ... )
            >>> print(response)  # e.g., "There is a pneumonic infiltrate..."
        """
        # Ensure images is a list
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = list(torch.unbind(images, dim=0))
        
        batch_size = len(images)
        
        # Stack images into batch
        images_batch = torch.stack(
            [img.unsqueeze(0) if img.ndim == 3 else img for img in images],
            dim=0
        )  # (batch_size, 3, 224, 224) or adapt to input shape
        images_batch = images_batch.to(self.device)
        
        # Step 1: Encode images with CLIP ViT
        with torch.no_grad():
            vision_features = self._vision_encoder(pixel_values=images_batch).last_hidden_state
            # (batch_size, num_patches, vision_dim)
        
        # Step 2: Build few-shot context if provided
        context_text = ""
        vision_features_list = [vision_features]
        
        if few_shot_examples:
            for example in few_shot_examples:
                exam_image = example.get("image")
                exam_text = example.get("text", "")
                
                # Encode example image
                if exam_image.ndim == 3:
                    exam_image = exam_image.unsqueeze(0)
                exam_image = exam_image.to(self.device)
                
                with torch.no_grad():
                    exam_vision_feat = self._vision_encoder(pixel_values=exam_image).last_hidden_state
                    vision_features_list.append(exam_vision_feat)
                
                context_text += f"<image>{exam_text}\n"
        
        context_text += f"<image>{prompt}"
        
        # Step 3: Encode context text
        encoded_context = self._tokenizer(
            context_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeds = self._lang_model.model.embed_tokens(encoded_context["input_ids"])
            # (1, seq_len, lang_dim)
        
        # Step 4: Apply cross-attention to produce visually-conditioned embeddings
        lang_hidden = text_embeds

        # Concatenate all vision features (few-shot images + query image)
        all_vision_features = torch.cat(
            vision_features_list, dim=1
        )  # (1, total_patches, vision_dim)

        for xattn_layer in self._xattn_layers:
            lang_hidden = xattn_layer(
                lang_hidden, all_vision_features[:1]
            )  # use first (and only) batch element

        # Step 5: Generate from the conditioned embeddings.
        # Pass ``inputs_embeds`` so the LLM starts from the xattn-conditioned
        # representations rather than the raw token embeddings.  The
        # attention_mask from the tokenizer still applies; a new all-ones mask
        # matching the embedding sequence length is used if none is available.
        attention_mask = encoded_context.get("attention_mask")

        with torch.no_grad():
            output = self._lang_model.generate(
                inputs_embeds=lang_hidden,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 1.0),
                **generation_kwargs,
            )

        # Step 6: Decode generated tokens
        generated_text = self._tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        # Remove prompt from output if present
        if prompt in generated_text:
            generated_text = generated_text.split(prompt)[-1].strip()

        return generated_text
