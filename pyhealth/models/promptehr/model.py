"""PromptEHR: BART-based generative model for synthetic EHR generation.

This module provides the main PromptEHR model that combines demographic-conditioned
prompts with BART encoder-decoder architecture for realistic patient record generation.

Ported from pehr_scratch/prompt_bart_model.py (lines 16-276, excluding auxiliary losses).
"""

import os
import random
import sys
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Temporarily hide torchvision so transformers skips the
# image_utils → torchvision → PIL import chain (which fails in Colab
# due to mixed-version Pillow files). PromptEHR only needs BART,
# not any vision functionality from transformers.
_tv = sys.modules.pop("torchvision", None)
try:
    from transformers import BartConfig, BartForConditionalGeneration
    from transformers.modeling_outputs import Seq2SeqLMOutput
finally:
    if _tv is not None:
        sys.modules["torchvision"] = _tv

del _tv

from pyhealth.models import BaseModel
from .conditional_prompt import ConditionalPromptEncoder
from .bart_encoder import PromptBartEncoder
from .bart_decoder import PromptBartDecoder


class PromptBartModel(BartForConditionalGeneration):
    """BART model with demographic prompt conditioning for EHR generation.

    Extends HuggingFace's BartForConditionalGeneration with:
    1. Dual prompt encoders (separate for encoder/decoder)
    2. Demographic conditioning via learned prompt vectors
    3. Label smoothing for diverse generation

    This is the core generative model WITHOUT auxiliary losses (those caused
    mode collapse and are excluded per implementation decision D003).

    Args:
        config: BART configuration from transformers
        n_num_features: Number of continuous features (1 for age)
        cat_cardinalities: Category counts for categorical features ([2] for gender M/F)
        d_hidden: Intermediate reparameterization dimension (default: 128)
        prompt_length: Number of prompt vectors per feature (default: 1)

    Example:
        >>> from transformers import BartConfig
        >>> config = BartConfig.from_pretrained("facebook/bart-base")
        >>> model = PromptBartModel(
        ...     config,
        ...     n_num_features=1,           # age
        ...     cat_cardinalities=[2],      # gender (M/F)
        ...     d_hidden=128,
        ...     prompt_length=1
        ... )
        >>> # Forward pass with demographics
        >>> age = torch.randn(16, 1)        # [batch, 1]
        >>> gender = torch.randint(0, 2, (16, 1))  # [batch, 1]
        >>> input_ids = torch.randint(0, 1000, (16, 100))
        >>> labels = torch.randint(0, 1000, (16, 50))
        >>> output = model(
        ...     input_ids=input_ids,
        ...     labels=labels,
        ...     x_num=age,
        ...     x_cat=gender
        ... )
        >>> loss = output.loss
    """

    def __init__(
        self,
        config: BartConfig,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list] = None,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize PromptBART model with dual prompt conditioning.

        Args:
            config: BART configuration
            n_num_features: Number of continuous features (e.g., 1 for age)
            cat_cardinalities: Category counts for categorical features [n_genders]
            d_hidden: Intermediate reparameterization dimension (default: 128)
            prompt_length: Number of prompt vectors per feature (default: 1)
        """
        super().__init__(config)

        # Replace encoder and decoder with prompt-aware versions
        self.model.encoder = PromptBartEncoder(config, self.model.shared)
        self.model.decoder = PromptBartDecoder(config, self.model.shared)

        # Add SEPARATE conditional prompt encoders for encoder and decoder
        # This provides stronger demographic conditioning than shared prompts (dual injection)
        if n_num_features is not None or cat_cardinalities is not None:
            # Encoder prompt encoder
            self.encoder_prompt_encoder = ConditionalPromptEncoder(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            # Decoder prompt encoder (separate parameters for dual injection)
            self.decoder_prompt_encoder = ConditionalPromptEncoder(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            self.num_prompts = self.encoder_prompt_encoder.get_num_prompts()
        else:
            self.encoder_prompt_encoder = None
            self.decoder_prompt_encoder = None
            self.num_prompts = 0

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        """Forward pass with demographic conditioning.

        Args:
            input_ids: [batch, seq_len] encoder input token IDs
            attention_mask: [batch, seq_len] encoder attention mask
            decoder_input_ids: [batch, tgt_len] decoder input token IDs
            decoder_attention_mask: [batch, tgt_len] decoder attention mask
            labels: [batch, tgt_len] target labels for loss computation
            x_num: [batch, n_num_features] continuous demographic features (e.g., age)
            x_cat: [batch, n_cat_features] categorical demographic features (e.g., gender)
            Other args: Standard BART arguments

        Returns:
            Seq2SeqLMOutput with:
                - loss: Cross-entropy loss with label smoothing=0.1
                - logits: [batch, tgt_len, vocab_size] prediction logits
                - past_key_values: Cached key-value states (if use_cache=True)
                - decoder_hidden_states: Decoder layer outputs (if output_hidden_states=True)
                - encoder_last_hidden_state: Final encoder output
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode demographic prompts separately for encoder and decoder
        # Only prepend prompts on first step (when no cache exists)
        encoder_prompt_embeds = None
        decoder_prompt_embeds = None
        if (x_num is not None or x_cat is not None) and past_key_values is None:
            if self.encoder_prompt_encoder is not None:
                encoder_prompt_embeds = self.encoder_prompt_encoder(x_num=x_num, x_cat=x_cat)
            if self.decoder_prompt_encoder is not None:
                decoder_prompt_embeds = self.decoder_prompt_encoder(x_num=x_num, x_cat=x_cat)

        # Prepare decoder input IDs (shift labels right for teacher forcing)
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Encoder forward pass (with encoder prompts)
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                inputs_prompt_embeds=encoder_prompt_embeds,  # Encoder-specific prompts
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Extend encoder attention mask for prompts
        encoder_attention_mask = attention_mask
        if encoder_prompt_embeds is not None and attention_mask is not None:
            batch_size, n_prompts = encoder_prompt_embeds.shape[:2]
            prompt_mask = torch.ones(batch_size, n_prompts, dtype=attention_mask.dtype, device=attention_mask.device)
            encoder_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Decoder forward pass (with decoder prompts)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            inputs_prompt_embeds=decoder_prompt_embeds,  # Decoder-specific prompts
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Language modeling head
        lm_logits = self.lm_head(decoder_outputs[0])

        # If decoder prompts were prepended, slice them off before loss computation
        if decoder_prompt_embeds is not None and labels is not None:
            # decoder_outputs[0] shape: [batch, n_prompts + seq_len, hidden_dim]
            # We only want logits for the actual sequence positions
            n_prompts = decoder_prompt_embeds.shape[1]
            lm_logits = lm_logits[:, n_prompts:, :]  # Remove prompt positions

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Label smoothing = 0.1 to prevent overconfidence and encourage diversity
            # Softens target distributions: 90% on correct token, 10% distributed to alternatives
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **kwargs
    ):
        """Prepare inputs for autoregressive generation.

        Args:
            decoder_input_ids: [batch, cur_len] current decoder input IDs
            past_key_values: Cached key-value states from previous steps
            x_num: [batch, n_num_features] continuous demographics (passed through)
            x_cat: [batch, n_cat_features] categorical demographics (passed through)
            Other args: Standard BART generation arguments

        Returns:
            Dictionary of inputs for next generation step
        """
        # Cut decoder_input_ids if past is used (only need last token)
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "x_num": x_num,  # Pass demographics through generation
            "x_cat": x_cat,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size=1,
        is_encoder_decoder=True,
        attention_mask=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **model_kwargs,
    ):
        """Expand inputs for beam search or multiple samples.

        Args:
            input_ids: [batch, seq_len] input token IDs
            expand_size: Number of beams/samples per input
            x_num: [batch, n_num_features] continuous demographics
            x_cat: [batch, n_cat_features] categorical demographics
            Other args: Standard expansion arguments

        Returns:
            Expanded input_ids and model_kwargs
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if encoder_outputs is not None:
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        # Expand demographics for beam search
        if x_num is not None:
            model_kwargs["x_num"] = x_num.index_select(0, expanded_return_idx)

        if x_cat is not None:
            model_kwargs["x_cat"] = x_cat.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """Shift input ids one token to the right for teacher forcing.

    Args:
        input_ids: [batch, seq_len] target token IDs
        pad_token_id: ID for padding token
        decoder_start_token_id: ID for decoder start token (BOS)

    Returns:
        [batch, seq_len] shifted token IDs with BOS prepended
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id must be defined for sequence generation")

    # Replace -100 in labels with pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class _PromptEHRVocab:
    """Internal vocabulary bridging NestedSequenceProcessor indices to BART token IDs.

    Token layout (7 special tokens + N diagnosis codes):
        0 = <pad>   (BartConfig.pad_token_id)
        1 = <bos>   (BartConfig.bos_token_id / decoder_start_token_id)
        2 = <eos>   (BartConfig.eos_token_id)
        3 = <unk>
        4 = <v>     (visit start)
        5 = </v>    (visit end)
        6 = <END>   (sequence terminator)
        7+ = diagnosis codes

    NestedSequenceProcessor uses pad=0, unk=1, codes=2+.
    Mapping: processor_idx i -> BART token i + 5  (for i >= 2).
    Total BART vocab size = processor.vocab_size() + 5.

    Args:
        code_vocab (dict): Mapping of code string to processor index, as
            returned by ``NestedSequenceProcessor.code_vocab``. Must include
            ``"<pad>"`` -> 0 and ``"<unk>"`` -> 1.

    Examples:
        >>> vocab = _PromptEHRVocab({"<pad>": 0, "<unk>": 1, "428": 2, "410": 3})
        >>> isinstance(vocab, _PromptEHRVocab)
        True
        >>> vocab.total_size
        9
    """

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    VISIT_START = 4
    VISIT_END = 5
    SEQ_END = 6
    CODE_OFFSET = 7

    def __init__(self, code_vocab: dict):
        """Build vocab from NestedSequenceProcessor.code_vocab dict."""
        self._bart_to_code: Dict[int, str] = {}
        for code, pid in code_vocab.items():
            if pid >= 2:  # skip <pad> and <unk>
                self._bart_to_code[pid + 5] = code
        self.total_size = len(code_vocab) + 5  # 7 special - 2 reused + N codes

    def encode_visits(self, visits_tensor: torch.Tensor) -> List[int]:
        """Encode a processed [n_visits, max_codes] LongTensor to a token ID list.

        Args:
            visits_tensor (torch.Tensor): LongTensor of shape
                ``(n_visits, max_codes_per_visit)`` from NestedSequenceProcessor.
                Values 0 = pad, 1 = unk, 2+ = code index.

        Returns:
            list of int: Token IDs in format
                ``[<v>, code, ..., </v>, <v>, ..., </v>, <END>]``.
        """
        tokens = []
        for visit in visits_tensor:
            codes_in_visit = [
                int(c.item()) + 5  # processor idx 2+ → BART idx 7+
                for c in visit
                if c.item() >= 2   # skip pad and unk
            ]
            if codes_in_visit:
                tokens.append(self.VISIT_START)
                tokens.extend(codes_in_visit)
                tokens.append(self.VISIT_END)
        tokens.append(self.SEQ_END)
        return tokens

    def decode_tokens(self, token_ids: List[int]) -> List[List[str]]:
        """Decode a generated token ID list back to visit structure.

        Args:
            token_ids (list of int): Raw generated token IDs from BART.

        Returns:
            list of list of str: Decoded diagnosis code strings per visit.
        """
        visits: List[List[str]] = []
        current_visit: List[str] = []
        in_visit = False
        for tid in token_ids:
            if tid in (self.PAD, self.BOS, self.EOS):
                continue  # skip framing tokens (BOS is first in generate output)
            if tid == self.SEQ_END:
                break
            if tid == self.VISIT_START:
                in_visit = True
                current_visit = []
            elif tid == self.VISIT_END:
                if in_visit:
                    visits.append(current_visit)
                    in_visit = False
            elif in_visit and tid >= self.CODE_OFFSET:
                code = self._bart_to_code.get(tid)
                if code:
                    current_visit.append(code)
        if in_visit and current_visit:
            visits.append(current_visit)
        return visits


def _promptehr_collate_fn(batch):
    """Collate PromptEHR training samples, padding token sequences in a batch.

    Pads ``input_ids`` and ``labels`` to the longest sequence in the batch using
    ``pad_sequence``. Builds the attention mask from padded positions.

    Args:
        batch (list of dict): Each dict has ``"input_ids"``, ``"labels"``,
            ``"x_num"``, and ``"x_cat"`` tensors.

    Returns:
        dict: Batched tensors ready for ``PromptBartModel.forward()``.
    """
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=_PromptEHRVocab.PAD,
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = (input_ids != _PromptEHRVocab.PAD).long()
    x_num = torch.cat([item["x_num"] for item in batch], dim=0)
    x_cat = torch.cat([item["x_cat"] for item in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "x_num": x_num,
        "x_cat": x_cat,
    }


class PromptEHR(BaseModel):
    """PromptEHR: demographic-conditioned BART model for synthetic EHR generation.

    Wraps ``PromptBartModel`` (HuggingFace BART with dual prompt conditioning)
    in a PyHealth ``BaseModel`` interface. Training is handled by a HuggingFace
    ``Trainer`` loop; generation is autoregressive token-by-token decoding.

    Demographics (age as continuous, gender as categorical) are injected via
    learned prompt vectors prepended to both encoder and decoder hidden states.

    Args:
        dataset (SampleDataset): PyHealth sample dataset produced by
            ``set_task(promptehr_generation_mimic3_fn)``. Must have
            ``input_processors["visits"]`` (NestedSequenceProcessor).
        n_num_features (int): Continuous demographic features (1 for age).
            Default: 1.
        cat_cardinalities (list of int): Category counts per categorical
            feature ([2] for binary gender M/F). Default: [2].
        d_hidden (int): Reparameterization dimension for prompt encoder.
            Default: 128.
        prompt_length (int): Number of prompt vectors per feature. Default: 1.
        bart_config_name (str): Pretrained BART config to use.
            Default: ``"facebook/bart-base"``.
        epochs (int): Training epochs. Default: 20.
        batch_size (int): Training batch size. Default: 16.
        lr (float): AdamW learning rate. Default: 1e-5.
        warmup_steps (int): Linear warmup steps. Default: 1000.
        max_seq_length (int): Maximum token sequence length. Default: 512.
        save_dir (str): Directory for checkpoints. Default: ``"./save/"``.

    Examples:
        >>> from pyhealth.datasets.sample_dataset import InMemorySampleDataset
        >>> samples = [
        ...     {"patient_id": "p1", "visits": [["428", "427"], ["410"]], "age": 65.0, "gender": 0},
        ...     {"patient_id": "p2", "visits": [["250"], ["401", "272"]], "age": 52.0, "gender": 1},
        ... ]
        >>> dataset = InMemorySampleDataset(
        ...     samples=samples,
        ...     input_schema={"visits": "nested_sequence"},
        ...     output_schema={},
        ... )
        >>> model = PromptEHR(dataset, d_hidden=32, prompt_length=1)
        >>> isinstance(model, PromptEHR)
        True
    """

    def __init__(
        self,
        dataset,
        n_num_features: int = 1,
        cat_cardinalities: Optional[list] = None,
        d_hidden: int = 128,
        prompt_length: int = 1,
        bart_config_name: "Union[str, BartConfig]" = "facebook/bart-base",
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-5,
        warmup_steps: int = 1000,
        max_seq_length: int = 512,
        save_dir: str = "./save/",
    ):
        """Initialize PromptEHR with vocab derived from the dataset processor."""
        super().__init__(dataset)

        self.mode = None  # skip discriminative evaluation
        self.save_dir = save_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        self._demo_pool: List[tuple] = []  # (age, gender) pairs from training data

        if cat_cardinalities is None:
            cat_cardinalities = [2]

        # Derive vocab from the dataset's NestedSequenceProcessor
        visits_processor = dataset.input_processors["visits"]
        self._vocab = _PromptEHRVocab(visits_processor.code_vocab)
        bart_vocab_size = self._vocab.total_size

        # Configure BART with our custom vocab and special token IDs
        if isinstance(bart_config_name, str):
            bart_config = BartConfig.from_pretrained(bart_config_name)
        else:
            # Accept a BartConfig object directly (useful for tiny test models)
            bart_config = bart_config_name
        bart_config.vocab_size = bart_vocab_size
        bart_config.pad_token_id = _PromptEHRVocab.PAD
        bart_config.bos_token_id = _PromptEHRVocab.BOS
        bart_config.eos_token_id = _PromptEHRVocab.EOS
        bart_config.decoder_start_token_id = _PromptEHRVocab.BOS
        bart_config.forced_eos_token_id = _PromptEHRVocab.SEQ_END
        bart_config.dropout = 0.3
        bart_config.attention_dropout = 0.3
        bart_config.activation_dropout = 0.3

        self.bart_model = PromptBartModel(
            config=bart_config,
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_hidden=d_hidden,
            prompt_length=prompt_length,
        )

    def forward(self, **kwargs) -> Dict:
        """Not implemented — PromptEHR is a generative model without a discriminative forward.

        Raises:
            NotImplementedError: Always. Use ``train_model`` and
                ``synthesize_dataset`` instead.
        """
        raise NotImplementedError(
            "PromptEHR is a generative model. Use train_model() and synthesize_dataset()."
        )

    def train_model(self, train_dataset, val_dataset=None) -> None:
        """Train PromptEHR using a HuggingFace Trainer loop.

        Converts PyHealth SampleDataset samples to BART token sequences and
        trains with HuggingFace ``Trainer``. Demographics (age, gender) are
        passed as ``x_num`` / ``x_cat`` via a custom data collator.

        Named ``train_model`` (not ``train``) to avoid shadowing
        ``nn.Module.train()``.

        Args:
            train_dataset (SampleDataset): Training set with ``"visits"``,
                ``"age"``, and ``"gender"`` fields.
            val_dataset (SampleDataset, optional): Validation set for loss
                monitoring. Default: None.
        """
        from torch.utils.data import Dataset as TorchDataset
        from transformers import Trainer, TrainingArguments

        vocab = self._vocab
        max_len = self.max_seq_length

        class _EHRDataset(TorchDataset):
            def __init__(self, samples):
                self._samples = list(samples)

            def __len__(self):
                return len(self._samples)

            def __getitem__(self, idx):
                s = self._samples[idx]
                tokens = vocab.encode_visits(s["visits"])
                if len(tokens) > max_len:
                    tokens = tokens[:max_len - 1] + [vocab.SEQ_END]
                age = float(s.get("age", 60.0))
                gender = int(s.get("gender", 0))
                return {
                    "input_ids": torch.tensor(tokens, dtype=torch.long),
                    "labels": torch.tensor(tokens, dtype=torch.long),
                    "x_num": torch.tensor([[age]], dtype=torch.float32),
                    "x_cat": torch.tensor([[gender]], dtype=torch.long),
                }

        train_samples = list(train_dataset)
        # Store demographics pool for synthesize_dataset sampling
        self._demo_pool = [
            (float(s.get("age", 60.0)), int(s.get("gender", 0)))
            for s in train_samples
        ]

        os.makedirs(self.save_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.save_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            save_strategy="epoch",
            logging_steps=50,
            remove_unused_columns=False,  # essential: keeps x_num/x_cat
            use_cpu=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.bart_model,
            args=training_args,
            train_dataset=_EHRDataset(train_samples),
            eval_dataset=_EHRDataset(list(val_dataset)) if val_dataset else None,
            data_collator=_promptehr_collate_fn,
        )
        trainer.train()

        self.save_model(os.path.join(self.save_dir, "checkpoint.pt"))

    def synthesize_dataset(
        self, num_samples: int, random_sampling: bool = True
    ) -> List[Dict]:
        """Generate a synthetic patient dataset.

        Samples demographics from the training data distribution (if available)
        and generates autoregressive token sequences via BART. Each sequence is
        decoded back to a nested list of diagnosis code strings.

        Args:
            num_samples (int): Number of synthetic patients to generate.
            random_sampling (bool): If True, uses multinomial sampling with
                ``temperature=0.7, top_p=0.95``. If False, uses greedy decoding.
                Default: True.

        Returns:
            list of dict: One record per synthetic patient. Each dict has:
                ``"patient_id"`` (str): unique identifier, e.g. ``"synthetic_0"``.
                ``"visits"`` (list of list of str): decoded code strings per visit.
        """
        self.bart_model.eval()
        # Use bart_model's device, not self.device — HuggingFace Trainer
        # moves bart_model to GPU but doesn't move the parent PromptEHR module.
        device = next(self.bart_model.parameters()).device

        results = []
        with torch.no_grad():
            for i in range(num_samples):
                # Sample demographics from training distribution (or defaults)
                if self._demo_pool:
                    age, gender = self._demo_pool[
                        random.randrange(len(self._demo_pool))
                    ]
                else:
                    age, gender = 60.0, 0

                x_num = torch.tensor([[age]], dtype=torch.float32, device=device)
                x_cat = torch.tensor([[gender]], dtype=torch.long, device=device)

                # PAD token as minimal encoder input; prompts carry the signal
                encoder_input = torch.tensor(
                    [[_PromptEHRVocab.PAD]], dtype=torch.long, device=device
                )

                output_ids = self.bart_model.generate(
                    input_ids=encoder_input,
                    attention_mask=torch.ones_like(encoder_input),
                    x_num=x_num,
                    x_cat=x_cat,
                    max_length=self.max_seq_length,
                    num_beams=1,
                    do_sample=random_sampling,
                    temperature=0.7 if random_sampling else 1.0,
                    top_p=0.95 if random_sampling else 1.0,
                    pad_token_id=_PromptEHRVocab.PAD,
                    eos_token_id=_PromptEHRVocab.SEQ_END,
                    bos_token_id=_PromptEHRVocab.BOS,
                )

                visits = self._vocab.decode_tokens(output_ids[0].tolist())
                results.append({
                    "patient_id": f"synthetic_{i}",
                    "visits": visits,
                })

        return results

    def save_model(self, path: str) -> None:
        """Save model weights and vocab to a checkpoint file.

        Args:
            path (str): Destination file path (e.g. ``"./save/checkpoint.pt"``).

        Examples:
            >>> import tempfile, os
            >>> tmpdir = tempfile.mkdtemp()
            >>> model.save_model(os.path.join(tmpdir, "ckpt.pt"))
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.bart_model.state_dict(),
                "vocab": self._vocab,
                "bart_config": self.bart_model.config,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load model weights from a checkpoint saved by ``save_model``.

        Args:
            path (str): Path to checkpoint file produced by ``save_model``.

        Examples:
            >>> model.load_model("./save/checkpoint.pt")
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.bart_model.load_state_dict(checkpoint["model"])
        if "vocab" in checkpoint:
            self._vocab = checkpoint["vocab"]
