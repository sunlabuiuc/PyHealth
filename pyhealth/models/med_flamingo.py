"""Med-Flamingo model wrapper for generative medical VQA in PyHealth."""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MedFlamingo(BaseModel):
    """Med-Flamingo/OpenFlamingo wrapper with LoRA fine-tuning support.

    This model expects raw batch dictionaries from ``GenerativeMedicalVQA`` where
    values are Python lists produced by the default dictionary collate function.
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset],
        llama_path: str,
        hf_repo_id: str = "med-flamingo/med-flamingo",
        checkpoint_filename: str = "model.pt",
        quantization: str = "auto",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_shots: int = 0,
        sampling_strategy: str = "random",
        seed: int = 42,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        clip_vision_encoder_path: str = "ViT-L-14",
        clip_vision_encoder_pretrained: str = "openai",
        cross_attn_every_n_layers: int = 4,
        enable_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_bias: str = "none",
        gradient_checkpointing: bool = True,
        train_max_new_tokens: int = 0,
    ) -> None:
        super().__init__(dataset=dataset)

        self.llama_path = llama_path
        self.hf_repo_id = hf_repo_id
        self.checkpoint_filename = checkpoint_filename
        self.quantization = quantization.lower().strip()

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
        self.train_max_new_tokens = train_max_new_tokens

        self.num_shots = num_shots
        self.sampling_strategy = sampling_strategy.lower().strip()
        self.seed = seed

        self.cache_dir = cache_dir
        self.device_str = device
        self.clip_vision_encoder_path = clip_vision_encoder_path
        self.clip_vision_encoder_pretrained = clip_vision_encoder_pretrained
        self.cross_attn_every_n_layers = cross_attn_every_n_layers

        self.enable_lora = enable_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias
        self.gradient_checkpointing = gradient_checkpointing

        self.feature_keys = [
            "image_path",
            "question",
            "split",
            "question_id",
            "image_id",
            "dataset",
        ]
        self.label_keys = ["answer"]

        self.support_pool: List[Dict[str, str]] = []
        self._rng = random.Random(seed)

        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self._effective_quantization = "fp16"

        self._lora_configured = False
        self._lora_target_modules_resolved: List[str] = []

        self._initialize_components()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_components(self) -> None:
        llama_path = Path(self.llama_path)
        if not llama_path.exists():
            raise FileNotFoundError(
                "llama_path does not exist. Download or mount local LLaMA weights "
                f"and pass the directory path. Received: {llama_path}"
            )

        self._effective_quantization = self._resolve_quantization_mode()
        load_kwargs = self._get_language_model_load_kwargs(self._effective_quantization)

        create_model_and_transforms = self._import_open_flamingo_factory()
        self.model, self.image_processor, self.tokenizer = self._create_open_flamingo_model(
            create_model_and_transforms=create_model_and_transforms,
            load_kwargs=load_kwargs,
        )

        checkpoint_path = self._download_checkpoint()
        self._load_checkpoint(checkpoint_path)

        self._freeze_backbones()
        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        if self.enable_lora:
            self.configure_lora()

        target_device = self.device_str or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self._effective_quantization != "8bit":
            self.model.to(target_device)
        self.to(target_device)

    def _import_open_flamingo_factory(self):
        if importlib.util.find_spec("open_flamingo") is None:
            raise ImportError(
                "open_flamingo is required for MedFlamingo. Install it first, e.g. "
                "`pip install open-flamingo` and ensure compatible versions."
            )
        from open_flamingo import create_model_and_transforms

        return create_model_and_transforms

    def _resolve_quantization_mode(self) -> str:
        if self.quantization not in {"auto", "8bit", "fp16"}:
            raise ValueError("quantization must be one of: auto, 8bit, fp16")

        if self.quantization != "auto":
            return self.quantization

        if not torch.cuda.is_available():
            return "fp16"

        total_memory_gb = (
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            / (1024**3)
        )
        return "8bit" if total_memory_gb <= 16.5 else "fp16"

    def _get_language_model_load_kwargs(self, mode: str) -> Dict[str, Any]:
        if mode == "8bit":
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError(
                    "8-bit quantization requested but `bitsandbytes` is not installed. "
                    "Install with `pip install bitsandbytes` or set quantization='fp16'."
                )

            from transformers import BitsAndBytesConfig

            return {
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                "device_map": "auto",
            }

        return {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

    def _create_open_flamingo_model(self, create_model_and_transforms, load_kwargs: Dict):
        from transformers import AutoModelForCausalLM

        original_from_pretrained = AutoModelForCausalLM.from_pretrained

        def patched_from_pretrained(*args, **kwargs):
            if args and isinstance(args[0], type):
                args = args[1:]
            kwargs = dict(kwargs)
            kwargs.update(load_kwargs)
            return original_from_pretrained(*args, **kwargs)

        AutoModelForCausalLM.from_pretrained = patched_from_pretrained
        try:
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path=self.clip_vision_encoder_path,
                clip_vision_encoder_pretrained=self.clip_vision_encoder_pretrained,
                lang_encoder_path=str(self.llama_path),
                tokenizer_path=str(self.llama_path),
                cross_attn_every_n_layers=self.cross_attn_every_n_layers,
                cache_dir=self.cache_dir,
            )
        finally:
            AutoModelForCausalLM.from_pretrained = original_from_pretrained

        return model, image_processor, tokenizer

    def _download_checkpoint(self) -> str:
        if importlib.util.find_spec("huggingface_hub") is None:
            raise ImportError(
                "huggingface_hub is required to download Med-Flamingo checkpoint. "
                "Install with `pip install huggingface_hub`."
            )

        from huggingface_hub import hf_hub_download

        hf_token = os.environ.get("HF_TOKEN")
        try:
            return hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.checkpoint_filename,
                cache_dir=self.cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            raise FileNotFoundError(
                "Unable to download Med-Flamingo checkpoint. Confirm hf_repo_id, "
                "checkpoint_filename, and Hugging Face auth token if needed."
            ) from exc

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Checkpoint missing %d keys when loading MedFlamingo", len(missing))
        if unexpected:
            logger.warning(
                "Checkpoint has %d unexpected keys when loading MedFlamingo",
                len(unexpected),
            )

    def _freeze_backbones(self) -> None:
        # Freeze all base weights by default. LoRA adapters are attached on top.
        self.model.requires_grad_(False)

    def _enable_gradient_checkpointing(self) -> None:
        for module in (self.model, getattr(self.model, "lang_encoder", None)):
            if module is None:
                continue
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Prompting and batching
    # ------------------------------------------------------------------
    def set_support_pool(self, samples: Sequence[Dict[str, Any]]) -> None:
        self.support_pool = [
            {
                "image_path": str(sample["image_path"]),
                "question": str(sample["question"]),
                "answer": str(sample["answer"]),
                "question_id": str(sample.get("question_id", "")),
                "image_id": str(sample.get("image_id", "")),
                "dataset": str(sample.get("dataset", "")),
                "split": str(sample.get("split", "")),
            }
            for sample in samples
            if "image_path" in sample and "question" in sample and "answer" in sample
        ]

    def build_support_pool_from_dataset(
        self,
        sample_dataset: SampleDataset,
        split: str = "train",
    ) -> None:
        normalized_split = split.strip().lower()
        collected: List[Dict[str, Any]] = []
        for sample in sample_dataset:
            sample_split = str(sample.get("split", "")).strip().lower()
            if sample_split == normalized_split:
                collected.append(sample)
        self.set_support_pool(collected)

    def _sample_support_examples(self, question_id: str) -> List[Dict[str, str]]:
        if self.num_shots <= 0 or not self.support_pool:
            return []

        candidates = [
            sample
            for sample in self.support_pool
            if str(sample.get("question_id", "")) != str(question_id)
        ]
        if not candidates:
            return []

        count = min(self.num_shots, len(candidates))
        if self.sampling_strategy == "sequential":
            return candidates[:count]
        return self._rng.sample(candidates, k=count)

    def _build_prompt_for_generate(
        self,
        question: str,
        support_examples: Sequence[Dict[str, str]],
    ) -> str:
        chunks: List[str] = []
        for example in support_examples:
            chunks.append(
                "<image>Question: "
                f"{example['question']}\n"
                f"Answer: {example['answer']} <|endofchunk|>"
            )
        chunks.append(f"<image>Question: {question}\nAnswer:")
        return "\n".join(chunks)

    def _build_prompt_for_train(self, question: str, answer: str) -> Tuple[str, str]:
        prefix = f"<image>Question: {question}\nAnswer:"
        full_text = f"{prefix} {answer} <|endofchunk|>"
        return prefix, full_text

    def _build_prompt_and_media(
        self,
        image_path: str,
        question: str,
        question_id: str,
    ) -> tuple[str, List[str]]:
        support_examples = self._sample_support_examples(question_id)
        prompt = self._build_prompt_for_generate(question, support_examples)
        media_paths = [example["image_path"] for example in support_examples]
        media_paths.append(image_path)
        return prompt, media_paths

    def _prepare_batch_vision_x(
        self,
        batch_media_paths: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        if not batch_media_paths:
            raise ValueError("batch_media_paths cannot be empty")

        sample_tensors: List[torch.Tensor] = []
        max_media = 0
        template = None

        for media_paths in batch_media_paths:
            if not media_paths:
                raise ValueError("Each sample must include at least one image path")

            image_tensors: List[torch.Tensor] = []
            for path_str in media_paths:
                path = Path(path_str)
                if not path.exists():
                    raise FileNotFoundError(f"Image path does not exist: {path}")
                with Image.open(path) as image:
                    tensor = self.image_processor(image.convert("RGB"))
                image_tensors.append(tensor)

            sample_tensor = torch.stack(image_tensors, dim=0)
            sample_tensors.append(sample_tensor)
            max_media = max(max_media, sample_tensor.shape[0])
            if template is None:
                template = sample_tensor[0]

        assert template is not None
        padded_samples: List[torch.Tensor] = []
        for sample_tensor in sample_tensors:
            if sample_tensor.shape[0] < max_media:
                pad_count = max_media - sample_tensor.shape[0]
                pad_tensor = template.new_zeros((pad_count, *template.shape))
                sample_tensor = torch.cat([sample_tensor, pad_tensor], dim=0)
            padded_samples.append(sample_tensor)

        stacked = torch.stack(padded_samples, dim=0)
        vision_x = stacked.unsqueeze(2)
        return vision_x.to(self.device)

    def _prepare_vision_x(self, media_paths: Sequence[str]) -> torch.Tensor:
        return self._prepare_batch_vision_x([media_paths])

    def _tokenize_for_train(
        self,
        prefixes: Sequence[str],
        full_texts: Sequence[str],
        answer_masking: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized_full = self.tokenizer(
            list(full_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = input_ids.clone()

        if answer_masking:
            tokenized_prefix = self.tokenizer(
                list(prefixes),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            prefix_lengths = tokenized_prefix["attention_mask"].sum(dim=1).tolist()
            for index, prefix_len in enumerate(prefix_lengths):
                labels[index, : int(prefix_len)] = -100

        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    # ------------------------------------------------------------------
    # LoRA configuration and adapter lifecycle
    # ------------------------------------------------------------------
    def _resolve_lora_target_modules(self) -> List[str]:
        linear_module_names = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        ]

        if self.lora_target_modules:
            targets = []
            for module_name in linear_module_names:
                if any(
                    module_name == pattern or module_name.endswith(pattern)
                    for pattern in self.lora_target_modules
                ):
                    targets.append(module_name)
        else:
            targets = [
                name
                for name in linear_module_names
                if "lang_encoder.gated_cross_attn_layers" in name
                or name.startswith("perceiver")
                or ".perceiver" in name
            ]

        if not targets:
            candidates = linear_module_names[:15]
            raise ValueError(
                "No LoRA target modules were matched. Provide lora_target_modules with "
                f"valid module names. Example candidates: {candidates}"
            )

        return sorted(set(targets))

    def _log_trainable_parameter_report(self) -> Dict[str, float]:
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        trainable_ratio = trainable_params / max(total_params, 1)
        logger.info(
            "MedFlamingo params - total: %d, trainable: %d, ratio: %.6f",
            total_params,
            trainable_params,
            trainable_ratio,
        )
        return {
            "total_params": float(total_params),
            "trainable_params": float(trainable_params),
            "trainable_ratio": float(trainable_ratio),
        }

    def configure_lora(self) -> None:
        if importlib.util.find_spec("peft") is None:
            raise ImportError(
                "LoRA requested but `peft` is not installed. Install with `pip install peft`."
            )

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        target_modules = self._resolve_lora_target_modules()

        if self._effective_quantization == "8bit":
            try:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.gradient_checkpointing,
                )
            except Exception:
                logger.warning(
                    "prepare_model_for_kbit_training failed; continuing with base model."
                )

        self.model.requires_grad_(False)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
        )
        self.model = get_peft_model(self.model, lora_config)
        self._lora_configured = True
        self._lora_target_modules_resolved = target_modules
        self._log_trainable_parameter_report()

    def save_lora_adapter(self, output_dir: str) -> None:
        if importlib.util.find_spec("peft") is None:
            raise ImportError("peft is required to save LoRA adapters")

        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            raise ValueError(
                "Current model is not a PEFT model. Call configure_lora() and train first."
            )

        adapter_path = Path(output_dir)
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(adapter_path))

        metadata = {
            "llama_path": str(self.llama_path),
            "hf_repo_id": self.hf_repo_id,
            "checkpoint_filename": self.checkpoint_filename,
            "clip_vision_encoder_path": self.clip_vision_encoder_path,
            "clip_vision_encoder_pretrained": self.clip_vision_encoder_pretrained,
            "cross_attn_every_n_layers": self.cross_attn_every_n_layers,
            "quantization": self.quantization,
            "effective_quantization": self._effective_quantization,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "num_beams": self.num_beams,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_shots": self.num_shots,
            },
            "lora": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "bias": self.lora_bias,
                "target_modules": self._lora_target_modules_resolved,
            },
        }
        (adapter_path / "med_flamingo_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _validate_adapter_metadata(self, adapter_path: Path) -> None:
        metadata_file = adapter_path / "med_flamingo_config.json"
        if not metadata_file.exists():
            return

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        checks = {
            "hf_repo_id": self.hf_repo_id,
            "checkpoint_filename": self.checkpoint_filename,
            "clip_vision_encoder_path": self.clip_vision_encoder_path,
            "clip_vision_encoder_pretrained": self.clip_vision_encoder_pretrained,
            "cross_attn_every_n_layers": self.cross_attn_every_n_layers,
        }
        mismatched: List[str] = []
        for key, expected in checks.items():
            found = metadata.get(key)
            if found != expected:
                mismatched.append(f"{key}: expected={expected}, found={found}")

        if mismatched:
            mismatch_text = "; ".join(mismatched)
            raise ValueError(
                "Adapter metadata is incompatible with current base model config: "
                f"{mismatch_text}"
            )

    def load_lora_adapter(self, adapter_dir: str, is_trainable: bool = False) -> None:
        if importlib.util.find_spec("peft") is None:
            raise ImportError("peft is required to load LoRA adapters")

        from peft import PeftModel

        adapter_path = Path(adapter_dir)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                "adapter_config.json not found. This directory is not a saved LoRA adapter."
            )

        self._validate_adapter_metadata(adapter_path)

        base_model = self.model
        if isinstance(self.model, PeftModel):
            base_model = self.model.get_base_model()

        self.model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=is_trainable,
        )
        self._lora_configured = True
        self._log_trainable_parameter_report()

    # ------------------------------------------------------------------
    # Core forward paths
    # ------------------------------------------------------------------
    def _raise_oom(self, context: str, exc: RuntimeError) -> RuntimeError:
        return RuntimeError(
            f"GPU out of memory during MedFlamingo {context}. Try: quantization='8bit', "
            "reduce batch size, reduce num_shots, reduce max_new_tokens, or enable "
            "gradient accumulation."
        )

    def _extract_loss(self, outputs: Any, labels: torch.Tensor) -> torch.Tensor:
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if isinstance(outputs, dict) and outputs.get("loss") is not None:
            return outputs["loss"]

        logits = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs.get("logits")

        if logits is None:
            raise ValueError("Model output missing loss/logits for training path")

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def _generate_one(self, image_path: str, question: str, question_id: str) -> str:
        prompt, media_paths = self._build_prompt_and_media(
            image_path=image_path,
            question=question,
            question_id=question_id,
        )

        vision_x = self._prepare_vision_x(media_paths)
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        input_ids = language_inputs["input_ids"].to(self.device)
        attention_mask = language_inputs["attention_mask"].to(self.device)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.temperature == 0.0:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        completion_ids = output_ids[0][input_ids.shape[1] :]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def _forward_train_batch(
        self,
        image_paths: Sequence[str],
        questions: Sequence[str],
        answers: Sequence[str],
        question_ids: Sequence[str],
    ) -> Dict[str, Any]:
        prefixes: List[str] = []
        full_texts: List[str] = []
        for question, answer in zip(questions, answers):
            prefix, full_text = self._build_prompt_for_train(question, answer)
            prefixes.append(prefix)
            full_texts.append(full_text)

        vision_x = self._prepare_batch_vision_x([[path] for path in image_paths])
        input_ids, attention_mask, labels = self._tokenize_for_train(
            prefixes=prefixes,
            full_texts=full_texts,
            answer_masking=True,
        )

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(
            vision_x=vision_x,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = self._extract_loss(outputs, labels)

        return {
            "loss": loss,
            "generated_text": [],
            "y_true_text": list(answers),
            "question_id": list(question_ids),
        }

    def forward(self, **kwargs) -> Dict[str, Any]:
        image_paths: List[str] = [str(path) for path in kwargs.get("image_path", [])]
        questions: List[str] = [str(question) for question in kwargs.get("question", [])]
        y_true_text: List[str] = [str(answer) for answer in kwargs.get("answer", [])]
        question_ids: List[str] = [
            str(question_id)
            for question_id in kwargs.get(
                "question_id",
                [f"q_{index}" for index in range(len(questions))],
            )
        ]

        if not image_paths:
            raise ValueError("Batch must include image_path")
        if not (len(image_paths) == len(questions) == len(y_true_text) == len(question_ids)):
            raise ValueError(
                "Batch lengths for image_path/question/answer/question_id must match."
            )

        try:
            if self.training:
                return self._forward_train_batch(
                    image_paths=image_paths,
                    questions=questions,
                    answers=y_true_text,
                    question_ids=question_ids,
                )

            generated_text: List[str] = []
            for image_path, question, question_id in zip(image_paths, questions, question_ids):
                generated_text.append(
                    self._generate_one(
                        image_path=image_path,
                        question=question,
                        question_id=question_id,
                    )
                )

            return {
                "loss": torch.zeros((), device=self.device),
                "generated_text": generated_text,
                "y_true_text": y_true_text,
                "question_id": question_ids,
            }
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                context = "training" if self.training else "generation"
                raise self._raise_oom(context=context, exc=exc) from exc
            raise

    # ------------------------------------------------------------------
    # Evaluation / training utilities
    # ------------------------------------------------------------------
    def predict_generation(
        self,
        dataloader,
        metrics: Optional[Iterable[str]] = ("exact_match",),
    ) -> Dict[str, object]:
        from pyhealth.metrics import generative_metrics_fn

        self.eval()
        y_true: List[str] = []
        y_pred: List[str] = []
        question_ids: List[str] = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self(**batch)
                y_true.extend(outputs["y_true_text"])
                y_pred.extend(outputs["generated_text"])
                question_ids.extend(outputs["question_id"])

        scores = generative_metrics_fn(y_true=y_true, y_pred=y_pred, metrics=metrics)
        predictions = [
            {
                "question_id": question_id,
                "y_true_text": true_text,
                "generated_text": pred_text,
            }
            for question_id, true_text, pred_text in zip(question_ids, y_true, y_pred)
        ]
        return {
            "metrics": scores,
            "predictions": predictions,
        }

    def evaluate_generation(
        self,
        dataloader,
        metrics: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        return self.predict_generation(dataloader=dataloader, metrics=metrics)

    def _current_trainable_params(self) -> List[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def fit_lora(
        self,
        train_dataloader,
        val_dataloader=None,
        epochs: int = 1,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.03,
        eval_every_n_steps: int = 0,
        metrics: Iterable[str] = ("exact_match",),
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        metric_names = list(metrics)
        if not self._lora_configured:
            self.configure_lora()

        trainable_params = self._current_trainable_params()
        if not trainable_params:
            raise ValueError("No trainable parameters found. Configure LoRA before training.")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        steps_per_epoch = len(train_dataloader)
        if steps_per_epoch == 0:
            raise ValueError("train_dataloader is empty")

        total_optimizer_steps = epochs * math.ceil(steps_per_epoch / grad_accum_steps)
        warmup_steps = int(total_optimizer_steps * warmup_ratio)
        scheduler = None
        if total_optimizer_steps > 0 and importlib.util.find_spec("transformers") is not None:
            from transformers import get_linear_schedule_with_warmup

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_optimizer_steps,
            )

        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        history: List[Dict[str, Any]] = []
        global_step = 0
        best_metric_name = metric_names[0] if metric_names else "exact_match"
        best_metric_value = -float("inf") if val_dataloader is not None else float("inf")

        use_amp = torch.cuda.is_available() and self.device.type == "cuda"

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0

            for step, batch in enumerate(train_dataloader, start=1):
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )
                with autocast_ctx:
                    output = self(**batch)
                    loss = output["loss"]
                    scaled_loss = loss / grad_accum_steps

                scaled_loss.backward()
                epoch_loss += float(loss.detach().cpu().item())

                should_step = step % grad_accum_steps == 0 or step == steps_per_epoch
                if should_step:
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1

                    if (
                        val_dataloader is not None
                        and eval_every_n_steps > 0
                        and global_step % eval_every_n_steps == 0
                    ):
                        val_result = self.predict_generation(
                            dataloader=val_dataloader,
                            metrics=metric_names,
                        )
                        val_score = float(val_result["metrics"].get(best_metric_name, 0.0))
                        if val_score > best_metric_value:
                            best_metric_value = val_score
                            if output_path is not None:
                                self.save_lora_adapter(str(output_path / "best_adapter"))

            epoch_train_loss = epoch_loss / steps_per_epoch
            epoch_record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": epoch_train_loss,
            }

            if val_dataloader is not None:
                val_result = self.predict_generation(
                    dataloader=val_dataloader,
                    metrics=metric_names,
                )
                val_metrics = {
                    f"val_{key}": value for key, value in val_result["metrics"].items()
                }
                epoch_record.update(val_metrics)
                candidate_score = float(val_result["metrics"].get(best_metric_name, 0.0))
                if candidate_score > best_metric_value:
                    best_metric_value = candidate_score
                    if output_path is not None:
                        self.save_lora_adapter(str(output_path / "best_adapter"))
            else:
                if epoch_train_loss < best_metric_value:
                    best_metric_value = epoch_train_loss
                    if output_path is not None:
                        self.save_lora_adapter(str(output_path / "best_adapter"))

            history.append(epoch_record)
            logger.info("LoRA epoch %d summary: %s", epoch + 1, epoch_record)

            if output_path is not None:
                self.save_lora_adapter(str(output_path / "last_adapter"))
                (output_path / "metrics_history.json").write_text(
                    json.dumps(history, indent=2),
                    encoding="utf-8",
                )

        result = {
            "best_metric": float(best_metric_value),
            "epochs": float(epochs),
            "global_steps": float(global_step),
        }
        if output_path is not None:
            (output_path / "fit_summary.json").write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
        return result
