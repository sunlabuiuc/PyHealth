"""
BHC to AVS Model

Generates patient-friendly After Visit Summaries (AVS) from Brief Hospital Course (BHC)
notes using a fine-tuned Mistral 7B model with a LoRA adapter.

This model requires access to a gated Hugging Face repository. Provide credentials
using one of the following methods:

1. Set an environment variable:
   export HF_TOKEN="hf_..."

2. Pass the token explicitly when creating the model:
   model = BHCToAVS(hf_token="hf_...")

If no token is provided and the repository is gated, a RuntimeError will be raised.
"""

# Author: Charan Williams
# NetID: charanw2


from dataclasses import dataclass, field
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM
from pyhealth.models.base_model import BaseModel

# System prompt used during inference
_SYSTEM_PROMPT = (
    "You are a clinical summarization model. Produce accurate, patient-friendly summaries "
    "using only information from the doctor's note. Do not add new details.\n\n"
)

# Prompt used during fine-tuning
_PROMPT = (
    "Summarize for the patient what happened during the hospital stay based on this doctor's note:\n"
    "{bhc}\n\n"
    "Summary for the patient:\n"
)


@dataclass
class BHCToAVS(BaseModel):
    """
    BHCToAVS is a model class designed to generate After-Visit Summaries (AVS) from
    Brief Hospital Course (BHC) notes using a pre-trained base model and a LoRA adapter.
    Attributes
    base_model_id : str
        The HuggingFace repository identifier for the base Mistral 7B model.
    adapter_model_id : str
        The HuggingFace repository identifier for the LoRA adapter weights.
    Methods
    _get_pipeline():
        Creates and caches a HuggingFace text-generation pipeline using the base model
        and LoRA adapter.
    predict(bhc_text: str) -> str:
        Generates a patient-friendly After-Visit Summary (AVS) from a given Brief
        Hospital Course (BHC) note.
    """

    base_model_id: str = field(default="mistralai/Mistral-7B-Instruct-v0.3")
    """HuggingFace repo containing the base Mistral 7B model."""

    adapter_model_id: str = field(default="williach31/mistral-7b-bhc-to-avs-lora")
    """HuggingFace repo containing only LoRA adapter weights."""

    hf_token: str | None = None

    def _resolve_token(self):
        return self.hf_token or os.getenv("HF_TOKEN")

    def _get_pipeline(self):
        """Create and cache the text-generation pipeline."""
        if not hasattr(self, "_pipeline"):
            # Resolve HuggingFace token
            token = self._resolve_token()

            # Throw RuntimeError if token is not found
            if token is None:
                raise RuntimeError(
                    "Hugging Face token not found. This model requires access to a gated repository.\n\n"
                    "Set the HF_TOKEN environment variable or pass hf_token=... when initializing BHCToAVS.\n\n"
                    "Example:\n"
                    "  export HF_TOKEN='hf_...'\n"
                    "  model = BHCToAVS()\n"
                )


            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
            )

            # Load LoRA adapter
            model = PeftModelForCausalLM.from_pretrained(
                base,
                self.adapter_model_id,
                torch_dtype=torch.bfloat16,
                token=token,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, token=token)

            # Create HF pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )

        return self._pipeline

    def predict(self, bhc_text: str) -> str:
        """
        Generate an After-Visit Summary (AVS) from a Brief Hospital Course (BHC) note.

        Parameters
        ----------
        bhc_text : str
            Raw BHC text.

        Returns
        -------
        str
            Patient-friendly summary.
        """

        # Validate input to provide clear error messages and avoid unexpected failures.
        if bhc_text is None:
            raise ValueError("bhc_text must not be None.")
        if not isinstance(bhc_text, str):
            raise TypeError(
                f"bhc_text must be a string, got {type(bhc_text).__name__}."
            )
        if not bhc_text.strip():
            raise ValueError("bhc_text must be a non-empty string.")
        prompt = _SYSTEM_PROMPT + _PROMPT.format(bhc=bhc_text)

        pipe = self._get_pipeline()
        eos_id = pipe.tokenizer.eos_token_id
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.0,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            return_full_text=False,
        )

        # Output is a single text string
        return outputs[0]["generated_text"].strip()
