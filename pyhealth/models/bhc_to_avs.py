# Author: Charan Williams
# NetID: charanw2
# Description: Converts clinical brief hospital course (BHC) data to after visit summaries using a fine-tuned Mistral 7B model.

from typing import Dict, Any
from dataclasses import dataclass, field
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
    base_model_id: str = field(default="mistralai/Mistral-7B-Instruct")
    """HuggingFace repo containing the base Mistral 7B model."""

    adapter_model_id: str = field(default="williach31/mistral-7b-bhc-to-avs-lora")
    """HuggingFace repo containing only LoRA adapter weights."""

    def _get_pipeline(self):
        """Create and cache the text-generation pipeline."""
        if not hasattr(self, "_pipeline"):
            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # Load LoRA adapter
            model = PeftModelForCausalLM.from_pretrained(
                base,
                self.adapter_model_id,
                torch_dtype=torch.bfloat16
            )

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

            # Create HF pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                model_kwargs={"torch_dtype": torch.bfloat16}
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

        prompt = _SYSTEM_PROMPT + _PROMPT.format(bhc=bhc_text)

        pipe = self._get_pipeline()
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.0,
            eos_token_id=[pipe.tokenizer.eos_token_id],
            pad_token_id=pipe.tokenizer.eos_token_id,
        )

        # Output is a single text string
        return outputs[0]["generated_text"].strip()