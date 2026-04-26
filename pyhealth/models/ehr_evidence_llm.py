"""Zero-shot LLM pipeline for EHR evidence retrieval.
Contributor: Abhisek Sinha (abhisek5@illinois.edu)
Paper: `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`
Implements the two-step prompting pipeline from:
    Ahsan et al. (2024) "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." CHIL 2024, PMLR 248:489-505.
    arXiv: 2309.04550

The pipeline:
1. Classification prompt  → yes/no + token-level confidence score.
2. Summarisation prompt   → free-text evidence (only when step 1 is "yes").

A Clinical-BERT dense-retrieval baseline is also available via
``use_cbert_baseline=True``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (based on the paper's two-step zero-shot design)
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = """\
You are a clinical assistant helping a radiologist review patient records.

Patient clinical notes:
{notes}

Question: Based solely on the notes above, does this patient currently have \
or show signs of risk for {query_diagnosis}?

Answer with YES or NO only."""

_SUMMARISE_PROMPT = """\
You are a clinical assistant helping a radiologist review patient records.

Patient clinical notes:
{notes}

The patient has been assessed as having or being at risk for {query_diagnosis}.

Task: Summarise the supporting evidence from the notes above that indicates \
the patient has or is at risk for {query_diagnosis}.
- Quote directly from the notes where possible.
- Do NOT include any information not present in the notes above.
- Be concise (2-4 sentences)."""


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------
class ZeroShotEvidenceLLM(BaseModel):
    """Zero-shot LLM pipeline for retrieving evidence from EHR notes.

    Implements the two-step zero-shot prompting strategy from Ahsan et al.
    (2024). Given a patient's concatenated clinical notes and a query
    diagnosis, the model:

    1. Runs a *classification* prompt to determine whether the patient has or
       is at risk of the condition (yes/no).
    2. If the answer is ``"yes"``, runs a *summarisation* prompt to extract
       and summarise supporting evidence from the notes.

    A confidence score is computed from the normalised token-level probability
    of the ``"yes"`` response (Step 1), which the paper shows achieves
    AUC > 0.9 for predicting hallucination risk.

    A dense-retrieval baseline based on ``emilyalsentzer/Bio_ClinicalBERT``
    (sentence cosine similarity) is available by setting
    ``use_cbert_baseline=True``.

    Args:
        dataset: PyHealth ``SampleDataset``.  Pass ``None`` when using the
            model standalone without the PyHealth training loop.
        model_name (str): HuggingFace model ID for the LLM backbone.
            Supported architectures: encoder-decoder (e.g. Flan-T5) and
            decoder-only (e.g. Mistral-Instruct).
            Defaults to ``"google/flan-t5-xxl"``.
        max_input_tokens (int): Maximum tokens to include from the patient
            notes before truncation. Defaults to ``2048``.
        max_new_tokens (int): Maximum tokens to generate in the summarisation
            step. Defaults to ``256``.
        device (Optional[str]): Device string (``"cuda"``, ``"cpu"``, etc.).
            Defaults to ``"cuda"`` if available, else ``"cpu"``.
        use_cbert_baseline (bool): When ``True`` the model runs the
            Clinical-BERT dense-retrieval baseline instead of the LLM
            pipeline. Defaults to ``False``.
        cbert_model_name (str): HuggingFace model ID for the CBERT baseline.
            Defaults to ``"emilyalsentzer/Bio_ClinicalBERT"``.

    Examples:
        Standalone usage (no dataset required)::

            >>> from pyhealth.models import ZeroShotEvidenceLLM
            >>> model = ZeroShotEvidenceLLM(dataset=None)
            >>> result = model.predict(
            ...     notes="Patient presents with irregular heartbeat ...",
            ...     query_diagnosis="atrial fibrillation",
            ... )
            >>> print(result)
            {'has_condition': True, 'evidence': '...', 'confidence': 0.91}

        Pipeline usage with a dataset::

            >>> from pyhealth.datasets import MIMIC3NoteDataset
            >>> from pyhealth.tasks import EHREvidenceRetrievalTask
            >>> from pyhealth.models import ZeroShotEvidenceLLM
            >>> dataset = MIMIC3NoteDataset(root="/path/to/mimic-iii/1.4")
            >>> task = EHREvidenceRetrievalTask(
            ...     query_diagnosis="small vessel disease",
            ...     condition_icd_codes=["437.3"],
            ... )
            >>> sample_dataset = dataset.set_task(task)
            >>> model = ZeroShotEvidenceLLM(dataset=sample_dataset)

    Citation:
        Ahsan et al. (2024) "Retrieving Evidence from EHRs with LLMs:
        Possibilities and Challenges." CHIL 2024, PMLR 248:489-505.
        https://arxiv.org/abs/2309.04550
    """

    def __init__(
        self,
        dataset: Any = None,
        model_name: str = "google/flan-t5-xxl",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256,
        device: Optional[str] = None,
        use_cbert_baseline: bool = False,
        cbert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    ) -> None:
        """Initialise ZeroShotEvidenceLLM.

        Args:
            dataset: PyHealth ``SampleDataset`` or ``None``.
            model_name (str): HuggingFace model ID.
            max_input_tokens (int): Input truncation limit.
            max_new_tokens (int): Generation length for summarisation.
            device (Optional[str]): Compute device.
            use_cbert_baseline (bool): Use CBERT instead of LLM.
            cbert_model_name (str): CBERT model ID.
        """
        super().__init__(dataset)

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.use_cbert_baseline = use_cbert_baseline
        self.cbert_model_name = cbert_model_name

        if device is None:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device_str = device

        # Lazy-loaded HuggingFace objects
        self._tokenizer = None
        self._hf_model = None
        self._is_encoder_decoder: Optional[bool] = None
        self._cbert_model = None
        self._cbert_tokenizer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_llm(self) -> None:
        """Lazy-load the tokenizer and language model."""
        if self._hf_model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "transformers is required: pip install transformers"
            ) from exc

        logger.info("Loading LLM: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Detect architecture type
        try:
            self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self._device_str)
            self._is_encoder_decoder = True
            logger.info("Loaded as encoder-decoder model.")
        except Exception:
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self._device_str)
            self._is_encoder_decoder = False
            logger.info("Loaded as decoder-only model.")

        self._hf_model.eval()

    def _load_cbert(self) -> None:
        """Lazy-load the Clinical BERT sentence encoder."""
        if self._cbert_model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required: pip install transformers"
            ) from exc

        logger.info("Loading CBERT baseline: %s", self.cbert_model_name)
        self._cbert_tokenizer = AutoTokenizer.from_pretrained(self.cbert_model_name)
        self._cbert_model = AutoModel.from_pretrained(self.cbert_model_name).to(
            self._device_str
        )
        self._cbert_model.eval()

    @torch.no_grad()
    def _encode_mean_pool(self, text: str) -> torch.Tensor:
        """Return mean-pooled BERT embedding for *text*."""
        inputs = self._cbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device_str)
        outputs = self._cbert_model(**inputs)
        # Mean pooling over token dimension
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    @torch.no_grad()
    def _llm_classify(
        self, notes: str, query_diagnosis: str
    ) -> Tuple[bool, float]:
        """Run the classification prompt and return (has_condition, confidence).

        Confidence is the normalised probability P(yes) / (P(yes) + P(no)).

        Args:
            notes (str): Patient notes (already truncated if needed).
            query_diagnosis (str): Free-text condition query.

        Returns:
            Tuple[bool, float]: Whether the model predicts the condition is
            present, and a scalar confidence score in [0, 1].
        """
        prompt = _CLASSIFY_PROMPT.format(
            notes=notes, query_diagnosis=query_diagnosis
        )

        if self._is_encoder_decoder:
            # Flan-T5 style: generate "yes"/"no" token
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self._device_str)

            # Score both "yes" and "no" by comparing generation probabilities
            yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = self._tokenizer.encode("no", add_special_tokens=False)[0]

            # Force-decode a single token and capture logits
            decoder_input = torch.tensor(
                [[self._tokenizer.pad_token_id]], device=self._device_str
            )
            outputs = self._hf_model(
                **inputs, decoder_input_ids=decoder_input
            )
            logits = outputs.logits[0, 0, :]  # (vocab_size,)
            yes_score = logits[yes_id].item()
            no_score = logits[no_id].item()

        else:
            # Decoder-only (Mistral-Instruct style)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self._device_str)
            outputs = self._hf_model(**inputs)
            logits = outputs.logits[0, -1, :]
            yes_id = self._tokenizer.encode(" yes", add_special_tokens=False)[-1]
            no_id = self._tokenizer.encode(" no", add_special_tokens=False)[-1]
            yes_score = logits[yes_id].item()
            no_score = logits[no_id].item()

        # Softmax normalisation for calibrated confidence
        yes_prob = torch.softmax(
            torch.tensor([yes_score, no_score]), dim=0
        )[0].item()
        has_condition = yes_prob >= 0.5
        return has_condition, float(yes_prob)

    @torch.no_grad()
    def _llm_summarise(self, notes: str, query_diagnosis: str) -> str:
        """Generate a free-text evidence summary given positive classification.

        Args:
            notes (str): Patient notes.
            query_diagnosis (str): Free-text condition query.

        Returns:
            str: Generated evidence summary.
        """
        prompt = _SUMMARISE_PROMPT.format(
            notes=notes, query_diagnosis=query_diagnosis
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self._device_str)

        output_ids = self._hf_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        if self._is_encoder_decoder:
            generated = output_ids[0]
        else:
            generated = output_ids[0][inputs["input_ids"].shape[-1] :]

        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _cbert_retrieve(
        self, notes: str, query_diagnosis: str
    ) -> Dict[str, Any]:
        """Dense retrieval baseline: return the most similar sentence.

        Splits *notes* into sentences, encodes each with Clinical BERT, and
        returns the single sentence whose embedding is most similar to the
        query embedding (cosine similarity).

        Args:
            notes (str): Patient notes (may contain note separators).
            query_diagnosis (str): Free-text condition query.

        Returns:
            Dict[str, Any]: Result dict compatible with :meth:`predict`.
        """
        self._load_cbert()

        # Split into sentences (simple heuristic)
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", notes)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 20]

        if not sentences:
            return {
                "has_condition": False,
                "evidence": "",
                "confidence": 0.0,
                "model": self.cbert_model_name,
            }

        query_emb = self._encode_mean_pool(query_diagnosis)
        best_sent = ""
        best_score = -1.0
        for sent in sentences:
            sent_emb = self._encode_mean_pool(sent)
            score = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0), sent_emb.unsqueeze(0)
            ).item()
            if score > best_score:
                best_score = score
                best_sent = sent

        return {
            "has_condition": best_score > 0.5,
            "evidence": best_sent,
            "confidence": float(best_score),
            "model": self.cbert_model_name,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, notes: str, query_diagnosis: str) -> Dict[str, Any]:
        """Run the two-step zero-shot evidence retrieval pipeline.

        Args:
            notes (str): Concatenated patient clinical notes.
            query_diagnosis (str): Free-text condition to query.

        Returns:
            Dict[str, Any]: Result dictionary with keys:

            - ``"has_condition"`` (bool): ``True`` if the model predicts the
              condition is present.
            - ``"evidence"`` (str): Free-text evidence summary (empty string
              when ``has_condition`` is ``False``).
            - ``"confidence"`` (float): Normalised P(yes) score from Step 1.
            - ``"model"`` (str): Model name used for inference.
        """
        if self.use_cbert_baseline:
            return self._cbert_retrieve(notes, query_diagnosis)

        self._load_llm()
        has_condition, confidence = self._llm_classify(notes, query_diagnosis)

        evidence = ""
        if has_condition:
            evidence = self._llm_summarise(notes, query_diagnosis)

        return {
            "has_condition": has_condition,
            "evidence": evidence,
            "confidence": confidence,
            "model": self.model_name,
        }

    def predict_batch(
        self, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run :meth:`predict` over a list of sample dicts.

        Each dict must contain ``"notes"`` and ``"query_diagnosis"`` keys,
        as produced by :class:`~pyhealth.tasks.EHREvidenceRetrievalTask`.

        Args:
            samples (List[Dict[str, Any]]): List of sample dicts.

        Returns:
            List[Dict[str, Any]]: Corresponding list of prediction dicts.
        """
        return [
            self.predict(s["notes"], s["query_diagnosis"]) for s in samples
        ]

    def forward(self, notes: List[str], query_diagnosis: List[str], **kwargs) -> Dict[str, Any]:
        """PyHealth-compatible forward pass for batch inference.

        Runs :meth:`predict` for each (notes, query) pair in the batch and
        returns aggregated results.  Note that this model is inference-only;
        no gradient computation is performed and no loss is returned.

        Args:
            notes (List[str]): Batch of concatenated note strings.
            query_diagnosis (List[str]): Batch of query diagnosis strings.
            **kwargs: Additional keys from the sample dict (ignored).

        Returns:
            Dict[str, Any]: Batch results with keys:

            - ``"has_condition"`` (List[bool])
            - ``"evidence"`` (List[str])
            - ``"confidence"`` (List[float])
        """
        results = [
            self.predict(n, q) for n, q in zip(notes, query_diagnosis)
        ]
        return {
            "has_condition": [r["has_condition"] for r in results],
            "evidence": [r["evidence"] for r in results],
            "confidence": [r["confidence"] for r in results],
        }
