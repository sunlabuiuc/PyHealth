"""LLMSYN: LLM-based Synthetic EHR Generation model.

Reference:
    Hao et al., "LLMSYN: Generating Synthetic Electronic Health Records
    Without Patient-Level Data", MLHC 2024, PMLR 252.
    https://proceedings.mlr.press/v252/hao24a.html
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel



# ---------------------------------------------------------------------------
# Prompt templates (one per Markov step)
# ---------------------------------------------------------------------------

_STEP_TEMPLATES: Dict[int, str] = {
    0: """\
TASK:
Generate a synthetic EHR patient demographic profile.
Return ONLY these fields in exactly this format.

EXAMPLE OUTPUT:
Age: 67
Gender: Female
Ethnicity: WHITE
Insurance: Medicare
Survived: Yes

PRIOR STATISTICS:
{stats_summary}
""",
    1: """\
TASK:
Generate the main ICD-9 diagnosis for a synthetic patient.
Return ONLY this field in exactly this format.

EXAMPLE OUTPUT:
MainDiagnosis: ICD9:41401

PREVIOUS OUTPUT:
{prev_output}

PRIOR STATISTICS:
{stats_summary}
""",
    2: """\
TASK:
Generate complication ICD-9 codes for a synthetic patient.
Return ONLY this field. Use a comma-separated list or "None".

EXAMPLE OUTPUT:
Complications: ICD9:4280,ICD9:42731

PREVIOUS OUTPUT:
{prev_output}

MEDICAL KNOWLEDGE:
{medical_text}
""",
    3: """\
TASK:
Generate CPT procedure codes for a synthetic patient.
Return ONLY this field. Use a comma-separated list or "None".

EXAMPLE OUTPUT:
Procedures: CPT:93000,CPT:36415

PREVIOUS OUTPUT:
{prev_output}

MEDICAL KNOWLEDGE:
{medical_text}
""",
}

#: Built-in fallback statistics (MIMIC-III approximate values).
_DEFAULT_STATS: Dict[str, Any] = {
    "mortality_rate": 0.09926,
    "top_diseases": [
        {"icd9_code": "4019",  "desc": "Unspecified essential hypertension", "admission_count": 20696, "mortality_rate": 0.09775},
        {"icd9_code": "4280",  "desc": "Congestive heart failure, unspecified", "admission_count": 13111, "mortality_rate": 0.14370},
        {"icd9_code": "42731", "desc": "Atrial fibrillation", "admission_count": 12886, "mortality_rate": 0.14977},
        {"icd9_code": "41401", "desc": "Coronary atherosclerosis of native coronary artery", "admission_count": 12428, "mortality_rate": 0.07169},
        {"icd9_code": "5849",  "desc": "Acute kidney failure, unspecified","admission_count":  9118, "mortality_rate": 0.19950},
        {"icd9_code": "25000", "desc": "Diabetes mellitus without mention of complication, type II", "admission_count":  9057, "mortality_rate": 0.11958},
        {"icd9_code": "2724",  "desc": "Other and unspecified hyperlipidemia", "admission_count":  8689, "mortality_rate": 0.07481},
        {"icd9_code": "51881", "desc": "Acute respiratory failure", "admission_count":  7497, "mortality_rate": 0.30439},
        {"icd9_code": "5990",  "desc": "Urinary tract infection, site not specified", "admission_count":  6555, "mortality_rate": 0.12372},
        {"icd9_code": "53081", "desc": "Esophageal reflux", "admission_count":  6324, "mortality_rate": 0.07606},
    ],
    "demographics": {
        "age_mean": 55.0,
        "age_std": 27.2,
        "gender": {"M": 0.5615, "F": 0.4385},
        "insurance": {"Medicare": 0.4784, "Private": 0.3829, "Medicaid": 0.0981, "Government": 0.0302, "Self Pay": 0.0104},
        "ethnicity": {"WHITE": 0.6951, "BLACK/AFRICAN AMERICAN": 0.0922, "HISPANIC OR LATINO": 0.0288, "UNKNOWN/NOT SPECIFIED": 0.0767},
    },
}

def _compact_stats(stats: Dict[str, Any]) -> str:
    """Render a compact text summary of prior statistics for prompting.

    Args:
        stats: Statistics dict with ``mortality_rate``, ``top_diseases``,
            and ``demographics`` keys.

    Returns:
        Multi-line string for inclusion in an LLM prompt.
    """
    top = stats.get("top_diseases", [])[:5]
    lines = [
        f"Overall mortality rate: {stats.get('mortality_rate', 'unknown')}"
    ]
    lines.append("Top diseases:")
    for d in top:
        lines.append(
            f"  - {d.get('icd9_code')}: {d.get('desc', '')} "
            f"(n={d.get('admission_count')}, "
            f"mort={d.get('mortality_rate')})"
        )
    demo = stats.get("demographics", {})
    lines.append(
        f"Age mean/std: {demo.get('age_mean')}/{demo.get('age_std')}"
    )
    lines.append(f"Gender: {json.dumps(demo.get('gender', {}))}")
    lines.append(f"Insurance: {json.dumps(demo.get('insurance', {}))}")
    return "\n".join(lines)


def _build_prompt(
    step: int,
    prev_output: Optional[str],
    stats: Dict[str, Any],
    medical_text: Optional[str],
) -> str:
    """Build a structured prompt for the given Markov generation step.

    Args:
        step: Step index (0-3).
        prev_output: Raw LLM output from the previous step.
        stats: Prior statistics dict.
        medical_text: Retrieved medical knowledge (steps 2 and 3).

    Returns:
        Formatted prompt string.
    """
    template = _STEP_TEMPLATES.get(step, "Generate structured EHR output.")
    return template.format(
        stats_summary=_compact_stats(stats),
        prev_output=prev_output or "None",
        medical_text=medical_text or "None",
    )


def _normalize_icd9(code: str) -> str:
    code = code.strip().upper()
    if code.startswith("ICD9:"):
        return code
    return f"ICD9:{code}"


def _normalize_cpt(code: str) -> str:
    code = code.strip().upper()
    if code.startswith("CPT:"):
        return code
    return f"CPT:{code}"


def _parse_codes(text: str, prefix: str) -> List[str]:
    pattern = rf"{prefix}:[\w]+"
    matches = re.findall(pattern, text, flags=re.I)
    if prefix == "ICD9":
        return [_normalize_icd9(m) for m in matches]
    return [_normalize_cpt(m) for m in matches]


def _parse_output(text: str) -> Dict[str, Any]:
    """Parse one generation step's LLM output into a structured dict.

    Args:
        text: Raw LLM response text.

    Returns:
        Dict with keys: Age, Gender, Ethnicity, Insurance, Survived,
        MainDiagnosis, Complications, Procedures.
    """
    record: Dict[str, Any] = {
        "Age": None,
        "Gender": None,
        "Ethnicity": None,
        "Insurance": None,
        "Survived": None,
        "MainDiagnosis": None,
        "Complications": [],
        "Procedures": [],
    }
    _scalar_patterns = {
        "Age": (r"Age:\s*(\d{1,3})", lambda m: int(m.group(1))),
        "Gender": (
            r"Gender:\s*(Male|Female|Other)",
            lambda m: m.group(1).title(),
        ),
        "Ethnicity": (r"Ethnicity:\s*(.+)", lambda m: m.group(1).strip()),
        "Insurance": (r"Insurance:\s*(.+)", lambda m: m.group(1).strip()),
        "Survived": (
            r"Survived:\s*(Yes|No)", lambda m: m.group(1).title()
        ),
        "MainDiagnosis": (
            r"MainDiagnosis:\s*((?:ICD9:)?[A-Za-z0-9]+)",
            lambda m: _normalize_icd9(m.group(1)),
        ),
    }
    for key, (pat, fn) in _scalar_patterns.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            record[key] = fn(m)

    comp_m = re.search(r"Complications:\s*(.+)", text, flags=re.I)
    if comp_m:
        val = comp_m.group(1).strip()
        if val.lower() not in ("none", "n/a", ""):
            record["Complications"] = _parse_codes(val, "ICD9")

    proc_m = re.search(r"Procedures:\s*(.+)", text, flags=re.I)
    if proc_m:
        val = proc_m.group(1).strip()
        if val.lower() not in ("none", "n/a", ""):
            record["Procedures"] = _parse_codes(val, "CPT")
    return record


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


class _BaseLLMBackend(ABC):
    """Abstract interface for LLM generation backends."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """


class _MockLLMBackend(_BaseLLMBackend):
    """Deterministic mock backend for testing (no API calls required).

    Args:
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Return a canned response matched to the prompt step."""
        rng = self._rng
        if "patient demographic profile" in prompt:
            return (
                f"Age: {rng.randint(18, 90)}\n"
                f"Gender: {rng.choice(['Male', 'Female'])}\n"
                f"Ethnicity: "
                f"{rng.choice(['WHITE', 'BLACK/AFRICAN AMERICAN'])}\n"
                f"Insurance: {rng.choice(['Medicare', 'Private'])}\n"
                f"Survived: {rng.choice(['Yes'] * 85 + ['No'] * 15)}"
            )
        if "main ICD-9 diagnosis" in prompt:
            return (
                f"MainDiagnosis: "
                f"ICD9:{rng.choice(['4019', '41401', '42731'])}"
            )
        if "complication ICD-9 codes" in prompt:
            k = rng.randint(0, 2)
            if k == 0:
                return "Complications: None"
            opts = ["ICD9:4280", "ICD9:42731", "ICD9:5849"]
            return f"Complications: {','.join(rng.choices(opts, k=k))}"
        if "CPT procedure codes" in prompt:
            k = rng.randint(0, 2)
            if k == 0:
                return "Procedures: None"
            opts = ["CPT:93000", "CPT:36415", "CPT:99232"]
            return f"Procedures: {','.join(rng.choices(opts, k=k))}"
        return "MainDiagnosis: ICD9:4019"


class _OpenAILLMBackend(_BaseLLMBackend):
    """OpenAI chat backend (openai>=1.0).

    Args:
        api_key: OpenAI API key.
        model: Model identifier (e.g. ``"gpt-3.5-turbo"``).
    """

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo") -> None:
        try:
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requires: pip install openai>=1.0"
            ) from exc
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Call OpenAI chat completions and return the response text."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content


class _AnthropicLLMBackend(_BaseLLMBackend):
    """Anthropic Claude backend.

    Args:
        api_key: Anthropic API key.
        model: Model identifier (e.g. ``"claude-sonnet-4-6"``).
    """

    def __init__(
        self, api_key: str, model: str = "claude-sonnet-4-6"
    ) -> None:
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires: pip install anthropic"
            ) from exc
        self._model = model

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Call Anthropic Messages API and return the response text."""
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


# ---------------------------------------------------------------------------
# Internal 4-step Markov generation pipeline
# ---------------------------------------------------------------------------


class _LLMSYNPipeline:
    """4-step Markov EHR generation pipeline (internal).

    Args:
        backend: LLM generation backend instance.
        stats: Prior statistics dict.
        noise_scale: Noise std for stats perturbation.
        prior_mode: ``"full"`` or ``"sampled"`` disease sampling.
        enable_rag: Whether to append retrieved medical knowledge.
    """

    def __init__(
        self,
        backend: _BaseLLMBackend,
        stats: Dict[str, Any],
        noise_scale: float = 0.0,
        prior_mode: str = "full",
        enable_rag: bool = False,
    ) -> None:
        self._backend = backend
        self._stats = stats
        self._noise_scale = noise_scale
        self._prior_mode = prior_mode
        self._enable_rag = enable_rag
        self._icd_desc: Dict[str, str] = {
            str(d["icd9_code"]): d.get("desc", str(d["icd9_code"]))
            for d in stats.get("top_diseases", [])
        }

    def _perturb_stats(self) -> Dict[str, Any]:
        import copy
        s = copy.deepcopy(self._stats)
        diseases = s.get("top_diseases", [])
        if diseases:
            weights = [max(1, d.get("admission_count", 1)) for d in diseases]
            sampled = random.choices(diseases, weights=weights, k=1)[0]
            if self._prior_mode == "sampled":
                s["top_diseases"] = [sampled]
            else:
                rest = [d for d in diseases if d is not sampled]
                s["top_diseases"] = [sampled] + rest

        # Age jitter is always applied
        demo = s.get("demographics", {})
        if "age_mean" in demo:
            demo["age_mean"] = max(0.0, demo["age_mean"] + random.uniform(-5.0, 5.0))
        if "age_std" in demo:
            demo["age_std"] = max(1.0, demo["age_std"] + random.uniform(-2.0, 2.0))

        # Larger noise only when noise_scale > 0
        if self._noise_scale > 0:
            ns = self._noise_scale
            mr = s.get("mortality_rate", 0.0)
            s["mortality_rate"] = max(0.0, min(1.0, mr + random.uniform(-ns, ns)))
            for d in s.get("top_diseases", []):
                if "mortality_rate" in d:
                    d["mortality_rate"] = max(0.0, min(1.0, d["mortality_rate"] + random.uniform(-ns, ns)))
                if "admission_count" in d:
                    d["admission_count"] = max(1, int(d["admission_count"] * (1 + random.uniform(-ns, ns))))
        return s

    def _get_medical_text(
        self,
        parsed: Dict[str, Any],
        stats: Dict[str, Any],
    ) -> Optional[str]:
        """Build medical context for Step-2/3 prompts.

        Args:
            parsed: Parsed Step-1 output (contains MainDiagnosis).
            stats: Perturbed stats for this record.

        Returns:
            Medical context string or ``None``.
        """
        main = parsed.get("MainDiagnosis")
        if not main:
            return None
        code = main.replace("ICD9:", "").strip()
        text: Optional[str] = None
        for d in stats.get("top_diseases", []):
            if str(d.get("icd9_code")) == code:
                text = (
                    f"Code {code}: {d.get('desc', 'unknown')}. "
                    f"Observed mortality: {d.get('mortality_rate')}. "
                    "Generate clinically plausible complications."
                )
                break
        if text is None:
            text = (
                f"Code {code}: generate plausible downstream "
                "clinical details."
            )
        if self._enable_rag:
            desc = self._icd_desc.get(code, code)
            try:
                from pyhealth.datasets.llmsyn_utils import (
                    get_medical_knowledge,
                )

                retrieved = get_medical_knowledge(desc)
                if retrieved:
                    text += f"\nMedical Knowledge: {retrieved}"
            except Exception:
                pass
        return text

    def generate_record(self) -> Dict[str, Any]:
        """Run the 4-step Markov loop and return one synthetic EHR record.

        Returns:
            Dict with keys: Age, Gender, Ethnicity, Insurance, Survived,
            MainDiagnosis, Complications, Procedures, Raw.
        """
        stats = self._perturb_stats()
        prev_output: Optional[str] = None
        medical_text: Optional[str] = None
        step_outputs: Dict[str, Any] = {}

        for step_idx in range(4):
            prompt = _build_prompt(
                step_idx, prev_output, stats, medical_text
            )
            raw = self._backend.generate(prompt)
            parsed = _parse_output(raw)
            step_outputs[f"step_{step_idx}_raw"] = raw

            if step_idx == 0:
                step_outputs["demographics"] = parsed
            elif step_idx == 1:
                step_outputs["diagnosis"] = parsed
                medical_text = self._get_medical_text(parsed, stats)
            elif step_idx == 2:
                step_outputs["complications"] = parsed
            elif step_idx == 3:
                step_outputs["procedures"] = parsed

            prev_output = raw  # Markov: thread output to next step

        demo = step_outputs.get("demographics", {})
        diag = step_outputs.get("diagnosis", {})
        comp = step_outputs.get("complications", {})
        proc = step_outputs.get("procedures", {})

        return {
            "Age": demo.get("Age"),
            "Gender": demo.get("Gender"),
            "Ethnicity": demo.get("Ethnicity"),
            "Insurance": demo.get("Insurance"),
            "Survived": demo.get("Survived"),
            "MainDiagnosis": diag.get("MainDiagnosis"),
            "Complications": comp.get("Complications", []),
            "Procedures": proc.get("Procedures", []),
            "Raw": step_outputs,
        }

    def generate_n(self, n: int) -> List[Dict[str, Any]]:
        """Generate ``n`` synthetic EHR records sequentially.

        Args:
            n: Number of records to generate.

        Returns:
            List of EHR record dicts.
        """
        return [self.generate_record() for _ in range(n)]


# ---------------------------------------------------------------------------
# Public PyHealth model
# ---------------------------------------------------------------------------


class LLMSYNModel(BaseModel):
    """LLM-based Synthetic EHR Generation (LLMSYN) model.

    Implements the 4-step Markov generation pipeline from:

        Hao et al., "LLMSYN: Generating Synthetic Electronic Health Records
        Without Patient-Level Data", MLHC 2024, PMLR 252.
        https://proceedings.mlr.press/v252/hao24a.html

    The pipeline conditions each step on the previous output (Markov
    property), reducing hallucination:

    - **Step 0** - Demographics: age, gender, ethnicity, insurance, survival.
    - **Step 1** - Main ICD-9 diagnosis (conditioned on Step 0).
    - **Step 2** - Complication ICD-9 codes (conditioned on Steps 0-1 + RAG).
    - **Step 3** - CPT procedure codes (conditioned on Steps 0-2 + RAG).

    Three ablation variants are supported:

    - **LLMSYNfull** (``prior_mode="full"``, ``enable_rag=True``):
      Prior statistics + RAG from Mayo Clinic.
    - **LLMSYNprior** (``prior_mode="full"``, ``enable_rag=False``):
      Prior statistics only.
    - **LLMSYNbase** (``prior_mode="sampled"``, ``enable_rag=False``):
      Minimal prompts, single sampled disease per record.

    For PyHealth integration, ``forward()`` frames generation as a TSTR
    (Train-on-Synthetic, Test-on-Real) mortality prediction task: a synthetic
    record is generated per batch item, and its predicted survival is compared
    against the real mortality label via BCE loss.
    Use :meth:`generate` for standalone synthetic EHR generation.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` providing
            ``input_schema`` / ``output_schema`` context for BaseModel.
        llm_provider: Backend for LLM calls. ``"mock"`` (default, no API
            key needed), ``"openai"``, or ``"claude"``.
        noise_scale: Gaussian noise std added to prior statistics at each
            generation call. ``0.0`` disables noise.
        prior_mode: ``"full"`` (weighted sampling) or ``"sampled"``
            (single disease per record).
        enable_rag: Append retrieved Mayo Clinic knowledge to prompts.
        stats: Pre-loaded statistics dict (``mortality_rate``,
            ``top_diseases``, ``demographics``). Defaults to built-in
            MIMIC-III approximate values when ``None``.
         stats_path: Path to a JSON file to load statistics from (e.g.
            ``"test-resources/llmsyn/stats.json"``). Overrides ``stats``
            when provided.
        api_key: API key for ``"openai"`` or ``"claude"`` providers.
        seed: Random seed for ``"mock"`` backend.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import LLMSYNModel
        >>> dataset = create_sample_dataset(
        ...     samples=[{
        ...         "patient_id": "p0", "visit_id": "v0",
        ...         "conditions": ["4019", "41401"], "mortality": 0,
        ...     }],
        ...     input_schema={"conditions": "sequence"},
        ...     output_schema={"mortality": "binary"},
        ...     dataset_name="llmsyn_test",
        ... )
        >>> model = LLMSYNModel(dataset=dataset, llm_provider="mock", seed=0)
        >>> records = model.generate(n=2)
        >>> print(list(records[0].keys()))
        ['Age', 'Gender', 'Ethnicity', 'Insurance', 'Survived',
         'MainDiagnosis', 'Complications', 'Procedures']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        llm_provider: str = "mock",
        noise_scale: float = 0.0,
        prior_mode: str = "full",
        enable_rag: bool = False,
        stats: Optional[Dict[str, Any]] = None,
        stats_path: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(dataset)
        self.label_key = self.label_keys[0]
        # Projection layer converts the binary survival signal to a logit,
        # satisfying PyHealth's forward() contract with a trainable parameter.
        self._proj = nn.Linear(1, 1)
        backend = self._build_backend(llm_provider, api_key, seed)
        if stats_path is not None:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
        effective_stats = stats if stats is not None else _DEFAULT_STATS
        self._pipeline = _LLMSYNPipeline(
            backend=backend,
            stats=effective_stats,
            noise_scale=noise_scale,
            prior_mode=prior_mode,
            enable_rag=enable_rag,
        )

    @staticmethod
    def _build_backend(
        provider: str,
        api_key: Optional[str],
        seed: Optional[int],
    ) -> _BaseLLMBackend:
        """Instantiate the requested LLM backend.

        Args:
            provider: ``"mock"``, ``"openai"``, or ``"claude"``.
            api_key: Required for real providers.
            seed: Seed for mock backend.

        Returns:
            :class:`_BaseLLMBackend` instance.

        Raises:
            ValueError: Unknown provider or missing api_key.
        """
        if provider == "mock":
            return _MockLLMBackend(seed=seed)
        if provider == "openai":
            if not api_key:
                raise ValueError(
                    "api_key is required for llm_provider='openai'"
                )
            return _OpenAILLMBackend(api_key=api_key)
        if provider == "claude":
            if not api_key:
                raise ValueError(
                    "api_key is required for llm_provider='claude'"
                )
            return _AnthropicLLMBackend(api_key=api_key)
        raise ValueError(
            f"Unknown llm_provider '{provider}'. "
            "Choose from: 'mock', 'openai', 'claude'."
        )

    def generate(self, n: int = 1) -> List[Dict[str, Any]]:
        """Generate synthetic EHR records using the LLMSYN pipeline.

        Primary API for standalone bulk generation, independent of the
        PyHealth training loop.

        Args:
            n: Number of records to generate.

        Returns:
            List of ``n`` EHR record dicts, each with keys:
            ``Age``, ``Gender``, ``Ethnicity``, ``Insurance``,
            ``Survived``, ``MainDiagnosis``, ``Complications``,
            ``Procedures``.
        """
        return self._pipeline.generate_n(n)

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Generate one synthetic record per batch item and compute TSTR loss.

        Generates a synthetic EHR record for each batch item via the 4-step
        Markov pipeline, then computes BCE loss between the generated
        survival prediction and the real mortality label, implementing the
        TSTR evaluation framing from the paper.

        Args:
            **kwargs: All keys from ``dataset.input_schema`` plus the label
                key from ``dataset.output_schema``.

        Returns:
            Dict with keys:

            - ``loss``: scalar BCE loss tensor.
            - ``y_prob``: predicted survival probabilities,
              shape ``(batch_size, 1)``.
            - ``y_true``: real mortality labels, shape ``(batch_size,)``.
            - ``logit``: raw projected logits, shape ``(batch_size, 1)``.
        """
        first = kwargs[self.feature_keys[0]]
        batch_size = (
            first.shape[0]
            if isinstance(first, torch.Tensor)
            else len(first)
        )

        records = self._pipeline.generate_n(batch_size)

        survived_raw = torch.tensor(
            [
                [1.0 if r.get("Survived") == "Yes" else 0.0]
                for r in records
            ],
            dtype=torch.float32,
            device=self.device,
        )
        logit = self._proj(survived_raw)  # (batch_size, 1)

        y_true = kwargs[self.label_key].float().to(self.device)
        y_true_2d = (
            y_true.unsqueeze(1) if y_true.dim() == 1 else y_true
        )
        loss = nn.BCEWithLogitsLoss()(logit, y_true_2d)
        y_prob = torch.sigmoid(logit)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }