import hashlib
import os
import re
import time
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from pyhealth.tasks.sdoh_utils import TARGET_CODES


PROMPT_TEMPLATE = """\
# SDOH Medical Coding Task

You are an expert medical coder. Your job is to find Social Determinants of Health (SDOH) codes in clinical notes.

## CRITICAL PREDICTION RULES:
1. **IF YOU FIND EVIDENCE FOR A CODE → YOU MUST PREDICT THAT CODE**
2. **BE CONSERVATIVE**: Only predict codes with STRONG evidence
3. **Evidence detection = Automatic code prediction**

## CODE HIERARCHY - ABSOLUTELY CRITICAL:
**HOMELESS (V600) COMPLETELY EXCLUDES V602:**
- ❌ If you see homelessness → ONLY predict V600, NEVER V602
- ❌ "Homeless and can't afford X" → STILL only V600
- ✅ V602 is EXCLUSIVELY for housed people with explicit financial hardship

**V602 ULTRA-STRICT REQUIREMENTS:**
- ✅ Must have exact quotes: "cannot afford medications", "no money for food"
- ❌ NEVER infer from: homelessness, unemployment, substance use, mental health
- ❌ NEVER predict from: "poor", "disadvantaged", social circumstances


## Available ICD-9 Codes:

**Housing & Resources:**
- V600: Homeless (living on streets, shelters, hotels, motels, temporary housing)
- V602: Cannot afford basic needs. **ULTRA-STRICT RULES**:
  - ✅ ONLY if EXPLICIT financial statements: "cannot afford medications", "unable to pay for treatment", "no money for food", "financial hardship preventing care"
  - ❌ NEVER predict from social circumstances: substance abuse, unemployment, mental health, divorce
  - ❌ NEVER predict from discharge inventory: "no money/wallet returned"
  - ❌ NEVER predict for insurance/benefits mentions: "on disability", "has Medicaid"
- V604: No family/caregiver available. **CRITICAL RULES**:
  - ❌ NOT just "lives alone" or "elderly"
  - ✅ ONLY if: "no family contact" AND "no support" AND "no one to help"
  - ❌ NOT for: "lives alone but daughter visits", "independent"

**Employment & Legal:**
- V620: Unemployed (current employment status) — **EXPLICIT ONLY**
  - ✅ **Predict only if one of these exact employment-status phrases appears (word boundaries, case-insensitive):**
    - unemploy(ed|ment)
    - jobless | no job
    - out of work | without work (employment context only; see exclusions)
    - not working (employment context only; see exclusions)
    - between jobs
    - on unemployment | receiving unemployment
    - Recent loss: laid off | fired | terminated | lost (my|his|her|their)? job
    - Label–value fields: Employment|Employment status|Work: → unemployed|none|not working ⇒ V620
  - ❌ Do NOT infer from former-only phrases: used to work | formerly | previously employed | last worked in ... UNLESS illness/disability is given as the reason for stopping work
  - ❌ **Contradictions (block V620 if present nearby, ±1 sentence)**: employed | works at/as | working | return(ed) to work | full-time/part-time | self-employed | contractor | on leave/LOA | work excuse | at work today
  - ❌ **Non-employment uses of "work" (never trigger)**: work of breathing | workup/work-up | social work | line not working | device not working | therapy not working | meds not working
  - ❌ **Context rule for "not/without work"**: Only treat as unemployment when it clearly refers to employment status (e.g., in Social History/Employment section or followed by "due to X" re job). Otherwise, do not code
  - ❌ **Exclusions**: retired | student | stay-at-home parent | on medical leave — unless an explicit unemployment phrase above is present
  - ✅ Include cases where patient stopped working or is unable to work due to illness/disability (including SSDI/SSI).
- V625: Legal problems. **ACTIVE LEGAL PROCESSES ONLY**:
  - ✅ Criminal: Arrest, jail, prison, parole, probation, active charges, bail, or court case pending
  - ✅ Civil: Court-ordered custody, restraining order, supervised release, or active litigation
  - ✅ Guardianship: Legal capacity hearing, court-appointed guardian, or power of attorney via court
  - ✅ Child welfare: DCF/CPS removed children, placed them in care, or court-ordered custody change
  - ❌ Do NOT predict V625 for:
    - History of crime, substance use, or homelessness without active legal process
    - Drug use history alone (marijuana, cocaine, etc.) without legal consequences
    - Psych history, social issues, or missed appointments without legal involvement
    - DCF "awareness" without custody change or legal action

**Health History:**
- V1541: Physical/sexual abuse (violence BY another person). **CRITICAL**:
  - ✅ Physical violence, sexual abuse, assault, domestic violence, rape
  - ❌ NOT accidents, falls, fights where patient was aggressor
- V1542: Pure emotional/psychological abuse. **MUTUALLY EXCLUSIVE WITH V1541**:
  - ✅ **ONLY if NO physical/sexual abuse mentioned AND explicit emotional abuse:**
  - ✅ Witnessed violence: "witnessed violence as a child", "saw domestic violence"
  - ✅ Verbal abuse: "verbal abuse", "emotionally abusive", "psychological abuse"
  - ✅ Emotional manipulation: "jealous and controlling", "isolation", "intimidation"
  - ❌ **NEVER predict V1542 if ANY physical/sexual abuse mentioned (use V1541 instead)**
  - ❌ **NEVER predict both V1541 AND V1542 for same patient**
  - ❌ Depression, anxiety, suicidal ideation alone without explicit abuse = NO CODE
  - ❌ "History of abuse" without specifying type = NO CODE
  - ❌ Psychiatric history alone = NO CODE
- V1584: Past asbestos exposure

**Family History:**
- V6141: Family alcoholism (family member drinks) — PREDICT if any kinship term + alcohol term appear together.
  - ✅ Kinship terms: father, mother, dad, mom, brother, sister, son, daughter, uncle, aunt, cousin, grand*; Headers: FH, FHx, Family History
  - ✅ Alcohol terms (case-insensitive, synonyms OK): ETOH/EtOH/etoh, alcohol, alcoholism, AUD, alcohol use disorder, EtOH abuse, EtOH dependence, "alcohol problem(s)", "drinks heavily", "alcoholic"
  - ✅ Mixed substance OK: If text says "drug and alcohol problems," still predict V6141
  - ✅ Outside headers OK: If kinship + alcohol appear in same clause/sentence or within ±1 line (e.g., "Pt's father ... has hx of etoh"), predict V6141
  - ✅ Examples to capture: "FH: Father – ETOH", "Mother has h/o alcoholism", "Father with depression and alcoholism", "Multiple family members with ETOH abuse ... (cousin, sister, uncle, aunt, father)", "Both brothers have drug and alcohol problems"
  - ❌ Negations: Do not predict if explicitly denied (e.g., "denies family history of alcoholism")
  - ❌ NEVER for PATIENT'S own history: "history of alcohol abuse", "with a history of alcohol use", "past medical history significant for heavy EtOH abuse", "patient alcoholic", "ETOH abuse"

## ENHANCED NEGATIVE EXAMPLES:

**V602 FALSE POSITIVES TO AVOID:**
❌ "Homeless patient" → Predict V600 ONLY, NEVER V602
❌ "Lives in shelter, gets food stamps" → V600 ONLY, NEVER V602
❌ "Homeless, on disability" → V600 ONLY, NEVER V602
❌ "No permanent address, has Medicaid" → V600 ONLY, NEVER V602
❌ "Homeless and can't afford medications" → V600 ONLY, NEVER V602
❌ "Unemployed alcoholic" → V620 (unemployment is explicit), NEVER V602
❌ "Lives in poverty" → NEVER V602 (too vague)
❌ "Financial strain from divorce" → NEVER V602 (circumstantial)

**V604 FALSE POSITIVES TO AVOID:**
❌ "82 year old lives alone" → NO CODE unless no support mentioned
❌ "Lives by herself" → NO CODE unless isolation confirmed
❌ "Widowed, lives alone, son calls daily" → NO CODE (has support)

**V1542 FALSE POSITIVES TO AVOID:**
❌ "History of physical and sexual abuse" → V1541 ONLY (physical trumps emotional)
❌ "PTSD from rape at age 7" → V1541 ONLY (sexual abuse)
❌ "Childhood sexual abuse by uncle" → V1541 ONLY (sexual abuse)
❌ "History of domestic abuse" → V1541 ONLY (physical abuse)
❌ "Depression and anxiety" → NO CODE (psychiatric symptoms alone)
❌ "Suicide attempts" → NO CODE (mental health history alone)
❌ "History of abuse" → NO CODE (unspecified type)
❌ "Recent argument with partner" → NO CODE (relationship conflict)

**V1542 TRUE POSITIVES TO CAPTURE:**
✅ "Witnessed violence as a child" → V1542 (pure emotional trauma, no physical)
✅ "Emotionally abusive relationship for 14 years" → V1542 (explicit emotional abuse)
✅ "Verbal abuse from controlling partner" → V1542 (explicit emotional abuse)
✅ "Jealous and controlling behavior" → V1542 (emotional manipulation)

## CONFIDENCE RULES:

**HIGH CONFIDENCE (Predict):**
- Direct statement of condition
- Multiple supporting evidence pieces
- Explicit language matching code definition

**LOW CONFIDENCE (Don't Predict):**
- Ambiguous language
- Single weak indicator
- Contradictory evidence

## Key Rules:

1. **Precision over Recall**: Better to miss a code than falsely predict
2. **Evidence-Driven**: Strong evidence required for prediction
3. **Multiple codes allowed**: But each needs independent evidence
4. **Conservative approach**: When in doubt, don't predict

## Output Format:
Return applicable codes separated by commas, or "None" if no codes apply.

Example:
```
V600, V625
```

or if no codes apply:
```
None
```

---

**Clinical Note to Analyze:**
{note}
"""


def _load_prompt_template() -> str:
    return PROMPT_TEMPLATE


class SDOHICD9LLM:
    """Admission-level SDOH ICD-9 V-code detector using an LLM.

    This model sends each note for an admission to an LLM, parses predicted
    ICD-9 V-codes, and aggregates the codes across notes (set union).

    Notes:
        - Use ``dry_run=True`` to skip LLM calls while exercising the pipeline.
        - Predictions are derived entirely from the LLM response parsing logic.

    Examples:
        >>> from pyhealth.models.sdoh_icd9_llm import SDOHICD9LLM
        >>> notes = [
        ...     "Pt is homeless and has no family support.",
        ...     "Social work consulted for housing resources.",
        ... ]
        >>> model = SDOHICD9LLM(dry_run=True, max_notes=2)
        >>> codes, note_results = model.predict_admission_with_notes(notes)
        >>> codes
        set()

        >>> model = SDOHICD9LLM(model_name="gpt-4o-mini", max_notes=1)
        >>> codes, note_results = model.predict_admission_with_notes(notes)
    """

    def __init__(
        self,
        target_codes: Optional[Sequence[str]] = None,
        model_name: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 100,
        max_chars: int = 100000,
        temperature: float = 0.0,
        sleep_s: float = 0.2,
        max_notes: Optional[int] = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the LLM wrapper.

        Args:
            target_codes: Target ICD-9 codes to retain after parsing.
            model_name: OpenAI model name.
            prompt_template: Optional prompt template override. Uses built-in
                SDOH template if not provided.
            api_key: OpenAI API key. Defaults to ``OPENAI_API_KEY`` env var.
            max_tokens: Max tokens for LLM response.
            max_chars: Max chars from each note to send.
            temperature: LLM temperature.
            sleep_s: Delay between per-note requests (seconds).
            max_notes: Optional limit on notes per admission.
            dry_run: If True, skips API calls and returns "None" responses.
        """
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.model_name = model_name
        self.prompt_template = prompt_template or _load_prompt_template()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.temperature = temperature
        self.sleep_s = sleep_s
        self.max_notes = max_notes
        self.dry_run = dry_run
        self._client = None

        if not self.api_key and not self.dry_run:
            raise EnvironmentError(
                "OPENAI_API_KEY is required unless dry_run=True."
            )

    def _get_client(self):
        """Initialize and cache the OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call_openai_api(self, text: str) -> str:
        """Send a single note to the LLM and return the raw response.

        Args:
            text: Note text to send.

        Returns:
            Raw string response from the LLM.
        """
        self._write_prompt_preview(text)

        if self.dry_run:
            return "```None```"

        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n\n[Note truncated due to length...]"

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.prompt_template.format(note=text)},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _write_prompt_preview(self, text: str) -> None:
        """Write the fully rendered prompt (with note) to a local file."""
        prompt = self.prompt_template.format(note=text)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
        filename = f"sdoh_prompt_{digest}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(prompt)

    def _parse_llm_response(self, response: str) -> Set[str]:
        """Parse the LLM response into a set of valid target codes.

        Returns:
            A set of ICD-9 codes intersected with ``target_codes``.
        """
        if not response:
            return set()

        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        if matches:
            response = matches[0].strip()
        else:
            response = response.strip()

        if response.lower().strip() == "none":
            return set()

        response = response.replace("Answer:", "").replace("Codes:", "").strip()
        for delimiter in [",", ";", " ", "\n"]:
            if delimiter in response:
                parts = [c.strip() for c in response.split(delimiter)]
                break
        else:
            parts = [response.strip()]

        valid = {code.upper() for code in parts if code.strip()}
        target_set = {code.upper() for code in self.target_codes}
        return {code for code in valid if code in target_set}

    def _predict_admission(
        self,
        notes: Iterable[str],
        note_categories: Optional[Iterable[str]] = None,
        chartdates: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[str], List[dict]]:
        """Run per-note predictions and aggregate codes for one admission.

        Args:
            notes: Iterable of note texts.
            note_categories: Optional note categories aligned to ``notes``.
            chartdates: Optional chart dates aligned to ``notes``.

        Returns:
            A tuple of (aggregated_codes, per_note_results).
        """
        aggregated: Set[str] = set()
        note_results: List[dict] = []
        categories = list(note_categories) if note_categories is not None else []
        dates = list(chartdates) if chartdates is not None else []
        notes_list = list(notes)
        if self.max_notes and self.max_notes > 0:
            notes_list = notes_list[: self.max_notes]
            categories = categories[: self.max_notes]
            dates = dates[: self.max_notes]

        for idx, note in enumerate(notes_list):
            category = categories[idx] if idx < len(categories) else "Unknown"
            date = dates[idx] if idx < len(dates) else "Unknown"
            response = self._call_openai_api(note)
            predicted = self._parse_llm_response(response)
            aggregated.update(predicted)

            note_results.append(
                {
                    "category": category,
                    "date": date,
                    "predicted_codes": sorted(predicted),
                    "llm_response": response,
                }
            )
            if self.sleep_s > 0 and not self.dry_run:
                time.sleep(self.sleep_s)

        return aggregated, note_results

    def predict_admission_with_notes(
        self,
        notes: Iterable[str],
        note_categories: Optional[Iterable[str]] = None,
        chartdates: Optional[Iterable[str]] = None,
    ) -> Tuple[Set[str], List[dict]]:
        """Predict codes for one admission using per-note LLM calls.

        Args:
            notes: Iterable of note texts.
            note_categories: Optional note categories aligned to ``notes``.
            chartdates: Optional chart dates aligned to ``notes``.

        Returns:
            A tuple of (aggregated_codes, per_note_results).
        """
        return self._predict_admission(notes, note_categories, chartdates)
