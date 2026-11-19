"""Boundary-aware medication recommendation using a trained EHR model.

GiantMed: Lv et al., KDD 2026, https://doi.org/10.1145/3770854.3780297.
The refiner operates on raw task samples and probabilities from a separately
trained model. Retrieval must be fitted on training patients only.
"""

import ast
import csv
import gzip
import hashlib
import json
import math
import re
import string
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

_DEFAULT_PROMPT_TEMPLATE = (
    "You are a clinical pharmacy expert. Revise medication probabilities "
    "in {boundary} predicted by a deep model. "
    "A score >= 0.5 recommends a drug. Use the current visit, prior "
    "medication history, similar training visits and potential interactions. "
    "Treat model scores as priors, not sufficient justification. Raise "
    "clinically necessary medications toward 0.70-0.90 and lower unsupported "
    "ones; avoid leaving scores in the ambiguous boundary. Avoid recommending "
    "medications with serious interactions or functional redundancy."
)


class GiantMed:
    """Refine boundary medication probabilities using similar visits and an LLM.

    Args:
        medication_codes: Codes in the exact order of the backbone output.
        ddi_matrix: Interaction adjacency in the same order.
        generate: Callable accepting a prompt and returning generated text.
        train_samples: Raw drug-recommendation task samples from training patients.
        code_names: Optional mapping from codes to names for the prompt. A mapping
            may also use field names (conditions, procedures, drugs) as keys.
        boundary_low: Inclusive lower boundary, default 0.3.
        boundary_high: Exclusive upper boundary, default 0.7.
        top_k: Number of similar training visits, default 3.
        upper_inclusive: Include the upper boundary as in the paper. The
            released LLM refinement script uses an exclusive upper boundary.
        evidence_filter: Optional callable ``(query, neighbour, drug_code) -> bool``
            for filtering retrieved medication evidence. The reference code uses
            MEDI indication mappings and previous medication history for this.
        prompt_token_counter: Optional callable counting input tokens including
            the chat template. Required with ``max_prompt_tokens``.
        max_prompt_tokens: Maximum counted prompt tokens, after any caller's
            reserved provider overhead. Supporting evidence may be truncated.
        prompt_template: Optional single-line instruction template containing
            ``{boundary}``. The renderer retains control of evidence and output
            format. The template must not contain patient-specific information.
        candidate_batch_size: Optional maximum target medications per prompt.
            Requires a token counter and budget. Each batch includes every
            boundary interaction involving its targets, including interactions
            with candidates targeted in other batches.

    Notes:
        The LLM output uses medication codes to avoid ambiguous names. Missing,
        duplicate, unknown and invalid responses retain the backbone score.
        DDI information guides the LLM; it is not a hard safety constraint.
        This component is intended for research, not clinical use.

    Examples:
        >>> model = GiantMed(["A", "B"], [[0, 1], [1, 0]],
        ...                  lambda prompt: '{"A": 0.8}', [])
        >>> result = model.refine_with_details({"patient_id": "p"}, [0.5, 0.9])
        >>> result["probabilities"].tolist()
        [0.8, 0.9]
    """

    def __init__(
        self,
        medication_codes: Sequence[str],
        ddi_matrix,
        generate: Callable,
        train_samples: Sequence[Mapping],
        code_names: Mapping | None = None,
        boundary_low: float = 0.3,
        boundary_high: float = 0.7,
        top_k: int = 3,
        evidence_filter: Callable | None = None,
        upper_inclusive: bool = False,
        prompt_token_counter: Callable | None = None,
        max_prompt_tokens: int | None = None,
        candidate_batch_size: int | None = None,
        prompt_template: str | None = None,
    ):
        self.codes = list(medication_codes)
        if not all(isinstance(code, str) and code for code in self.codes):
            raise ValueError("medication_codes must contain nonempty strings")
        if not self.codes or len(set(self.codes)) != len(self.codes):
            raise ValueError("medication_codes must be nonempty and unique")
        self.ddi = np.asarray(ddi_matrix)
        if self.ddi.shape != (len(self.codes), len(self.codes)):
            raise ValueError("ddi_matrix must match the ordered medication vocabulary")
        if not np.isin(self.ddi, [0, 1]).all():
            raise ValueError("ddi_matrix must be binary")
        if not 0 <= boundary_low < boundary_high <= 1:
            raise ValueError("boundary must satisfy 0 <= low < high <= 1")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be positive")
        self.low, self.high, self.top_k = boundary_low, boundary_high, top_k
        if prompt_template is not None:
            if (
                not isinstance(prompt_template, str)
                or not prompt_template.strip()
                or len(prompt_template.splitlines()) != 1
            ):
                raise ValueError(
                    "prompt_template must be a nonempty single-line string"
                )
            try:
                fields = [
                    (field, spec, conversion)
                    for _, field, spec, conversion in string.Formatter().parse(
                        prompt_template
                    )
                    if field is not None
                ]
            except ValueError as exc:
                raise ValueError("invalid prompt_template") from exc
            if not fields or any(
                field != "boundary" or spec or conversion
                for field, spec, conversion in fields
            ):
                raise ValueError(
                    "prompt_template must use only the {boundary} placeholder"
                )
        self.prompt_template = prompt_template
        self.generate = generate
        self.names = dict(code_names or {})
        self.evidence_filter = evidence_filter
        self.upper_inclusive = upper_inclusive
        if (prompt_token_counter is None) != (max_prompt_tokens is None):
            raise ValueError(
                "provide prompt_token_counter and max_prompt_tokens together"
            )
        if max_prompt_tokens is not None and (
            not isinstance(max_prompt_tokens, int) or max_prompt_tokens < 1
        ):
            raise ValueError("max_prompt_tokens must be a positive integer")
        self.prompt_token_counter = prompt_token_counter
        self.max_prompt_tokens = max_prompt_tokens
        if candidate_batch_size is not None:
            if (
                not isinstance(candidate_batch_size, int)
                or isinstance(candidate_batch_size, bool)
                or candidate_batch_size < 1
            ):
                raise ValueError("candidate_batch_size must be a positive integer")
            if prompt_token_counter is None:
                raise ValueError(
                    "candidate batching requires a token counter and budget"
                )
        if candidate_batch_size is not None and any(
            code.isascii() and code.isdigit() for code in self.codes
        ):
            raise ValueError("numeric medication codes are ambiguous in batched mode")
        self.candidate_batch_size = candidate_batch_size
        self.train_samples = list(train_samples)
        if any("patient_id" not in s for s in self.train_samples):
            raise ValueError("training samples must contain patient_id")

    @staticmethod
    def _current(sample, field):
        visits = sample.get(field, [])
        return list(visits[-1]) if visits else []

    def retrieve(self, sample):
        """Return the most similar training visits, excluding the query patient."""
        if "patient_id" not in sample:
            raise ValueError("query sample must contain patient_id")
        query = set(self._current(sample, "conditions"))
        candidates = []
        for row in self.train_samples:
            if str(row["patient_id"]) == str(sample["patient_id"]):
                continue
            diagnoses = set(self._current(row, "conditions"))
            union = query | diagnoses
            score = len(query & diagnoses) / len(union) if union else 0.0
            candidates.append(
                (score, str(row["patient_id"]), str(row.get("visit_id", "")), row)
            )
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in candidates[: self.top_k]]

    def _describe(self, code, field="drugs"):
        names = self.names.get(field, self.names)
        return f"{code} ({names.get(code, code)})"

    def _instructions(self):
        """Shared clinical guidance for every prompt format."""
        bracket = "]" if self.upper_inclusive else ")"
        return (self.prompt_template or _DEFAULT_PROMPT_TEMPLATE).format(
            boundary=f"[{self.low}, {self.high}{bracket}"
        )

    def prompt_metadata(self):
        """Return patient-independent template identity and renderer version.

        Examples:
            >>> model = GiantMed(["A"], [[0]], lambda p: "{}", [])
            >>> model.prompt_metadata()["renderer_version"]
            'giantmed-renderer-v1'
        """
        source = self.prompt_template or _DEFAULT_PROMPT_TEMPLATE
        return {
            "template": source,
            "template_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "renderer_version": "giantmed-renderer-v1",
        }

    def build_prompt(self, sample, probabilities, indices, neighbours):
        """Construct the evidence prompt without current-visit target drugs."""
        current_d = set(self._current(sample, "conditions"))
        current_p = set(self._current(sample, "procedures"))
        candidates = {self.codes[i] for i in indices}
        lines = [
            self._instructions()
            + " Return only a JSON object mapping each candidate medication code "
            'to a numeric probability in [0, 1], for example {"A10A": 0.8}. '
            "Do not use nested objects.",
            "Current diagnoses: "
            + json.dumps([self._describe(c, "conditions") for c in sorted(current_d)]),
            "Current procedures: "
            + json.dumps([self._describe(c, "procedures") for c in sorted(current_p)]),
            "Previous medications: "
            + json.dumps(
                [
                    [self._describe(c) for c in visit]
                    for visit in sample.get("drugs_hist", [])[:-1]
                ]
            ),
            "Candidates: "
            + json.dumps(
                {self.codes[i]: round(float(probabilities[i]), 4) for i in indices}
            ),
            "Candidate names: "
            + json.dumps(
                {
                    self.codes[i]: self.names.get("drugs", self.names).get(
                        self.codes[i], self.codes[i]
                    )
                    for i in indices
                }
            ),
        ]
        evidence = []
        for row in neighbours:
            drugs = [
                c
                for c in row.get("drugs", [])
                if c in candidates
                and (
                    self.evidence_filter is None or self.evidence_filter(sample, row, c)
                )
            ]
            evidence.append(
                {
                    "overlapping_diagnoses": [
                        self._describe(c, "conditions")
                        for c in sorted(
                            current_d & set(self._current(row, "conditions"))
                        )
                    ],
                    "overlapping_procedures": [
                        self._describe(c, "procedures")
                        for c in sorted(
                            current_p & set(self._current(row, "procedures"))
                        )
                    ],
                    "medications": [self._describe(c) for c in sorted(set(drugs))],
                }
            )
        lines.append("Similar training visits: " + json.dumps(evidence))
        pairs = [
            (self.codes[i], self.codes[j])
            for pos, i in enumerate(indices)
            for j in indices[pos + 1 :]
            if self.ddi[i, j] or self.ddi[j, i]
        ]
        lines.append("Potential interactions among candidates: " + json.dumps(pairs))
        return "\n".join(lines)

    def refine(self, sample: Mapping, probabilities):
        """Return refined probabilities and the prompt, without changing input."""
        result = self.refine_with_details(sample, probabilities)
        return result["probabilities"], result["prompt"]

    def _fit_prompt(self, prompt):
        """Fit supporting evidence while retaining current visit and candidates."""
        if self.prompt_token_counter is None:
            return prompt, None
        lines = prompt.splitlines()
        sections = {
            line.split(": ", 1)[0]: json.loads(line.split(": ", 1)[1])
            for line in lines[1:]
        }
        history = sections["Previous medications"]
        neighbours = sections["Similar training visits"]
        pairs = sections["Potential interactions among candidates"]
        candidate_scores = sections["Candidates"]
        candidate_names = sections["Candidate names"]
        candidate_order = list(candidate_scores)
        candidate_index = {code: index for index, code in enumerate(candidate_order)}
        ordered_names = sorted(
            candidate_names, key=lambda code: (-candidate_scores[code], code)
        )

        # Names for overlapping diagnoses and procedures already appear in the
        # current visit. Use codes here to avoid sending those descriptions again.
        def described_code(value):
            return value.split(" (", 1)[0]

        neighbours = [
            {
                "diagnoses": [described_code(x) for x in row["overlapping_diagnoses"]],
                "procedures": [
                    described_code(x) for x in row["overlapping_procedures"]
                ],
                "medications": [
                    candidate_index[described_code(x)] for x in row["medications"]
                ],
            }
            for row in neighbours
        ]
        # Keep interactions involving higher-scored candidates first. This is
        # a deterministic prioritization, not an interaction severity ranking.
        pairs.sort(
            key=lambda pair: (
                -min(candidate_scores[pair[0]], candidate_scores[pair[1]]),
                -max(candidate_scores[pair[0]], candidate_scores[pair[1]]),
                pair,
            )
        )
        selected = {
            "Candidate names": {},
            "Previous medications": [],
            "Similar training visits": [],
            "Potential interactions among candidates": [],
        }

        def render():
            parts = [
                lines[0],
                (
                    "Candidates are rows [code, score, name], numbered from 0 in order. "
                    "Return scores keyed by medication code, not row number. Similar-visit medication numbers and "
                    "interaction numbers refer to candidate row numbers. Similar-visit "
                    "diagnosis and procedure codes refer to the current-visit lists."
                ),
            ]
            for name in (
                "Current diagnoses",
                "Current procedures",
            ):
                parts.append(
                    name + ": " + json.dumps(sections[name], separators=(",", ":"))
                )
            parts.append(
                "Candidates: "
                + json.dumps(
                    [
                        [
                            code,
                            candidate_scores[code],
                            selected["Candidate names"].get(code),
                        ]
                        for code in candidate_order
                    ],
                    separators=(",", ":"),
                )
            )
            for name, values in selected.items():
                if name == "Candidate names":
                    continue
                if name == "Potential interactions among candidates":
                    # An adjacency list avoids repeating the first code in each
                    # pair. Each key interacts with every code in its list.
                    grouped = {}
                    for left, right in values:
                        grouped.setdefault(candidate_index[left], []).append(
                            candidate_index[right]
                        )
                    values = grouped
                parts.append(name + ": " + json.dumps(values, separators=(",", ":")))
            parts.append(
                "Interactions are an adjacency list: the candidate at each key's index interacts with "
                "the candidates at the indices in its list. "
                "Candidate names and supporting evidence may be incomplete when the input budget is reached."
            )
            return "\n".join(parts)

        if self.prompt_token_counter(render()) > self.max_prompt_tokens:
            raise ValueError(
                "current visit and candidate scores exceed the prompt token budget "
                f"(core tokens: {self.prompt_token_counter(render())}, limit: {self.max_prompt_tokens})"
            )
        # If all evidence fits, retain it without applying truncation heuristics.
        selected["Candidate names"] = dict(candidate_names)
        selected["Previous medications"] = history
        selected["Similar training visits"] = neighbours
        selected["Potential interactions among candidates"] = pairs
        all_fit = self.prompt_token_counter(render()) <= self.max_prompt_tokens
        if not all_fit:
            selected["Candidate names"] = {}
            selected["Previous medications"] = []
            selected["Similar training visits"] = []
            selected["Potential interactions among candidates"] = []

        if not all_fit:
            # Each category has a fixed share of the space beyond the core.
            # This prevents long histories from starving retrieved visits or DDIs.
            available = self.max_prompt_tokens - self.prompt_token_counter(render())
            allocations = (
                ("Candidate names", ordered_names, False, 25),
                ("Potential interactions among candidates", pairs, False, 35),
                ("Similar training visits", neighbours, False, 25),
                ("Previous medications", history, True, 15),
            )
            for name, options, recent, share in allocations:
                limit = self.prompt_token_counter(render()) + available * share // 100
                low, high = 0, len(options)
                while low < high:
                    middle = (low + high + 1) // 2
                    if name == "Candidate names":
                        selected[name] = {
                            code: candidate_names[code] for code in options[:middle]
                        }
                    else:
                        selected[name] = (
                            options[-middle:] if recent else options[:middle]
                        )
                    if self.prompt_token_counter(render()) <= limit:
                        low = middle
                    else:
                        high = middle - 1
                if name == "Candidate names":
                    selected[name] = {
                        code: candidate_names[code] for code in options[:low]
                    }
                else:
                    selected[name] = (
                        (options[-low:] if recent else options[:low]) if low else []
                    )
        fitted = render()
        tokens = self.prompt_token_counter(fitted)
        if tokens > self.max_prompt_tokens:
            raise ValueError("unable to fit prompt within token budget")
        return fitted, {
            "prompt_tokens": tokens,
            "candidate_names": [len(selected["Candidate names"]), len(candidate_names)],
            "history_visits": [len(selected["Previous medications"]), len(history)],
            "similar_visits": [
                len(selected["Similar training visits"]),
                len(neighbours),
            ],
            "ddi_pairs": [
                len(selected["Potential interactions among candidates"]),
                len(pairs),
            ],
        }

    def _batched_prompts(self, sample, values, indices, neighbours):
        """Plan all requests before generation, retaining every incident DDI."""
        all_pairs = [
            (i, j)
            for pos, i in enumerate(indices)
            for j in indices[pos + 1 :]
            if self.ddi[i, j] or self.ddi[j, i]
        ]
        diagnoses = set(self._current(sample, "conditions"))
        procedures = set(self._current(sample, "procedures"))
        history = [
            [self._describe(c) for c in visit]
            for visit in sample.get("drugs_hist", [])[:-1]
        ]
        drug_names = self.names.get("drugs", self.names)

        def build(targets, add_support):
            target_set = set(targets)
            incident = [
                (i, j) for i, j in all_pairs if i in target_set or j in target_set
            ]
            partners = {index for pair in incident for index in pair} - target_set
            ordered = list(targets) + sorted(partners)
            positions = {index: position for position, index in enumerate(ordered)}
            candidate_rows = [
                [
                    self.codes[i],
                    round(float(values[i]), 4),
                    drug_names.get(self.codes[i], self.codes[i]),
                ]
                for i in ordered
            ]
            links = {}
            for i, j in incident:
                links.setdefault(positions[i], []).append(positions[j])
            target_codes = {self.codes[i] for i in targets}
            retrieved = []
            for row in neighbours:
                drugs = [
                    c
                    for c in row.get("drugs", [])
                    if c in target_codes
                    and (
                        self.evidence_filter is None
                        or self.evidence_filter(sample, row, c)
                    )
                ]
                retrieved.append(
                    {
                        "diagnoses": sorted(
                            diagnoses & set(self._current(row, "conditions"))
                        ),
                        "procedures": sorted(
                            procedures & set(self._current(row, "procedures"))
                        ),
                        "target_medications": sorted(
                            {positions[i] for i in targets if self.codes[i] in drugs}
                        ),
                    }
                )
            chosen = {"Previous medications": [], "Similar training visits": []}

            def render():
                header = (
                    self._instructions() + " Candidate rows are "
                    "[code, score, name] and are numbered from zero. Only the first "
                    f"{len(targets)} rows are targets: return a JSON object with exactly "
                    f"{len(targets)} entries, one for every target row index from 0 to "
                    f"{len(targets) - 1}. Keys must be row numbers as strings and values "
                    'must be numeric probabilities between 0 and 1, e.g. {"0":0.8}. '
                    "Other rows are interaction context and must not be returned. A score >= 0.5 "
                    "recommends a drug. Interaction adjacency keys and values are row "
                    "numbers. All boundary interactions involving targets are included; "
                    "other interactions are not necessarily listed. Similar-visit codes "
                    "refer to the current visit; its medication numbers refer to target rows. "
                    "Similar visits and medication history may be incomplete."
                )
                sections = {
                    "Current diagnoses": [
                        self._describe(c, "conditions") for c in sorted(diagnoses)
                    ],
                    "Current procedures": [
                        self._describe(c, "procedures") for c in sorted(procedures)
                    ],
                    "Candidates": candidate_rows,
                    "Potential interactions": links,
                    **chosen,
                }
                return (
                    header
                    + "\n"
                    + "\n".join(
                        name + ": " + json.dumps(value, separators=(",", ":"))
                        for name, value in sections.items()
                    )
                )

            core = render()
            if self.prompt_token_counter(core) > self.max_prompt_tokens:
                return None
            if add_support:
                for name, options, recent in (
                    ("Similar training visits", retrieved, False),
                    ("Previous medications", history, True),
                ):
                    low, high = 0, len(options)
                    while low < high:
                        middle = (low + high + 1) // 2
                        chosen[name] = options[-middle:] if recent else options[:middle]
                        if (
                            self.prompt_token_counter(render())
                            <= self.max_prompt_tokens
                        ):
                            low = middle
                        else:
                            high = middle - 1
                    chosen[name] = (
                        (options[-low:] if recent else options[:low]) if low else []
                    )
            prompt = render()
            count = self.prompt_token_counter(prompt)
            if count > self.max_prompt_tokens:
                raise ValueError("batch prompt exceeds its input token budget")
            return {
                "prompt": prompt,
                "indices": list(targets),
                "pairs": incident,
                "prompt_tokens": count,
                "history_visits": [len(chosen["Previous medications"]), len(history)],
                "similar_visits": [
                    len(chosen["Similar training visits"]),
                    len(retrieved),
                ],
            }

        batches = []
        start = 0
        while start < len(indices):
            maximum = min(self.candidate_batch_size, len(indices) - start)
            # The core generally grows with the batch, but tokenization and
            # shared neighbours can make its size non-monotonic: check each
            # candidate size from largest down instead of binary search.
            for size in range(maximum, 0, -1):
                targets = indices[start : start + size]
                planned = build(targets, True)
                if planned is not None:
                    batches.append(planned)
                    start += size
                    break
            else:
                raise ValueError(
                    "one candidate and its complete interaction context exceed the prompt token budget"
                )
        covered = {tuple(pair) for batch in batches for pair in batch["pairs"]}
        if covered != set(all_pairs):
            raise RuntimeError("batch planning did not cover all boundary interactions")
        membership = {
            index: batch_no
            for batch_no, batch in enumerate(batches)
            for index in batch["indices"]
        }
        return batches, {
            "batch_count": len(batches),
            "prompt_tokens": [batch["prompt_tokens"] for batch in batches],
            "candidate_names": [len(indices), len(indices)],
            "resolved_candidate_names": [
                sum(
                    bool(drug_names.get(self.codes[i]))
                    and drug_names.get(self.codes[i]) != self.codes[i]
                    for i in indices
                ),
                len(indices),
            ],
            "ddi_pairs": [len(covered), len(all_pairs)],
            "cross_batch_pairs": sum(
                membership[i] != membership[j] for i, j in all_pairs
            ),
            "support_per_batch": [
                {
                    "history_visits": batch["history_visits"],
                    "similar_visits": batch["similar_visits"],
                }
                for batch in batches
            ],
        }

    def refine_with_details(self, sample: Mapping, probabilities):
        """Refine probabilities and report candidate and response coverage.

        Returns:
            A dict containing ``probabilities``, ``prompt``, ``response``,
            ``candidates``, ``updated``, and ``status``. A malformed or partial
            response leaves unmatched scores unchanged and is visible in status.
        """
        values = np.asarray(probabilities, dtype=float)
        if (
            values.shape != (len(self.codes),)
            or not np.isfinite(values).all()
            or ((values < 0) | (values > 1)).any()
        ):
            raise ValueError(
                "probabilities must be a finite vector in [0, 1] aligned with medication_codes"
            )
        result = values.copy()
        upper = values <= self.high if self.upper_inclusive else values < self.high
        indices = np.flatnonzero((values >= self.low) & upper).tolist()
        details = {
            "probabilities": result,
            "prompt": None,
            "response": None,
            "candidates": [self.codes[i] for i in indices],
            "updated": [],
            "changed": [],
            "status": "no_candidates",
        }
        if not indices:
            return details
        neighbours = self.retrieve(sample)
        if self.candidate_batch_size is None:
            prompt = self.build_prompt(sample, values, indices, neighbours)
            prompt, budget = self._fit_prompt(prompt)
            batches = [{"prompt": prompt, "indices": indices}]
        else:
            batches, budget = self._batched_prompts(sample, values, indices, neighbours)
        details["prompt_budget"] = budget
        details["prompts"] = []
        details["responses"] = []
        valid_responses = 0
        for batch in batches:
            prompt = batch["prompt"]
            text = self.generate(prompt)
            details["prompts"].append(prompt)
            details["responses"].append(text)
            try:
                # Preserve duplicate keys to reject ambiguous scores.
                cleaned = re.sub(
                    r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
                )
                start = cleaned.find("{")
                if start < 0:
                    continue
                pairs, _ = json.JSONDecoder(object_pairs_hook=lambda x: x).raw_decode(
                    cleaned[start:]
                )
            except (ValueError, TypeError, AttributeError):
                continue
            if not isinstance(pairs, list) or any(
                not isinstance(p, tuple) or len(p) != 2 for p in pairs
            ):
                continue
            valid_responses += 1
            # Batched prompts ask for local target row indices. Accept ATC
            # code keys too, but reject duplicates after normalization.
            normalized = []
            numeric_keys = [
                code
                for code, _ in pairs
                if isinstance(code, str) and code.isascii() and code.isdigit()
            ]
            expected_keys = {str(i) for i in range(len(batch["indices"]))}
            # A partial set such as {"1", "2"} could be one-based. Accept
            # indexed output only when every canonical zero-based key is present
            # and no out-of-range index was returned.
            numeric_complete = set(numeric_keys) == expected_keys
            for code, score in pairs:
                indexed = False
                if (
                    self.candidate_batch_size is not None
                    and isinstance(code, str)
                    and code.isascii()
                    and code.isdigit()
                ):
                    position = int(code)
                    if position >= len(batch["indices"]):
                        continue
                    code = self.codes[batch["indices"][position]]
                    indexed = True
                normalized.append((code, score, indexed))
            counts = {}
            for code, _, _ in normalized:
                counts[code] = counts.get(code, 0) + 1
            allowed = {self.codes[i]: i for i in batch["indices"]}
            for code, score, indexed in normalized:
                if (
                    isinstance(score, list)
                    and len(score) == 1
                    and isinstance(score[0], tuple)
                    and score[0][0] == "probability"
                ):
                    score = score[0][1]
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        pass
                if (
                    code in allowed
                    and counts[code] == 1
                    and (not indexed or numeric_complete)
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(score)
                    and 0 <= score <= 1
                ):
                    if score != result[allowed[code]]:
                        details["changed"].append(code)
                    result[allowed[code]] = score
                    details["updated"].append(code)
        details["prompt"] = details["prompts"][0]
        details["response"] = details["responses"][0]
        details["status"] = (
            "complete"
            if len(details["updated"]) == len(indices)
            else "partial"
            if details["updated"]
            else "no_updates"
            if valid_responses
            else "invalid_response"
        )
        return details


class HuggingFaceGiantMedGenerator:
    """Optional Transformers adapter for a causal chat model such as Qwen3-8B.

    Pass an already loaded model and tokenizer to control device and memory use.
    Generation parameters default to those in the reference implementation.

    Examples:
        >>> generator = HuggingFaceGiantMedGenerator(model, tokenizer)  # doctest: +SKIP
        >>> response = generator("Refine these scores...")  # doctest: +SKIP
    """

    def __init__(self, model, tokenizer, **generation_kwargs):
        self.model, self.tokenizer = model, tokenizer
        self.kwargs = {
            "max_new_tokens": 4096,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
        self.kwargs.update(generation_kwargs)

    def __call__(self, prompt):
        import torch

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, **self.kwargs)
        return self.tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()


class MediEvidenceFilter:
    """Filter retrieved drugs by MEDI indications or the query's last medication list.

    The caller prepares the two mappings from the MEDI, NDC/RxCUI and ATC
    resources used by the reference implementation. Drug ingredients must
    come from the *training* visit, not the query's target prescriptions.

    Args:
        visit_drug_ingredients: Mapping ``(patient_id, visit_id, ATC3)`` to
            ingredient RxCUIs for the retrieved training visit.
        ingredient_diagnoses: Mapping ingredient RxCUI to supported ICD-9-CM
            diagnosis codes, with the desired MEDI profile already selected.
        ancestor_match: Also allow a MEDI code to match a more specific code.
        history_rescue: Retain drugs used in the query's preceding visit.

    Examples:
        >>> evidence = MediEvidenceFilter({("p", "v", "A10A"): {"42"}},
        ...                               {"42": {"25000"}})
        >>> evidence({"conditions": [["250.00"]]},
        ...          {"patient_id": "p", "visit_id": "v"}, "A10A")
        True
    """

    def __init__(
        self,
        visit_drug_ingredients,
        ingredient_diagnoses,
        ancestor_match=False,
        history_rescue=True,
    ):
        self.visit_drug_ingredients = visit_drug_ingredients
        self.ingredient_diagnoses = ingredient_diagnoses
        self.ancestor_match = ancestor_match
        self.history_rescue = history_rescue

    @classmethod
    def from_reference_files(
        cls,
        training_samples,
        prescriptions_path,
        medi_path,
        ndc_rxcui_path,
        ndc_atc_path,
        ingredient_cache_path,
        profile="all_medi",
        ancestor_match=False,
        history_rescue=True,
        ndc_to_atc3=None,
    ):
        """Build the filter from the released reference pipeline's input files.

        Args:
            training_samples: Raw samples from the training split only.
            prescriptions_path: MIMIC-III PRESCRIPTIONS CSV or CSV.GZ.
            medi_path: MEDI.csv with RXCUI, CODE, SAB and profile columns.
            ndc_rxcui_path: Python-literal NDC to clinical RxCUI dictionary.
            ndc_atc_path: CSV with RXCUI and ATC4 columns.
            ingredient_cache_path: Reference clinical-RxCUI to ingredient JSON.
            profile: One of ``all_medi``, ``medi1`` or ``hps``.
            ancestor_match: Allow diagnosis descendants of MEDI codes.
            history_rescue: Retain drugs used in the preceding visit.
            ndc_to_atc3: Optional callable returning label ATC3 codes for an
                NDC. Supply the task's own mapper to align the index to its
                labels. By default, use the reference RxCUI-to-ATC mapping.

        Returns:
            A configured evidence filter. Only training visit prescriptions are
            read into the resulting index.
        """
        if profile not in {"all_medi", "medi1", "hps"}:
            raise ValueError("unknown MEDI profile")

        def rows(path, required):
            path = Path(path)
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = {
                    str(name).strip().upper() for name in reader.fieldnames or []
                }
                missing = set(required) - columns
                if missing:
                    raise ValueError(
                        f"{path} is missing required columns: {sorted(missing)}"
                    )
                for row in reader:
                    yield {
                        str(key).strip().upper(): value
                        for key, value in row.items()
                        if key is not None
                    }

        def yes(value):
            return str(value).strip().upper() == "TRUE"

        indications = {}
        for row in rows(
            medi_path, ("RXCUI", "CODE", "SAB", "MEDI1", "MEDI1_HPS", "MEDI2_HPS")
        ):
            if row.get("SAB", "").strip().upper() != "ICD9CM":
                continue
            accepted = (
                profile == "all_medi"
                or (profile == "medi1" and yes(row.get("MEDI1")))
                or (
                    profile == "hps"
                    and (yes(row.get("MEDI1_HPS")) or yes(row.get("MEDI2_HPS")))
                )
            )
            ingredient = str(row.get("RXCUI", "")).strip()
            diagnosis = str(row.get("CODE", "")).replace(".", "").strip().upper()
            if accepted and ingredient and len(diagnosis) >= 3:
                indications.setdefault(ingredient, set()).add(diagnosis)

        ndc_to_clinical = ast.literal_eval(
            Path(ndc_rxcui_path).read_text(encoding="utf-8-sig")
        )
        ndc_to_clinical = {
            str(k).strip(): str(v).strip() for k, v in ndc_to_clinical.items()
        }
        clinical_to_atc = {}
        for row in rows(ndc_atc_path, ("RXCUI", "ATC4")):
            clinical, atc = (
                str(row.get("RXCUI", "")).strip(),
                str(row.get("ATC4", "")).strip(),
            )
            if clinical and atc:
                clinical_to_atc.setdefault(clinical, atc[:4])
        with open(ingredient_cache_path, encoding="utf-8-sig") as stream:
            cache = json.load(stream)
        cache = cache.get("clinical_rxcuis", cache)
        visits = {(str(s["patient_id"]), str(s["visit_id"])) for s in training_samples}
        ingredients = {}
        for row in rows(prescriptions_path, ("SUBJECT_ID", "HADM_ID", "NDC")):
            patient = str(row.get("SUBJECT_ID", "")).strip()
            visit = str(row.get("HADM_ID", "")).strip()
            if (patient, visit) not in visits:
                continue
            clinical = ndc_to_clinical.get(str(row.get("NDC", "")).strip())
            ndc = str(row.get("NDC", "")).strip()
            if ndc_to_atc3 is None:
                atc = clinical_to_atc.get(clinical)
                atc_codes = [atc] if atc is not None else []
            else:
                atc_codes = ndc_to_atc3(ndc)
            values = cache.get(clinical, {}).get("ingredients", [])
            for atc in atc_codes:
                if not isinstance(atc, str) or not atc:
                    continue
                key = (patient, visit, atc)
                ingredients.setdefault(key, set()).update(
                    str(item["rxcui"])
                    for item in values
                    if item.get("rxcui") is not None
                )
        return cls(ingredients, indications, ancestor_match, history_rescue)

    def evidence_status(self, query, neighbour, drug):
        """Explain whether evidence is retained or why it was excluded."""
        history = query.get("drugs_hist", [])
        if self.history_rescue and len(history) > 1 and drug in history[-2]:
            return "history_rescue"
        key = (str(neighbour["patient_id"]), str(neighbour["visit_id"]), drug)
        if key not in self.visit_drug_ingredients:
            return "missing_mapping"
        if not self.visit_drug_ingredients[key]:
            return "missing_ingredients"
        diagnoses = [
            str(code).replace(".", "").upper()
            for code in GiantMed._current(query, "conditions")
        ]
        for ingredient in self.visit_drug_ingredients.get(key, ()):
            for code in self.ingredient_diagnoses.get(ingredient, ()):
                code = str(code).replace(".", "").upper()
                if any(
                    d == code
                    or (self.ancestor_match and len(code) >= 3 and d.startswith(code))
                    for d in diagnoses
                ):
                    return "indication_match"
        return "indication_mismatch"

    def __call__(self, query, neighbour, drug):
        return self.evidence_status(query, neighbour, drug) in {
            "history_rescue",
            "indication_match",
        }
