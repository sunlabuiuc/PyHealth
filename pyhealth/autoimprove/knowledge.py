"""Literature knowledge: verified evidence that orders and justifies the feature search.

A :class:`KnowledgeBase` turns sources into evidence records, keeps only the
records whose citations check out, maps each one to the task's data, and
then:

- ranks the features to try first (:meth:`KnowledgeBase.ranked_specs`), each
  with its citations and expected direction;
- gives each predictor's expected direction, "ambiguous" when sources
  disagree (:meth:`KnowledgeBase.directions`);
- writes the gap report: predictors the task already has, can build, or
  cannot build from its data (:meth:`KnowledgeBase.gap_report`), the last of
  which is a ready data request;
- lists the published performance range (:meth:`KnowledgeBase.benchmark_range`).

Sources, in order of trust:

- **curated tables** (:meth:`KnowledgeBase.add_table_evidence`): one row per
  predictor and effect, as a literature review records them;
- **documents** (:meth:`KnowledgeBase.add_table_documents`, text files,
  :meth:`KnowledgeBase.fetch_pubmed`, :meth:`KnowledgeBase.search_pubmed`):
  abstracts or full text, from which
  an LLM extracts records (:meth:`KnowledgeBase.extract`).

**LLMs invent references.** A record is kept only if its source is one of the
gathered documents, its quoted sentence appears in that document, and its
effect size (and confidence interval, if given) appear in the quote
(:meth:`KnowledgeBase.verify`). An invented citation or a made-up number is
rejected by construction.

**The literature never decides.** Priors only order the candidates; a feature
is kept only if it wins on the task's own data (the feature search).

**Privacy.** The knowledge base never sees patient data. PubMed queries are
built from the task card's text alone, and mapping sees only the names of
event types, groups and columns.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .proposers import parse_json

RATIO_TYPES = ("RR", "OR", "HR", "IRR", "aOR", "aRR", "aHR", "aIRR")
"""Effect types on a ratio scale: above 1 raises risk."""

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class Document:
    """One source text: an abstract, a full text or a table row.

    Args:
        id: Its citation id, e.g. ``"PMID:31234567"`` or ``"file:refs.csv#3"``.
        text: The text records are verified against.
        title: Title, if known.
        year: Publication year, if known.
        authors: Authors, if known.

    Examples:
        >>> Document("PMID:1", "SABA overuse doubled the risk (RR 1.93).").id
        'PMID:1'
    """

    id: str
    text: str
    title: str = ""
    year: str = ""
    authors: str = ""


@dataclass
class Evidence:
    """One published predictor and its effect, with where it was stated.

    Args:
        predictor: The predictor, e.g. "SABA overuse".
        effect: ``{"type", "value", "ci", "direction"}``, e.g.
            ``{"type": "RR", "value": 1.93, "ci": [1.24, 3.03], "direction": "+"}``.
        source: The id of the :class:`Document` that states it.
        quote: The sentence in the source that states the effect.
        operationalisation: How the source defined it, e.g. ">= 3 canisters
            per year".
        population: The study population.
        outcome: The outcome studied.
        horizon: The prediction horizon.
        curated: Whether it came from a curated table, not an LLM.
        verified: Set by :meth:`KnowledgeBase.verify`.
        reason: Why it was rejected, if it was.
        mapping: Set by :meth:`KnowledgeBase.map`: ``{"status": "have" |
            "build" | "not in data", "column" | "spec" | "note", ...}``.
        prior: Set by :meth:`KnowledgeBase.map`: strength x population
            match x data support; it only orders candidates.

    Examples:
        >>> e = Evidence("SABA overuse", {"type": "RR", "value": 1.93}, "PMID:1",
        ...              "SABA overuse doubled the risk (RR 1.93).")
        >>> Evidence.from_dict(e.to_dict()) == e
        True
    """

    predictor: str
    effect: dict[str, Any]
    source: str
    quote: str
    operationalisation: str = ""
    population: str = ""
    outcome: str = ""
    horizon: str = ""
    curated: bool = False
    verified: bool | None = None
    reason: str = ""
    mapping: dict[str, Any] = field(default_factory=dict)
    prior: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """The record as JSON-ready types."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        """Rebuilds a record saved with :meth:`to_dict`."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def strength(self) -> float:
        """Effect strength on a common scale: |log ratio|, else 0."""
        value = self.effect.get("value")
        if self.effect.get("type") in RATIO_TYPES and isinstance(value, (int, float)) \
                and value > 0:
            return abs(math.log(value))
        return 0.0


class KnowledgeBase:
    """Evidence from the literature, verified and mapped to a task's data.

    Args:
        llm: Backend for extraction and mapping, e.g.
            :class:`~pyhealth.autoimprove.ClaudeLLM`; without one, only curated
            tables are used and nothing is mapped.
        fetch: ``fetch(url) -> str`` for PubMed; defaults to ``urllib``.
            Tests replace it.
        cache_dir: Where fetched abstracts are cached.
        max_chars: Longest text sent to the LLM in one extraction request.

    Examples:
        >>> kb = KnowledgeBase()
        >>> kb.add_documents([Document("PMID:1", "Prior ED visits raised the "
        ...                            "risk of exacerbation (OR 2.4, 1.8-3.1).")])
        >>> kb.add_evidence([Evidence("prior ED visits",
        ...     {"type": "OR", "value": 2.4, "ci": [1.8, 3.1], "direction": "+"},
        ...     "PMID:1", "Prior ED visits raised the risk of exacerbation "
        ...     "(OR 2.4, 1.8-3.1).")])
        >>> [e.verified for e in kb.verify()]
        [True]
    """

    def __init__(
        self,
        llm: Callable[[str, str], str] | None = None,
        fetch: Callable[[str], str] | None = None,
        cache_dir: str | None = None,
        max_chars: int = 8000,
    ):
        self.llm = llm
        self.fetch = fetch or _fetch
        self.cache_dir = cache_dir
        self.max_chars = max_chars
        self.documents: dict[str, Document] = {}
        self.evidence: list[Evidence] = []
        self.queries: list[str] = []

    # ---------------------------------------------------------------- sources
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Adds source texts."""
        for doc in documents:
            self.documents[doc.id] = doc

    def add_evidence(self, records: Sequence[Evidence]) -> None:
        """Adds evidence records (they are checked by :meth:`verify`)."""
        self.evidence.extend(records)

    def add_text_files(self, paths: Sequence[str]) -> None:
        """Adds text or Markdown files (and PDFs, if ``pypdf`` is installed)."""
        for path in paths:
            if path.lower().endswith(".pdf"):
                text = _pdf_text(path)
                if text is None:
                    continue
            else:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            self.documents[f"file:{os.path.basename(path)}"] = Document(
                f"file:{os.path.basename(path)}", text, title=os.path.basename(path))

    def add_table_documents(
        self,
        path: str,
        text_columns: Sequence[str],
        id_column: str | None = None,
        sheet: str | int = 0,
        title_column: str | None = None,
        year_column: str | None = None,
    ) -> int:
        """Adds one document per table row, e.g. a review's abstracts.

        Args:
            path: A CSV or Excel file.
            text_columns: Columns joined into the document text, e.g.
                ``["Title", "Abstract"]``.
            id_column: A column of PMIDs (or other ids); rows are otherwise
                cited as ``file:<name>#<row>``.

        Returns:
            The number of documents added.

        Examples:
            >>> kb.add_table_documents("refs.csv", ["Title", "Abstract"], "PMID")  # doctest: +SKIP
            22
        """
        rows = _read_table(path, sheet)
        added = 0
        for i, row in enumerate(rows):
            text = "\n".join(str(row.get(c, "")) for c in text_columns
                             if _present(row.get(c)))
            if not text.strip():
                continue
            ident = row.get(id_column) if id_column else None
            doc_id = (f"PMID:{_clean_id(ident)}" if id_column and id_column.upper() ==
                      "PMID" and _present(ident) else
                      f"{id_column}:{_clean_id(ident)}" if _present(ident) else
                      f"file:{os.path.basename(path)}#{i + 1}")
            self.documents[doc_id] = Document(
                doc_id, text,
                title=str(row.get(title_column, "")) if title_column else "",
                year=str(row.get(year_column, "")) if year_column else "")
            added += 1
        return added

    def add_table_evidence(
        self,
        path: str,
        predictor_column: str,
        effect_column: str,
        population_column: str | None = None,
        reference_column: str | None = None,
        sheet: str | int = 0,
    ) -> int:
        """Adds curated evidence: one record per table row.

        The row's text is the record's source, so verification checks the
        effect against the row as the reviewer wrote it. Effects are parsed
        from text such as ``"RR 1.94 (1.61-2.32)"`` or ``"OR 2.0-4.1"``.

        Returns:
            The number of records added.

        Examples:
            >>> kb.add_table_evidence("risk_factors.csv", "Risk factor",
            ...                       "Effect estimate (95% CI)", "Population",
            ...                       "References")  # doctest: +SKIP
            16
        """
        rows = _read_table(path, sheet)
        added = 0
        for i, row in enumerate(rows):
            predictor, effect_text = row.get(predictor_column), row.get(effect_column)
            if not _present(predictor) or not _present(effect_text):
                continue
            doc_id = f"file:{os.path.basename(path)}#{i + 1}"
            text = " | ".join(f"{k}: {v}" for k, v in row.items() if _present(v))
            self.documents[doc_id] = Document(doc_id, text, title=str(predictor))
            effect = parse_effect(str(effect_text))
            population = str(row.get(population_column, "")) if population_column else ""
            if reference_column and _present(row.get(reference_column)):
                population += f" (references {row[reference_column]})"
            self.evidence.append(Evidence(
                str(predictor), effect, doc_id, str(effect_text),
                population=population.strip(), curated=True))
            added += 1
        return added

    def search_pubmed(self, card: Any, max_results: int = 20) -> list[Document]:
        """Fetches PubMed abstracts for a query built from the task card alone.

        The query uses only words from the card's population and outcome
        (no data, no feature names), and every query is kept in
        ``self.queries`` so it can be audited. Abstracts are cached.

        Examples:
            >>> kb.search_pubmed(task.task_card())  # doctest: +SKIP
        """
        query = pubmed_query(card)
        self.queries.append(query)
        search = self._get(f"{EUTILS}/esearch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}))
        ids = json.loads(search).get("esearchresult", {}).get("idlist", [])
        return self.fetch_pubmed(ids)

    def fetch_pubmed(self, pmids: Sequence[Any]) -> list[Document]:
        """Fetches the abstracts of given PubMed ids, e.g. a review's references.

        Only the ids are sent. Abstracts are cached in ``cache_dir``.

        Examples:
            >>> kb.fetch_pubmed(["33831622", "31014587"])  # doctest: +SKIP
        """
        docs = []
        for value in pmids:
            found = re.search(r"\d+", _clean_id(value)) if _present(value) else None
            if found is None:
                continue
            pmid = found.group()
            text = self._get(f"{EUTILS}/efetch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}))
            docs.append(Document(f"PMID:{pmid}", text))
        self.add_documents(docs)
        return docs

    def _get(self, url: str) -> str:
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            key = re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:]
            path = os.path.join(self.cache_dir, key + ".txt")
            if os.path.exists(path):
                with open(path) as f:
                    return f.read()
            text = self.fetch(url)
            with open(path, "w") as f:
                f.write(text)
            return text
        return self.fetch(url)

    # ------------------------------------------------------------- extracting
    def extract(self, card: Any = None) -> list[Evidence]:
        """Asks the LLM for evidence records in each document.

        The source of each record is set to its document, never taken from
        the LLM, and each record must quote the sentence that states its
        effect. Records are checked by :meth:`verify`.

        Examples:
            >>> kb.extract(task.task_card())  # doctest: +SKIP
        """
        if self.llm is None:
            raise ValueError("extract needs an llm")
        context = _card_text(card)
        found = []
        curated_sources = {e.source for e in self.evidence if e.curated}
        for doc in self.documents.values():
            if doc.id in curated_sources:
                continue
            for chunk in _chunks(doc.text, self.max_chars):
                reply = self.llm(EXTRACT_SYSTEM, f"{context}\n\nSource text:\n{chunk}")
                try:
                    data = parse_json(reply)
                except ValueError:
                    continue
                for r in data.get("records", []) or []:
                    if not isinstance(r, dict) or not r.get("predictor"):
                        continue
                    effect = r.get("effect") or {}
                    found.append(Evidence(
                        str(r["predictor"]), dict(effect), doc.id, str(r.get("quote", "")),
                        operationalisation=str(r.get("operationalisation", "")),
                        population=str(r.get("population", "")),
                        outcome=str(r.get("outcome", "")),
                        horizon=str(r.get("horizon", ""))))
        self.evidence.extend(found)
        return found

    def verify(self) -> list[Evidence]:
        """Checks every record's citation; returns the records with verdicts.

        A record is kept (``verified``) only if its source is a gathered
        document, its quote appears in that document, and its effect size,
        and confidence interval if given, appear in the quote. A direction
        that contradicts a ratio effect is corrected from the value.

        Examples:
            >>> [e.reason for e in kb.verify() if not e.verified]  # doctest: +SKIP
            ['the source is not among the gathered documents']
        """
        for e in self.evidence:
            e.verified, e.reason = _check(e, self.documents.get(e.source))
            value = e.effect.get("value")
            if e.verified and e.effect.get("type") in RATIO_TYPES and isinstance(
                    value, (int, float)):
                e.effect["direction"] = "+" if value > 1 else "-" if value < 1 else ""
        return list(self.evidence)

    # ---------------------------------------------------------------- mapping
    def map(self, features: Any = None, card: Any = None) -> list[Evidence]:
        """Maps each verified predictor to the task's data, with the LLM.

        Each predictor becomes ``have`` (an existing column), ``build`` (a
        spec, validated by ``features``) or ``not in data`` (a data
        request). The LLM also rates the population match and data support;
        the prior is strength x match x support. The LLM sees the task card,
        the feature menu and column names, never data.

        Examples:
            >>> kb.map(fs, task.task_card())  # doctest: +SKIP
        """
        if self.llm is None:
            raise ValueError("map needs an llm")
        records = self.records()
        by_predictor: dict[str, list[Evidence]] = {}
        for e in records:
            by_predictor.setdefault(_norm_name(e.predictor), []).append(e)
        if not by_predictor:
            return []
        menu = features.describe() if features is not None else ""
        columns = sorted({c for cols in (getattr(card, "columns", {}) or {}).values()
                          for c in cols})
        lines = [f"- {group[0].predictor}: {group[0].operationalisation or ''} "
                 f"({group[0].population or 'population not stated'})"
                 for group in by_predictor.values()]
        prompt = "\n\n".join(filter(None, [
            _card_text(card), menu,
            ("Existing task columns: " + ", ".join(columns)) if columns else "",
            "Published predictors:\n" + "\n".join(lines),
        ]))
        try:
            data = parse_json(self.llm(MAP_SYSTEM, prompt))
        except ValueError:
            return records
        known_columns = set(columns)
        for item in data.get("mappings", []) or []:
            group = by_predictor.get(_norm_name(str(item.get("predictor", ""))))
            if not group:
                continue
            mapping = self._validate_mapping(item, group, features, known_columns)
            match = _unit(item.get("population_match"), 1.0)
            support = _unit(item.get("data_support"), 1.0)
            strength = max(e.strength for e in group)
            for e in group:
                e.mapping = dict(mapping)
                e.prior = strength * match * support
        return records

    def _validate_mapping(self, item, group, features, known_columns) -> dict[str, Any]:
        status = item.get("status")
        note = str(item.get("note", ""))
        if status == "have":
            column = item.get("column")
            if column in known_columns:
                return {"status": "have", "column": column, "note": note}
            return {"status": "not in data", "note": f"no column {column!r}"}
        if status == "build" and features is not None:
            spec = dict(item.get("spec") or {})
            spec["rationale"] = spec.get("rationale") or f"{group[0].predictor} predicts " \
                f"the outcome in the literature"
            spec["direction"] = self.direction_of(group[0].predictor)
            spec["citations"] = sorted({e.source for e in group})
            try:
                valid = features.validate(spec)
            except ValueError as e:
                return {"status": "not in data", "note": f"spec refused: {e}"}
            return {"status": "build", "spec": valid.to_dict(), "note": note}
        return {"status": "not in data", "note": note or "no data for this predictor"}

    # -------------------------------------------------------------- outputs
    def records(self, verified: bool = True) -> list[Evidence]:
        """The records, by default only the verified ones."""
        return [e for e in self.evidence if e.verified or not verified]

    def directions(self) -> dict[str, dict[str, Any]]:
        """Each predictor's expected direction: ``"+"``, ``"-"`` or ambiguous (``""``).

        Examples:
            >>> kb.directions()["controller use"]  # doctest: +SKIP
            {'direction': '', 'sources': {'+': ['PMID:1'], '-': ['PMID:2']}}
        """
        out: dict[str, dict[str, Any]] = {}
        for e in self.records():
            entry = out.setdefault(_norm_name(e.predictor),
                                   {"predictor": e.predictor, "sources": {}})
            d = e.effect.get("direction", "")
            if d:
                entry["sources"].setdefault(d, []).append(e.source)
        for entry in out.values():
            signs = [k for k in entry["sources"] if k in ("+", "-")]
            entry["direction"] = signs[0] if len(signs) == 1 else ""
        return out

    def direction_of(self, predictor: str) -> str:
        """One predictor's expected direction ("" if ambiguous or unknown)."""
        return self.directions().get(_norm_name(predictor), {}).get("direction", "")

    def ranked_specs(self) -> list[dict[str, Any]]:
        """Buildable specs, highest prior first, with citations and direction.

        Feed them to :class:`~pyhealth.autoimprove.QueueProposer` to try the
        literature's features in order without an LLM, or show them to an
        LLM proposer.

        Examples:
            >>> QueueProposer(kb.ranked_specs())  # doctest: +SKIP
        """
        seen, out = set(), []
        for e in sorted(self.records(), key=lambda r: -r.prior):
            spec = e.mapping.get("spec") if e.mapping.get("status") == "build" else None
            if spec and spec["name"] not in seen:
                seen.add(spec["name"])
                out.append({**spec, "prior": e.prior})
        return out

    def gap_report(self) -> dict[str, list[dict[str, Any]]]:
        """Predictors the task has, can build, or cannot build from its data.

        Examples:
            >>> kb.gap_report()["not in data"]  # doctest: +SKIP
            [{'predictor': 'FeNO', 'note': 'no FeNO results in the export', ...}]
        """
        out: dict[str, list[dict[str, Any]]] = {"have": [], "build": [],
                                                "not in data": [], "unmapped": []}
        groups: dict[str, list[Evidence]] = {}
        for e in self.records():
            groups.setdefault(_norm_name(e.predictor), []).append(e)
        for group in groups.values():
            m = group[0].mapping or {}
            status = m.get("status", "unmapped")
            row = {"predictor": group[0].predictor,
                   "direction": self.direction_of(group[0].predictor),
                   "sources": sorted({e.source for e in group}),
                   "prior": max(e.prior for e in group)}
            if status == "have":
                row["column"] = m["column"]
            elif status == "build":
                row.update(name=m["spec"]["name"], definition=m["spec"]["definition"])
            if m.get("note"):
                row["note"] = m["note"]
            out[status].append(row)
        for rows in out.values():
            rows.sort(key=lambda r: -r["prior"])
        return out

    def benchmark_range(self) -> list[dict[str, Any]]:
        """Published discrimination (AUROC, C-statistic, PR-AUC) from verified records."""
        kinds = ("AUC", "AUROC", "ROC-AUC", "C", "C-STATISTIC", "PR-AUC", "AUPRC")
        return [{"source": e.source, "type": e.effect.get("type"),
                 "value": e.effect.get("value"), "population": e.population}
                for e in self.records()
                if str(e.effect.get("type", "")).upper() in kinds]

    def save(self, path: str) -> None:
        """Saves the records and documents (citations and texts, no patient data)."""
        with open(path, "w") as f:
            json.dump({"documents": [asdict(d) for d in self.documents.values()],
                       "evidence": [e.to_dict() for e in self.evidence],
                       "queries": self.queries}, f, indent=1)

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> KnowledgeBase:
        """Loads a knowledge base saved with :meth:`save`."""
        with open(path) as f:
            data = json.load(f)
        kb = cls(**kwargs)
        kb.add_documents([Document(**d) for d in data["documents"]])
        kb.evidence = [Evidence.from_dict(e) for e in data["evidence"]]
        kb.queries = list(data.get("queries", []))
        return kb


EXTRACT_SYSTEM = """\
You extract evidence about predictors of a clinical outcome from a source text. \
Report only what the text states. For each predictor with an effect size, give a \
record with: "predictor", "operationalisation" (how the text defines it), \
"population", "outcome", "horizon", "effect" ({"type": e.g. "OR", "RR", "HR", "AUC", \
"value": number, "ci": [low, high] or null, "direction": "+" if it raises risk, \
"-" if it lowers it}), and "quote": the sentence from the text that states the \
effect, copied exactly. Do not cite other papers. Reply with a ```json block: \
{"records": [...]}"""

MAP_SYSTEM = """\
You map published predictors to a prediction task's data. For each predictor \
decide: "have" (an existing task column already measures it: give "column"), \
"build" (it can be computed from the task's events with the feature operators \
shown: give "spec", a JSON feature spec without rationale or direction), or \
"not in data" (the data cannot measure it: say what is missing in "note"). Also \
rate "population_match" (0-1: how close the study population and outcome are to \
the task's) and "data_support" (0-1: lower when the task's data is only a proxy). \
Reply with a ```json block: {"mappings": [{"predictor", "status", "column" | \
"spec" | "note", "population_match", "data_support"}, ...]}"""


def pubmed_query(card: Any) -> str:
    """A PubMed query from the task card's population and outcome only.

    Examples:
        >>> from pyhealth.tasks import TaskCard
        >>> pubmed_query(TaskCard(population="children with asthma",
        ...                       outcome="asthma exacerbation"))
        '(children AND asthma) AND (asthma AND exacerbation) AND (predict* OR "risk factor*")'
    """
    def terms(item: Any) -> str:
        text = item.get("text", "") if isinstance(item, dict) else str(item or "")
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", text)
                 if w.lower() not in _STOPWORDS]
        return " AND ".join(words[:4])

    parts = [terms(getattr(card, "population", "")), terms(getattr(card, "outcome", ""))]
    parts = [f"({p})" for p in parts if p]
    return " AND ".join(parts + ['(predict* OR "risk factor*")'])


def parse_effect(text: str) -> dict[str, Any]:
    """Parses an effect such as ``"RR 1.94 (1.61-2.32)"`` or ``"OR 2.0-4.1"``.

    Examples:
        >>> parse_effect("RR 1.94 (1.61–2.32)")
        {'type': 'RR', 'value': 1.94, 'ci': [1.61, 2.32], 'direction': '+'}
        >>> parse_effect("OR 2.0–4.1")["value"]
        2.0
    """
    text = _norm(text, lower=False)
    kind = re.search(r"\b(a?(?:RR|OR|HR|IRR)|AUROC|AUC|C-statistic)\b", text,
                     re.IGNORECASE)
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    effect: dict[str, Any] = {"type": _effect_type(kind.group(1)) if kind else "",
                              "value": None, "ci": None, "direction": ""}
    if numbers:
        effect["value"] = numbers[0]
    if len(numbers) >= 3:
        effect["ci"] = numbers[1:3]
    elif len(numbers) == 2:
        effect["range"] = numbers[:2]
    if effect["type"] in RATIO_TYPES and effect["value"] is not None:
        effect["direction"] = "+" if effect["value"] > 1 else "-" if effect[
            "value"] < 1 else ""
    return effect


# ------------------------------------------------------------------ helpers
_STOPWORDS = {"the", "and", "with", "of", "in", "at", "for", "to", "a", "an", "or",
              "who", "are", "is", "least", "after", "before", "within", "still",
              "from", "on", "by", "one", "two", "years", "year", "days", "hours"}


def _check(e: Evidence, doc: Document | None) -> tuple[bool, str]:
    if doc is None:
        return False, "the source is not among the gathered documents"
    quote = _norm(e.quote)
    if len(quote) < 8 or quote not in _norm(doc.text):
        return False, "the quote is not in the source"
    value = e.effect.get("value")
    if value is None and not e.effect.get("range"):
        return False, "the record has no effect size"
    numbers = [value] if value is not None else []
    numbers += list(e.effect.get("ci") or []) + list(e.effect.get("range") or [])
    for number in numbers:
        if not _number_in(number, quote):
            return False, f"the effect size {number:g} is not in the quote"
    return True, ""


def _number_in(number: float, text: str) -> bool:
    """Whether the text states the number.

    The sign is ignored, because a dash before a number may be a range
    ("1.8-3.1") rather than a minus; a proportion may be written as a
    percentage ("AUC 84%" for 0.84).
    """
    target = abs(number)
    for digits, percent in re.findall(r"(\d+(?:\.\d+)?)\s*(%?)", text):
        x = float(digits)
        if math.isclose(target, x, rel_tol=0, abs_tol=1e-9) or (
                percent and math.isclose(100 * target, x, rel_tol=0, abs_tol=1e-6)):
            return True
    return False


def _norm(text: str, lower: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower() if lower else text
    text = text.replace("·", ".").replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()


def _effect_type(name: str) -> str:
    """Canonical effect names: RR, OR, HR, IRR (aOR etc. when adjusted), AUC."""
    upper = name.upper()
    if upper in ("AUROC", "AUC"):
        return "AUC"
    if upper.startswith("C-"):
        return "C-statistic"
    if upper.startswith("A") and upper[1:] in ("RR", "OR", "HR", "IRR"):
        return "a" + upper[1:]
    return upper


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() not in ("", "nan", "None")


def _clean_id(value: Any) -> str:
    """An id from a table cell: ``33831622.0`` or an Excel ``=HYPERLINK(url, "id")``."""
    text = str(value).strip()
    link = re.match(r'=HYPERLINK\(\s*"[^"]*"\s*[,;]\s*"([^"]*)"\s*\)$', text, re.I)
    if link:
        text = link.group(1).strip()
    return text.removesuffix(".0")


def _unit(value: Any, default: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, max(len(text), 1), size)]


def _card_text(card: Any) -> str:
    if card is None:
        return ""
    return "Task:\n" + "\n".join(f"- {name}: {item['text']}"
                                 for name, item in card.design())


def _read_table(path: str, sheet: str | int = 0) -> list[dict[str, Any]]:
    import pandas as pd

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)
    return df.to_dict("records")


def _pdf_text(path: str) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "pyhealth-autoimprove"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")
