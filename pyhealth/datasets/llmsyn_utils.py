"""LLMSYN dataset utilities: MIMIC-III stats ingestion and Mayo Clinic RAG.

Two responsibilities:
1. Ingest raw MIMIC-III CSV files and produce the stats dict that
   LLMSYNModel uses as prior knowledge (port of mimic_ingest.py).
2. Fetch Mayo Clinic medical knowledge for a disease name, used by
   LLMSYNModel when enable_rag=True (port of mayo_scraper.py).

Usage — generate stats.json from your MIMIC-III files::

    from pyhealth.datasets.llmsyn_utils import compute_all_stats
    import json

    stats = compute_all_stats("/path/to/mimic-iii/")
    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=2)
"""

from __future__ import annotations

import time
import urllib.robotparser
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse


try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# MIMIC-III ingestion (ported from CS598 mimic_ingest.py)
# ---------------------------------------------------------------------------

def load_tables(raw_dir: str) -> Dict:
    """Load required MIMIC-III CSV tables from ``raw_dir``.

    Args:
        raw_dir: Directory containing MIMIC-III CSV files.

    Returns:
        Dict with keys ``admissions``, ``diagnoses_icd``, and optionally
        ``d_icd_diagnoses``, ``patients``.

    Raises:
        FileNotFoundError: If a required table is missing.
    """
    raw_dir = Path(raw_dir)
    table_names = {
        "admissions": True,
        "diagnoses_icd": True,
        "d_icd_diagnoses": False,
        "patients": False,
    }
    tables = {}
    for name, required in table_names.items():
        path = raw_dir / f"{name.upper()}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
        elif required:
            raise FileNotFoundError(f"Required MIMIC-III table not found: {path}")
    return tables


def compute_mortality_rate(admissions) -> float:
    """Compute overall hospital mortality rate from ADMISSIONS.HOSPITAL_EXPIRE_FLAG.

    Args:
        admissions: ADMISSIONS DataFrame.

    Returns:
        Float between 0 and 1 (e.g. 0.099 means 9.9% mortality).
    """
    return float(
        admissions["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int).mean()
    )


def compute_top_diseases(tables: Dict, top_n: int = 100) -> List[Dict]:
    """Compute the top N most frequent ICD-9 diagnoses.

    Joins DIAGNOSES_ICD with ADMISSIONS to compute per-disease admission
    counts and mortality rates. Merges with D_ICD_DIAGNOSES for descriptions
    when available.

    Args:
        tables: Dict from :func:`load_tables`.
        top_n: Number of top diseases to return (default 100, matching paper).

    Returns:
        List of dicts with keys: ``icd9_code``, ``admission_count``,
        ``deaths``, ``mortality_rate``, ``desc``.
    """
    admissions = tables["admissions"]
    diagnoses = tables["diagnoses_icd"]

    hadm_mort = admissions.set_index("HADM_ID")["HOSPITAL_EXPIRE_FLAG"]

    diag = diagnoses[["HADM_ID", "ICD9_CODE"]].dropna()
    diag = diag.rename(columns={"ICD9_CODE": "icd9_code"}).drop_duplicates()

    counts = (
        diag.groupby("icd9_code")
        .agg({"HADM_ID": "nunique"})
        .rename(columns={"HADM_ID": "admission_count"})
    )

    merged = (
        diag.set_index("HADM_ID")
        .join(hadm_mort.rename("died"), how="left")
    )
    merged["died"] = merged["died"].fillna(0).astype(int)
    mort = (
        merged.reset_index()
        .groupby("icd9_code")
        .agg({"died": "sum"})
        .rename(columns={"died": "deaths"})
    )

    stats = counts.join(mort)
    stats["mortality_rate"] = stats["deaths"] / stats["admission_count"]
    stats = stats.sort_values("admission_count", ascending=False)

    if "d_icd_diagnoses" in tables:
        icd_table = (
            tables["d_icd_diagnoses"][["ICD9_CODE", "LONG_TITLE"]]
            .rename(columns={"ICD9_CODE": "icd9_code", "LONG_TITLE": "desc"})
        )
        stats = (
            stats.reset_index()
            .merge(icd_table, on="icd9_code", how="left")
            .set_index("icd9_code")
        )
        stats["desc"] = stats["desc"].fillna("")

    return stats.head(top_n).reset_index().to_dict(orient="records")


def compute_demographics(admissions, patients=None) -> Dict:
    """Compute demographic distributions for Step 0 of the LLMSYN pipeline.

    Args:
        admissions: ADMISSIONS DataFrame.
        patients: PATIENTS DataFrame (optional — needed for age and gender).

    Returns:
        Dict with keys ``ethnicity``, ``insurance``, ``marital_status``,
        ``language``, and (if patients provided) ``gender``, ``age_mean``,
        ``age_std``.
    """
    demo: Dict = {}

    for col, key in [
        ("ETHNICITY", "ethnicity"),
        ("INSURANCE", "insurance"),
        ("MARITAL_STATUS", "marital_status"),
        ("LANGUAGE", "language"),
    ]:
        dist = admissions[col].fillna("UNKNOWN").value_counts(normalize=True)
        demo[key] = {str(k): round(float(v), 4) for k, v in dist.items()}

    if patients is not None:
        dist = patients["GENDER"].fillna("UNKNOWN").value_counts(normalize=True)
        demo["gender"] = {str(k): round(float(v), 4) for k, v in dist.items()}

        merged = admissions[["SUBJECT_ID", "ADMITTIME"]].merge(
            patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID"
        )
        merged["admittime"] = pd.to_datetime(merged["ADMITTIME"], errors="coerce")
        merged["dob"] = pd.to_datetime(merged["DOB"], errors="coerce")
        # Use year-level subtraction to avoid int64 overflow from MIMIC-III's
        # DOB shifting (~300 years back) for patients over 89.
        # A workaround to generate stats for older patients
        merged["age"] = (
            merged["admittime"].dt.year - merged["dob"].dt.year
        ).clip(0, 89)
        demo["age_mean"] = round(float(merged["age"].mean()), 1)
        demo["age_std"] = round(float(merged["age"].std()), 1)

    return demo


def compute_all_stats(raw_dir: str, top_n: int = 100) -> Dict:
    """Compute the full stats dict for LLMSYNModel from raw MIMIC-III files.

    Single entry point for preprocessing. Equivalent to running CS598's
    ``mimic_ingest.py`` script. Output can be saved as JSON and passed to
    ``LLMSYNModel(stats_path=...)``.

    Args:
        raw_dir: Path to directory containing ADMISSIONS.csv,
            DIAGNOSES_ICD.csv, D_ICD_DIAGNOSES.csv, PATIENTS.csv.
        top_n: Number of top diseases to include (default 100).

    Returns:
        Dict with keys ``mortality_rate``, ``top_diseases``, ``demographics``.

    Example::

        stats = compute_all_stats("/data/mimic-iii/")
        import json
        with open("stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    """
    tables = load_tables(raw_dir)
    return {
        "mortality_rate": compute_mortality_rate(tables["admissions"]),
        "top_diseases": compute_top_diseases(tables, top_n),
        "demographics": compute_demographics(
            tables["admissions"], tables.get("patients")
        ),
    }


# ---------------------------------------------------------------------------
# Mayo Clinic RAG (ported from CS598 mayo_scraper.py)
# ---------------------------------------------------------------------------

def _robots_allowed(url: str, user_agent: str = "llmsyn-bot") -> bool:
    parsed = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        return False
    return rp.can_fetch(user_agent, url)


def _fetch_page(url: str, timeout: int = 10, user_agent: str = "llmsyn-bot") -> str:
    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article") or soup.find(attrs={"class": "content"})
    if article:
        texts = [p.get_text(separator=" ", strip=True) for p in article.find_all("p")]
        return "\n\n".join(t for t in texts if t)
    ps = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join(ps[:20])


def _find_disease_url(search_html: str) -> Optional[str]:
    soup = BeautifulSoup(search_html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/diseases-conditions/" in href and "/symptoms-causes/" in href:
            if href.startswith("http"):
                return href
            return "https://www.mayoclinic.org" + href
    return None


def get_medical_knowledge(
    disease_name: str,
    cache_dir: str = "data/mayo",
) -> str:
    """Fetch Mayo Clinic article text for a disease name (used by RAG step).

    Called by ``LLMSYNModel`` when ``enable_rag=True``. Checks a local disk
    cache before making network requests. Returns an empty string gracefully
    if the page cannot be fetched or dependencies are unavailable.

    Args:
        disease_name: Human-readable disease name from MIMIC-III
            (e.g. ``"Acute respiratory failure"``).
        cache_dir: Directory for caching fetched pages.

    Returns:
        Plain text of the Mayo Clinic article, or ``""`` if unavailable.
    """
    if not _HAS_REQUESTS:
        return ""

    cache_path = Path(cache_dir)
    slug = disease_name.lower()[:60].replace(" ", "_").replace("/", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    cached = cache_path / f"{slug}.txt"

    if cached.exists():
        return cached.read_text(encoding="utf-8")

    if not _robots_allowed("https://www.mayoclinic.org/"):
        return ""

    try:
        search_url = (
            f"https://www.mayoclinic.org/search/search-results"
            f"?q={quote_plus(disease_name)}"
        )
        search_html = _fetch_page(search_url)
        disease_url = _find_disease_url(search_html)
        if not disease_url:
            return ""

        time.sleep(1.0)
        disease_html = _fetch_page(disease_url)
        text = _extract_main_text(disease_html)

        if text:
            cache_path.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")

        return text

    except Exception:
        return ""
