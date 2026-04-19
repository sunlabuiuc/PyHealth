"""
MIMIC-IV Extracted Discharge Instructions and Brief Hospital Course (DIBHC) dataset.

Builds on MIMIC4NoteDataset by loading the discharge table and applying a 7-step
preprocessing pipeline to produce clean `summary`, `hospital_course`, and
`brief_hospital_course` columns.

Some code taken from the research paper for preprocessing the dataset: https://arxiv.org/pdf/2402.15422.
"""

import itertools
import logging
import os
import random
import re
import string
import warnings
from typing import Optional

import nltk
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants (ported from the preprocessing notebook)
# ---------------------------------------------------------------------------

SPECIAL_CHARS_MAPPING_TO_ASCII = {
    u'\u0091': '\'',
    u'\u0092': '\'',
    u'\u0093': '\"',
    u'\u0094': '-',
    u'\u0096': '-',
    u'\u0097': '-',
    '·': '-',
    '¨': '-',
    u'\u0095': '\n',
}

ENCODE_STRINGS_DURING_PREPROCESSING = {
    'Dr.': '@D@',
}

SERVICE_MAPPING = {
    'MED': 'MEDICINE',
    'VSU': 'SURGERY',
    'OBS': 'OBSTETRICS/GYNECOLOGY',
    'ORT': 'ORTHOPAEDICS',
    'General Surgery': 'SURGERY',
    'Biologic': 'BIOLOGIC',
    'Biologic Service': 'BIOLOGIC',
    'GYN': 'OBSTETRICS/GYNECOLOGY',
    'Biologics': 'BIOLOGIC',
    'Neurology': 'NEUROLOGY',
    'ACS': 'SURGERY',
    'Biologics Service': 'BIOLOGIC',
    'NEURO': 'NEUROLOGY',
    'PSU': 'SURGERY',
    'TRA': 'SURGERY',
    'OP': 'SURGERY',
    'Neuromedicine': 'NEUROLOGY',
    'ENT': 'OTOLARYNGOLOGY',
    'OBSTERTRIC/GYNECOLOGY': 'OBSTETRICS/GYNECOLOGY',
    'OB service': 'OBSTETRICS/GYNECOLOGY',
    'Vascular Service': 'SURGERY',
    'OB-GYN': 'OBSTETRICS/GYNECOLOGY',
    'Vascular': 'SURGERY',
    'Surgical': 'SURGERY',
    'Ob-GYN': 'OBSTETRICS/GYNECOLOGY',
    'General surgery': 'SURGERY',
    'TRANSPLANT ': 'SURGERY',
    'ACS Service': 'SURGERY',
    'Thoracic Surgery Service': 'SURGERY',
    'Otolaryngology': 'OTOLARYNGOLOGY',
    'GU': 'UROLOGY',
    'CSU': 'SURGERY',
    'NME': 'NEUROLOGY',
    'BIOLOGICS': 'BIOLOGIC',
    'GENERAL SURGERY': 'SURGERY',
    'SURGICAL ONCOLOGY': 'SURGERY',
    'Surgical Oncology': 'SURGERY',
    '': 'UNKNOWN',
}

# Compiled regexes
_re_whitespace = re.compile(r'\s+', re.MULTILINE)
_re_multiple_whitespace = re.compile(r'  +', re.MULTILINE)
_re_paragraph = re.compile(r'\n{2,}', re.MULTILINE)
_re_line_punctuation = re.compile(
    r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|"|"|-|_)+$',
    re.MULTILINE,
)
_re_line_punctuation_wo_fs = re.compile(
    r'^(?:!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|\{|\||\}|\||~|»|«|"|"|-|_)+$',
    re.MULTILINE,
)
_re_line_punctuation_wo_underscore = re.compile(
    r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|"|"|-)+$',
    re.MULTILINE,
)
_re_ds_punctuation_wo_underscore = re.compile(
    r'^(?:\.|!|\"|#|\$|%|&|\'|\(|\)|\*|\+|,|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|`|\{|\||\}|\||~|»|«|"|"|-)+',
)
_re_fullstop = re.compile(r'^(?:\.)+$', re.MULTILINE)
_re_newline_in_text = re.compile(r'(?<=\w)\n(?=\w)', re.MULTILINE)
_re_incomplete_sentence_at_end = re.compile(r'(?<=\.)[^\.]+$', re.DOTALL)
_re_more_than_double_newline = re.compile(r'\n{3,}', re.MULTILINE)
_re_no_text = re.compile(r'^[^a-z_\n]+$', re.IGNORECASE | re.MULTILINE)

_re_item_element = r'(?:-|\. |\*|•|\d+ |\d+\.|\d\)|\(\d+\)|\d\)\.|o |# )'
_re_heading_general = r'[^\.\:\n]*(?::\n{1,2}|\?\n{1,2}|[^,]\n)'
_re_item_element_line_start = re.compile(r'^' + _re_item_element, re.MULTILINE)

ITEMIZE_ELEMENTS = [r'-', r'\. ', r'\*', r'•', r'\d+ ', r'\d+\.', r'\d\)', r'\(\d+\)', r'\d\)\.', r'o ', r'# ']

UNNECESSARY_SUMMARY_PREFIXES = {
    'template separator': re.compile(r'^={5,40}', re.MULTILINE),
    'template heading': re.compile(
        r'\A(?:Patient |CCU )?Discharge (?:Instructions|Worksheet):?\s*',
        re.IGNORECASE | re.DOTALL,
    ),
    'salutations': re.compile(
        r'\A(?:___,|(?:Dear|Hello|Hi|Ms|Mrs|Miss|Mr|Dr)(?: Ms| Mrs| Miss| Mr| Dr)?\.{0,1} (?:___)?(?: and family| family)?(?:,|\.|:|;| ){0,3}|)\s*',
        re.IGNORECASE,
    ),
    'thank you': re.compile(
        r'\A(?:[^\.!:;]*\.){0,1}[^\.!:;]*thank you[^\.!:;]*(?:\.|!|:|;)\s*',
        re.IGNORECASE | re.DOTALL,
    ),
    'pleasure': re.compile(
        r'\A(?:[^\.!:;]*\.){0,2}[^\.!:;]*(?:pleasure|priviledge|privilege)[^\.!:;]*(?:\.|!|:|;)\s*',
        re.IGNORECASE | re.DOTALL,
    ),
}

_WHY_WHAT_NEXT_HEADINGS = (
    r'^-{0,4}[^\S\r\n]{0,4}_{0,4}[^\S\r\n]{0,4}'
    r'(?:why .* admitted|why .* hospital|what brought .* hospital|why .* here|where .* hospital|why .* hospitalized|'
    r'what was done|what .* hospital|was I .* hospital|when you .* hospital|what .* here|what .* admitted|what .* for you|what .* hospitalization|what .* stay|what happened .* ___|while .* here|'
    r'what should .* next|what should .* hospital|what .* for me|when .* leave|what .* leave|what .* home|when .* home|what should .* leaving|what .* to do|'
    r'when .* hospital|when .* come back|what .* discharge|what .* discharged)'
    r'(?:\?|:|\?:)?\n{1,2}'
)
WHY_WHAT_NEXT_HEADINGS_DASHED_LIST = re.compile(
    _WHY_WHAT_NEXT_HEADINGS + r'-', re.MULTILINE | re.IGNORECASE
)
_subheading_regex = re.compile(_WHY_WHAT_NEXT_HEADINGS, re.MULTILINE | re.IGNORECASE)

_YOU_SUFFIXES = [
    'were admitted', 'were here', 'were followed', 'were started', 'were found', 'were maintained',
    'were able', 'were seen', 'were treated', 'were given', 'were told', 'were advised', 'were asked',
    'were instructed', 'were recommended', 'were initially evaluated', 'were hospitalized',
    'were complaining', 'were discharged', 'were also', 'were at',
    'will not need', 'will need to follow', 'will need to', 'will start this'
    'should hear', 'should follow',
    'have recovered', 'have healed',
    'are now ready', 'unfortunately developed', 'had chest pain', 'suffered', 'hit your head',
    'vomited', 'can expect to see', 'tolerated the procedure',
]
SIMPLE_DEIDENTIFICATION_PATTERNS = [
    ('You ', re.compile(
        r'(?:^|\. )___ (?=' + '|'.join(_YOU_SUFFIXES) + r')',
        re.MULTILINE | re.IGNORECASE,
    )),
    (' you ', re.compile(
        r'(?!' + ENCODE_STRINGS_DURING_PREPROCESSING['Dr.'] + r') ___ (?=' + '|'.join(_YOU_SUFFIXES) + r')',
        re.MULTILINE | re.IGNORECASE,
    )),
    (' you', re.compile(
        r'(?:(?<=giving)|(?<=giving thank)|(?<=giving we wish)|(?<=giving scheduled)|(?<=giving will call)|(?<=we assessed)) ___',
        re.MULTILINE | re.IGNORECASE,
    )),
    (' your ', re.compile(r' ___ (?=discharge|admission)', re.MULTILINE | re.IGNORECASE)),
    (' your ', re.compile(
        r'(?=directs all the other parts of|the brain is the part of|see occasional blood in) ___ ',
        re.MULTILINE | re.IGNORECASE,
    )),
]


def _create_heading_rs(heading: str) -> list[str]:
    """Create regex patterns for matching section headings.

    Args:
        heading: The section heading text (e.g., 'follow(-| ||)(?:up)? instructions').

    Returns:
        A list of two regex patterns for matching the heading with or without
        a preceding line break.
    """
    return [heading + r':', r'(?:^|\n)' + heading + '\n']


_SUFFIXES_DICT = {
    "followup headings": _create_heading_rs(r'follow(?:-| ||)(?:up)? instructions'),
    "followup sentences": [
        r'(?:you should|you have|you will|please)[^\.]{0,50} follow(?:-| ||)(?:up)?',
        r'(?:call|see|visit|attend)[^\.]{0,200} follow(?:-| ||)(?:up)?',
        r'follow(?:-| ||)(?:up)? with[^\.]{0,50} (?:primary care|pcp|doctor|neurologist|cardiologist)',
        r'you will [^\.]{10,50} (?:primary care|pcp|doctor|neurologist|cardiologist)',
        r'The number for [^\.]{10,200} is listed below',
    ],
    "discharge headings": (
        _create_heading_rs(r'discharge instructions')
        + _create_heading_rs(r'[^\.]{0,200} surgery discharge instructions')
    ),
    "discharge sentences": [
        r'Please follow [^\.]{0,30}discharge instructions',
        r'(?:cleared|ready for)[^\.]{0,50} discharge',
        r'(?:are|were|being|will be) (?:discharge|sending)[^\.]{0,200} (?:home|rehab|facility|assisted|house)',
        r'(?:note|take)[^\.]{0,100} discharge (?:instruction|paperwork)',
        r'Below are your discharge instructions regarding',
    ],
    "farewell pleasure": [r'It [^\.]{3,20} pleasure', r'was a pleasure'],
    "farewell priviledge": [r'It [^\.]{3,20} priviled?ge'],
    "farewell wish you": [r'wish(?:ing)? you', r'Best wishes', r'wish(?:ing)? [^\.]{0,20} luck'],
    "farewell general": [
        r'Sincerely', r'Warm regards', r'Thank you',
        r'Your[^\.]{0,10} care team', r'Your[^\.]{0,10} (?:doctor|PCP)',
    ],
    "activity headings": (
        _create_heading_rs(r'Activity')
        + _create_heading_rs(r'Activity and [^\.]{4,20}')
    ),
    "ama sentences": [r'You [^\.]{0,60}decided to leave the hospital'],
    "appointments sentences": [
        r'(?:keep|follow|attend|go to|continue)[^\.]{1,100} (?:appointment|follow(?:-| ||)up)',
        r'(?:appointment|follow(?:-| ||)up)[^\.]{1,100} (?:arranged|scheduled|made)',
        r'(?:contact|call|in touch)[^\.]{1,100} (?:appointment|follow(?:-| ||)up)',
        r'have[^\.]{0,100} (?:appointments?|follow(?:-| ||)up) with ',
        r'see[^\.]{0,100} (?:appointments?|follow(?:-| ||)up) below',
        r'provide[^\.]{0,100} phone number',
        r'getting an appointment for you',
    ],
    "case manager sentences": [
        r'contact[^\.]{0,100} case manager',
        r'case manager[^\.]{0,100} (?:contact|call|in touch|give|arrange|schedule|make)',
    ],
    "diet headings": (
        _create_heading_rs(r'Diet')
        + _create_heading_rs(r'Diet and [^\.]{4,20}')
    ),
    "forward info sentences": [r'forward[^\.]{0,100} (?:information|info|paper(?: |-||)work)'],
    "instructions sentences": [
        r'Please (?:review|follow|check)[^\.]{1,100} instructions?',
        r'should discuss this further with ',
    ],
    "medication headings": (
        _create_heading_rs(r'(?:medications?|medicines?|antibiotics?|pills?)')
        + _create_heading_rs(r'(?:medications?|medicines?|antibiotics?|pills?) ?(?:changes|list|as follows|on discharge|for [^\.]{0,80})')
        + _create_heading_rs(r'(?:take|administer|give|prescribe|order|direct|start|continue)[^\.]{0,100} doses')
        + _create_heading_rs(r'schedule for[^\.]{0,100}')
        + [r'(?:take|administer|give|prescribe|order|direct|start|continue)[^\.]{0,50} (?:medications?|medicines?|antibiotics?|pills?)[^\.]{0,100} (?:prescribe|list|as follows)']
    ),
    "medication sentences": [
        r'(?:following|not make|not make any|not make a|no) change[^\.]{0,100} (?:medications?|medicines?|antibiotics?|pills?)',
        r'(?:medications?|medicines?|antibiotics?|pills?)[^\.]{0,100} (?:prescribed|directed|ordered|listed below|change)',
        r'(?:continue|resume|take)[^\.]{0,100} (?:all|other|your)[^\.]{0,50} (?:medications?|medicines?|antibiotics?|pills?)',
        r'see[^\.]{0,100} list[^\.]{0,100} (?:medications?|medicines?|antibiotics?|pills?)',
        r'were given[^\.]{0,50} (?:presecription|prescription)',
    ],
    "medication items": [r'^(?:please)? (?:start|stop|continue) take'],
    "questions sentences": [
        r'call [^\.]{1,200} (?:questions|question|concerns|concern|before leave)',
        r'If [^\.]{1,200} (?:questions|question|concerns|concern)',
        r'Please do not hesitate to contact us',
    ],
    "home sentences": [r'(?:ready|when|safe)[^\.]{0,30} home'],
    "surgery procedure headings": (
        _create_heading_rs(r'Surgery[^\.]{0,10}Procedure')
        + _create_heading_rs(r'Surgery')
        + _create_heading_rs(r'Procedure')
        + _create_heading_rs(r'Your Surgery')
        + _create_heading_rs(r'Your Procedure')
        + _create_heading_rs(r'Recent Surgery')
        + _create_heading_rs(r'Recent Procedure')
    ),
    "warning signs sentences": [
        r'please seek medical (?:care|attention)',
        r'to[^\.]{0,100} (?:ED(?:\.|,|;| )|ER(?:\.|,|;| )|Emergency Department|Emergency Room)',
        r'(?:call|contact|experience|develop) [^\.]{1,200} following',
        r'(?:call|contact)[^\.]{0,100} (?:develop|experience|concerning symptom|if weight|weight goes|doctor|physician|surgeon|provider|nurse|clinic|office|neurologist|cardiologist|hospital)',
        r'Please (?:call|contact|seek)[^\.]{0,200} if',
        r'If[^\.]{0,100} (?:develop|experience|concerning symptoms|worse)',
    ],
    "wound care headings": (
        _create_heading_rs(r'Wound Care')
        + _create_heading_rs(r'Wound Care Instructions?')
    ),
    "wound care sentences": [
        r'GENERAL INSTRUCTIONS WOUND CARE You or a family member should inspect',
        r'GENERAL INSTRUCTIONS WOUND CARE\nYou or a family member should inspect',
        r'Please shower daily including washing incisions gently with mild soap[^\.]{0,10} no baths or swimming[^\.]{0,10} and look at your incisions',
        r'wash incisions gently with mild soap[^\.]{0,10} no baths or swimming[^\.]{0,10} look at your incisions daily',
        r'Do not smoke\. No pulling up, lifting more than 10 lbs\., or excessive bending or twisting\.',
        r'Have a friend/family member check your incision daily for signs of infection',
    ],
    "other headings": (
        list(itertools.chain(*[
            _create_heading_rs(h) for h in [
                r'Anticoagulation', r'Pain control', r'Prevena dressing instructions',
                r'your bowels', r'Dressings', r'Pain management', r'Incision care',
                r'What to expect', r'orthopaedic surgery', r'Physical Therapy',
                r'Treatment Frequency', r'IMPORTANT PATIENT DETAILS',
                r'IMPORTANT PATIENT DETAILS 1\.',
            ]
        ]))
        + _create_heading_rs(r'Please see below[^\.]{1,50} hospitalization')
        + _create_heading_rs(r'[^\.]{0,50} in the hospital we')
        + [r'CRITICAL THAT YOU QUIT SMOKING']
    ),
    "stroke template sentences": [
        r'a condition (?:where|in which) a blood vessel providing oxygen and nutrients to the brain (?:is blocked|bleed)',
        r'The brain is the part of your body that controls? and directs all the other parts of your body',
        r'damage to the brain[^\.]{0,200} can result in a variety of symptoms',
        r'can have many different causes, so we assessed you for medical conditions',
        r'In order to prevent future strokes,? we plan to modify those risk factors',
    ],
    "stone template sentences": [
        r'You can expect to see occasional blood in your urine and to possibly experience some urgency and frequency',
        r'You can expect to see blood in your urine for at least 1 week and to experience some pain with urination, urgency and frequency',
        r'The kidney stone may or may not [^\.]{0,30} AND\/or there may fragments\/others still in the process of passing',
        r'You may experiences? some pain associated with spasm? of your ureter',
    ],
    "aortic graft template sentences": [
        r'You tolerated the procedure well and are now ready to be discharged from the hospital',
        r'Please follow the recommendations below to ensure a speedy and uneventful recovery',
        r'Division of Vascular and Endovascular Surgery[^\.]{0,200}please note',
    ],
    "caotic endarterectomy template sentences": [
        r'You tolerated the procedure well and are now ready to be discharged from the hospital',
        r'You are doing well and are now ready to be discharged from the hospital',
        r'Please follow the recommendations below to ensure a speedy and uneventful recovery',
    ],
    "neck surgery template sentences": [
        r'Rest is important and will help you feel better\. Walking is also important\. It will help prevent problems',
    ],
    "TAVR template sentences": [
        r'If you stop these medications or miss[^\.]{0,30}, you risk causing a blood clot forming on your new valve',
        r'These medications help to prevent blood clots from forming on the new valve',
    ],
    "appendicitis template sentences": [r' preparing for discharge home with the following instructions'],
    "bowel obstruction template sentences": [
        r'You may return home to finish your recovery\. Please monitor'
        r'may or may not have had a bowel movement prior to[^\.]{0,20} discharge which is acceptable[^\.]{0,5} however it is important that[^\.]{0,30} have a bowel movement in',
    ],
    "small bowel obstruction template sentences": [
        r'You have tolerated a regular diet, are passing gas [^\.]{0,30} (?:not taking any pain medications|pain is controlled with pain medications by mouth)\.',
    ],
    "general headings": [
        r'^\w' + _re_heading_general + _re_item_element,
        r'(?<=\. )' + _re_heading_general + _re_item_element,
    ],
    "at least two items": [
        r'^(?:' + item + r'(?:[^\n]+\n){1,2}\n?){2,}' for item in ITEMIZE_ELEMENTS
    ],
}

RE_SUFFIXES_DICT = {
    name: re.compile('|'.join(patterns), re.IGNORECASE | re.MULTILINE)
    for name, patterns in _SUFFIXES_DICT.items()
}

_re_ds = re.compile(r"Discharge Instructions:\n", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MIMIC4NoteExtDIBHCDataset(BaseDataset):
    """
    MIMIC-IV Extracted Discharge Instructions and Brief Hospital Course (DIBHC) dataset.

    Loads the MIMIC-IV discharge notes table and applies a 7-step preprocessing
    pipeline to produce three cleaned text columns:

    - ``summary``             – cleaned discharge-instruction section.
    - ``hospital_course``     – raw text before the "Discharge Instructions:" split.
    - ``brief_hospital_course`` – extracted and normalised "Brief Hospital Course" section.

    The pipeline mirrors the preprocessing described in the original notebook
    ``creating_datasets.py`` and performs the following steps:

    1. Replace non-ASCII special characters with ASCII equivalents.
    2. Split on ``"Discharge Instructions:"`` and filter notes that lack it.
    3. Truncate unnecessary prefixes (salutations, template headers, etc.).
    4. Remove static boilerplate patterns and apply light de-identification.
    5. Truncate unnecessary suffixes (follow-up, medication lists, etc.).
    6. Drop summaries that fail minimum quality thresholds (length, sentence
       count, double-newline density, de-identification density).
    7. Drop records with missing or very short brief hospital courses.

    Args:
        root: Root directory of the MIMIC-IV Notes data.
        dataset_name: Name for this dataset instance.
        config_path: Optional path to a YAML config file.
        cache_dir: Optional directory for caching intermediate data.
        min_chars: Minimum character length for a valid summary (default 350).
        max_double_newlines: Maximum number of ``\\n\\n`` sequences allowed in a
            summary (default 5).
        min_sentences: Minimum number of sentences required in a summary
            (default 3).
        num_words_per_deidentified: Ratio threshold for ``___`` tokens —
            summaries with more than ``len(words) / num_words_per_deidentified``
            occurrences of ``___`` are dropped (default 10).
        min_chars_bhc: Minimum character length for a valid brief hospital
            course (default 500).
        **kwargs: Additional keyword arguments forwarded to :class:`BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import MIMIC4NoteExtDIBHCDataset
        >>> dataset = MIMIC4NoteExtDIBHCDataset(
        ...     root="/path/to/mimic-iv-note/2.2",
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "mimic4_note_ext_dibhc",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        # Quality-filter thresholds (all overridable)
        min_chars: int = 350,
        max_double_newlines: int = 5,
        min_sentences: int = 3,
        num_words_per_deidentified: int = 10,
        min_chars_bhc: int = 500,
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_noteextdibhc.yaml"
            )
            logger.info(f"Using default note config: {config_path}")

        # The DIBHC dataset is always built from the discharge table.
        tables = ["discharge"]
        warnings.warn(
            "Events from the discharge table only have date timestamps "
            "(no specific time). This may affect temporal ordering of events.",
            UserWarning,
        )

        # Store thresholds before calling super().__init__ so that load_data()
        # can access them if the parent calls it during initialisation.
        self.min_chars = min_chars
        self.max_double_newlines = max_double_newlines
        self.min_sentences = min_sentences
        self.num_words_per_deidentified = num_words_per_deidentified
        self.min_chars_bhc = min_chars_bhc

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            cache_dir=cache_dir,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full 7-step DIBHC preprocessing pipeline.

        Executes all preprocessing steps in sequence: special character
        replacement, discharge instructions split, hospital course extraction,
        prefix/suffix removal, boilerplate pattern removal, quality filtering,
        and hospital course validation. Each step progressively refines the
        data and reduces the row count based on quality criteria.

        Example:
            >>> dataset = MIMIC4NoteExtDIBHCDataset(root="/path/to/mimic-iv")
            >>> df_raw = dataset.load_raw_data()
            >>> df_processed = dataset.preprocess(df_raw)
            >>> print(df_processed[['summary', 'brief_hospital_course']].head())

        Args:
            df: Raw discharge-notes DataFrame with at least a 'text' column.

        Returns:
            Filtered DataFrame with additional columns 'summary',
            'hospital_course', and 'brief_hospital_course'. Total row count
            is reduced based on applied filters.
        """
        df = df.copy()
        df = self._step0_special_chars(df)
        df = self._step1_split_on_discharge_instructions(df)
        df = self._step2_encode_and_extract_hc(df)
        df = self._step3_truncate_prefixes(df)
        df = self._step4_remove_static_patterns(df)
        df = self._step5_truncate_suffixes(df)
        df = self._step6_quality_filter(df)
        df = self._step7_filter_hospital_course(df)
        return df

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    @staticmethod
    def _step0_special_chars(df: pd.DataFrame) -> pd.DataFrame:
        """Replace special Unicode characters with ASCII equivalents.

        Strips leading/trailing whitespace and replaces non-ASCII characters
        (e.g., curly quotes, dashes) with standard ASCII versions using the
        module-level SPECIAL_CHARS_MAPPING_TO_ASCII dictionary.

        Args:
            df: DataFrame with a 'text' column containing raw note text.

        Returns:
            DataFrame with cleaned 'text' column.
        """
        logger.info("Step 0: Replace special characters with ASCII equivalents.")
        df['text'] = df['text'].str.strip()
        df['text'] = df['text'].replace(SPECIAL_CHARS_MAPPING_TO_ASCII, regex=True)
        return df

    def _step1_split_on_discharge_instructions(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Split notes on 'Discharge Instructions:' and filter.

        Locates the "Discharge Instructions:" marker and splits each note into
        two parts: hospital_course (before the marker) and summary (after).
        Removes notes lacking this marker. Logs statistics at INFO level.

        Args:
            df: DataFrame with a 'text' column containing note text.

        Returns:
            DataFrame with 'hospital_course' and 'summary' columns added.
            Rows without the marker are removed.
        """
        logger.info("Step 1: Split on 'Discharge Instructions:' and filter.")
        old_len = len(df)
        df = df[df['text'].str.contains(_re_ds, regex=True)].copy()
        split_df = df['text'].str.split(_re_ds, n=1, expand=True)
        df['hospital_course'] = split_df[0].str.strip()
        df['summary'] = split_df[1].str.strip()
        logger.info(
            f"Removed {old_len - len(df)} / {old_len} notes without "
            f"'Discharge Instructions:'"
        )
        return df

    @staticmethod
    def _step2_encode_and_extract_hc(df: pd.DataFrame) -> pd.DataFrame:
        """Encode special strings and extract Brief Hospital Course section.

        Temporarily encodes abbreviations like 'Dr.' to prevent sentence
        tokenization errors. Extracts the Brief Hospital Course section using
        the _extract_hc helper. Filters out rows with empty or very short
        summaries.

        Args:
            df: DataFrame with 'summary' and 'hospital_course' columns.

        Returns:
            DataFrame with 'brief_hospital_course' column added. Rows with
            insufficient summary length are removed.
        """
        logger.info("Step 2: Encode special strings and extract brief hospital course.")
        for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
            df['summary'] = df['summary'].str.replace(k, v, regex=False)
        df['brief_hospital_course'] = df['hospital_course'].apply(
            MIMIC4NoteExtDIBHCDataset._extract_hc
        )
        df = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        return df

    def _step3_truncate_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary prefixes (headers, salutations, etc.) from summaries.

        Applies a series of regex-based filters to remove common boilerplate
        patterns such as template separators, discharge headings, and
        salutations. Normalizes whitespace and punctuation. Logs changes at
        DEBUG level.

        Args:
            df: DataFrame with a 'summary' column.

        Returns:
            DataFrame with cleaned 'summary' column. Rows with insufficient
            content are removed.
        """
        logger.info("Step 3: Truncate unnecessary prefixes of summaries.")
        df['summary'] = df['summary'].apply(
            lambda s: _re_multiple_whitespace.sub(' ', s)
        )
        df['summary'] = df['summary'].apply(
            lambda s: _re_line_punctuation_wo_underscore.sub('', s)
        )
        postprocess = lambda s: _re_ds_punctuation_wo_underscore.sub('', s.strip())
        df['summary'] = df['summary'].apply(postprocess)
        df = self._remove_regex_dict(
            df,
            UNNECESSARY_SUMMARY_PREFIXES,
            keep=1,
            postprocess=postprocess,
        )
        df = self._remove_empty_and_short_summaries(df)
        return df

    @staticmethod
    def _step4_remove_static_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Remove boilerplate patterns and apply light de-identification.

        Strips lines, removes punctuation-only lines, collapses whitespace,
        converts structured lists to prose, removes internal newlines from
        continuous text, and applies pattern-based de-identification to
        replace placeholders (___) with contextual pronouns.

        Args:
            df: DataFrame with a 'summary' column.

        Returns:
            DataFrame with cleaned and de-identified 'summary' column. Rows
            with insufficient content are removed.
        """
        logger.info("Step 4: Remove static patterns from summaries.")

        # Strip each line
        df['summary'] = df['summary'].apply(
            lambda s: '\n'.join(x.strip() for x in s.split('\n'))
        )
        # Remove lines consisting solely of punctuation
        df['summary'] = df['summary'].apply(
            lambda s: _re_line_punctuation_wo_fs.sub('', s)
        )
        df['summary'] = df['summary'].apply(
            lambda s: _re_fullstop.sub('', s)
        )
        # Collapse multiple spaces
        df['summary'] = df['summary'].apply(
            lambda s: _re_multiple_whitespace.sub(' ', s)
        )

        # Convert "Why admitted / What was done / What next" list blocks to prose
        df['summary'] = MIMIC4NoteExtDIBHCDataset._change_why_what_next_pattern_to_text(
            df['summary']
        )
        df['summary'] = df['summary'].apply(
            lambda s: _subheading_regex.sub('\n', s)
        )

        # Remove newlines within continuous prose
        df['summary'] = df['summary'].apply(
            lambda s: _re_newline_in_text.sub(' ', s)
        )
        df['summary'] = df['summary'].apply(
            lambda s: _re_multiple_whitespace.sub(' ', s)
        )

        # Light de-identification: replace ___ with contextual pronouns where safe
        for replacement, regex in SIMPLE_DEIDENTIFICATION_PATTERNS:
            df['summary'] = df['summary'].apply(lambda s: re.sub(regex, replacement, s))

        df = MIMIC4NoteExtDIBHCDataset._remove_empty_and_short_summaries(df)
        return df

    def _step5_truncate_suffixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unnecessary suffixes (follow-ups, meds lists, etc.) from summaries.

        Uses RE_SUFFIXES_DICT regex patterns to match and remove common trailing
        content such as follow-up instructions, medication lists, appointment
        details, and warning sign sections. Drops trailing incomplete sentences
        and removes lines with only symbols. Logs changes at DEBUG level.

        Args:
            df: DataFrame with a 'summary' column.

        Returns:
            DataFrame with cleaned 'summary' column. Rows with insufficient
            content are removed.
        """
        logger.info("Step 5: Truncate unnecessary suffixes of summaries.")
        postprocess = lambda s: _re_multiple_whitespace.sub(' ', s.strip())
        df['summary'] = df['summary'].apply(postprocess)
        df = self._remove_regex_dict(df, RE_SUFFIXES_DICT, postprocess, keep=0)
        # Drop trailing incomplete sentences
        df['summary'] = df['summary'].apply(
            lambda s: _re_incomplete_sentence_at_end.split(s, 1)[0]
        )
        # Remove lines with no text and leading itemise symbols
        df['summary'] = df['summary'].apply(lambda s: _re_no_text.sub('', s))
        df['summary'] = df['summary'].apply(
            lambda s: _re_item_element_line_start.sub('', s)
        )
        df = self._remove_empty_and_short_summaries(df)
        return df

    def _step6_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimum quality thresholds to filter low-quality summaries.

        Enforces multiple quality criteria: minimum character count, minimum
        number of sentences, maximum density of double-newlines, and maximum
        density of de-identification placeholders (___). Logs filter outcomes
        at INFO and DEBUG levels. Uses NLTK for sentence tokenization.

        Args:
            df: DataFrame with a 'summary' column.

        Returns:
            DataFrame with only high-quality summaries. Encoded special strings
            (e.g., @D@ for 'Dr.') are decoded back to original form.
        """
        logger.info("Step 6: Apply quality filters.")
        nltk.download('punkt_tab', quiet=True)

        old_len = len(df)
        df = df[df['summary'].map(len) >= self.min_chars]
        logger.info(
            f"  Removed {old_len - len(df)} summaries with "
            f"< {self.min_chars} characters."
        )

        old_len = len(df)
        df['sentences'] = df['summary'].apply(lambda s: list(nltk.sent_tokenize(s)))
        df = df[df['sentences'].map(len) >= self.min_sentences]
        logger.info(
            f"  Removed {old_len - len(df)} summaries with "
            f"< {self.min_sentences} sentences."
        )

        old_len = len(df)
        df = df[
            df['summary'].map(lambda s: s.count('\n\n')) <= self.max_double_newlines
        ]
        logger.info(
            f"  Removed {old_len - len(df)} summaries with "
            f"> {self.max_double_newlines} double newlines."
        )

        # Flatten sentences back to whitespace-separated text
        df['summary'] = df['sentences'].apply(
            lambda s: _re_whitespace.sub(' ', ' '.join(s))
        )
        df.drop(columns=['sentences'], inplace=True)

        # Decode encoded special strings
        for k, v in ENCODE_STRINGS_DURING_PREPROCESSING.items():
            df['summary'] = df['summary'].str.replace(v, k, regex=False)

        # Filter by de-identification density
        df['num_deidentified'] = df['summary'].apply(lambda s: s.count('___'))
        old_len = len(df)
        df = df[
            df['num_deidentified']
            <= df['summary'].map(
                lambda s: len(s.split(' ')) / self.num_words_per_deidentified
            )
        ]
        logger.info(
            f"  Removed {old_len - len(df)} summaries with > 1 '___' per "
            f"{self.num_words_per_deidentified} words."
        )
        df.drop(columns=['num_deidentified'], inplace=True)

        return df

    def _step7_filter_hospital_course(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with missing or insufficient hospital course sections.

        Filters out rows where either 'hospital_course' or
        'brief_hospital_course' are null or too short. Normalizes excessive
        blank lines (3+ consecutive newlines to 2). Logs filter outcomes at
        INFO level.

        Args:
            df: DataFrame with 'hospital_course' and 'brief_hospital_course'
                columns.

        Returns:
            DataFrame with only valid records meeting minimum length thresholds.
        """
        logger.info("Step 7: Filter insufficient hospital courses.")

        old_len = len(df)
        df = df[df['hospital_course'].notnull()]
        logger.info(
            f"  Removed {old_len - len(df)} / {old_len} records with "
            f"no hospital course."
        )

        old_len = len(df)
        df = df[df['brief_hospital_course'].notnull()]
        logger.info(
            f"  Removed {old_len - len(df)} / {old_len} records with "
            f"no brief hospital course."
        )

        # Normalise excessive blank lines
        df['hospital_course'] = df['hospital_course'].apply(
            lambda s: _re_more_than_double_newline.sub('\n\n', s)
        )
        df['brief_hospital_course'] = df['brief_hospital_course'].apply(
            lambda s: _re_more_than_double_newline.sub('\n\n', s)
        )

        old_len = len(df)
        df = df[df['brief_hospital_course'].map(len) >= self.min_chars_bhc]
        logger.info(
            f"  Removed {old_len - len(df)} brief hospital courses with "
            f"< {self.min_chars_bhc} characters."
        )

        return df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hc(txt: str) -> Optional[str]:
        """Extract the Brief Hospital Course section from a discharge note.

        Locates the "Brief Hospital Course:" marker and extracts text until
        one of several known end markers. Returns None if the marker is absent
        or if the full note is too short (<30 words).

        Args:
            txt: The raw discharge note text.

        Returns:
            The extracted Brief Hospital Course text, normalized to single-line
            format and stripped of leading/trailing whitespace. Returns None if
            extraction fails (missing marker, text too short, or invalid bounds).
        """
        start = txt.find("Brief Hospital Course:")
        if start < 0:
            return None
        end = txt.find("Medications on Admission:")
        if end == -1:
            end = txt.find("Discharge Medications:")
        if end == -1:
            end = txt.find("Discharge Disposition:")
        if end == 0 or start >= end:
            return None
        hc = txt[start:end].replace('\n', ' ')
        hc = ' '.join(hc.split())
        if len(txt.split(' ')) < 30:
            return None
        return hc

    @staticmethod
    def _remove_empty_and_short_summaries(
        df: pd.DataFrame,
        min_length_summary: int = 350,
    ) -> pd.DataFrame:
        """Remove empty and short summaries from the DataFrame.

        Filters out summaries with zero length or shorter than the specified
        minimum. Logs the number of rows removed at DEBUG level.

        Args:
            df: DataFrame with a 'summary' column (string type).
            min_length_summary: Minimum required character count for a valid
                summary. Defaults to 350.

        Returns:
            A copy of the input DataFrame with short/empty rows removed.
        """
        old_len = len(df)
        df = df[df['summary'].str.len() > 0].copy()
        empty_removed = old_len - len(df)
        df = df[df['summary'].str.len() >= min_length_summary].copy()
        short_removed = old_len - empty_removed - len(df)
        logger.debug(
            f"Removed {empty_removed} empty and {short_removed} short summaries "
            f"(< {min_length_summary} chars)."
        )
        return df

    @staticmethod
    def _remove_regex_dict(
        df: pd.DataFrame,
        regexes: dict,
        postprocess,
        keep: int = 0,
    ) -> pd.DataFrame:
        """Remove regex-matched suffixes or prefixes from summaries.

        For each regex pattern, splits the summary at the first match and keeps
        either the left side (keep=0) or the right side (keep=1). Applies a
        postprocessing function to each modified summary. Logs statistics for
        each pattern at DEBUG level.

        Args:
            df: DataFrame with a 'summary' column (string type).
            regexes: Dictionary mapping delimiter names to compiled regex
                patterns to match against summaries.
            postprocess: A callable that takes a string and returns a processed
                string, applied after each split.
            keep: Which side of the split to keep (0=left/prefix, 1=right/suffix).
                Defaults to 0.

        Returns:
            The input DataFrame with modified 'summary' column (modified in-place).
        """
        total_changed = 0
        for delimiter_name, regex in regexes.items():
            matches = df['summary'].apply(lambda s: regex.search(s) is not None)
            total_changed += matches.sum()
            logger.debug(f"  {delimiter_name}: {matches.sum()} / {len(df)}")
            df.loc[matches, 'summary'] = df.loc[matches, 'summary'].apply(
                lambda s: regex.split(s, 1)[keep]
            )
            df['summary'] = df['summary'].apply(postprocess)
        logger.debug(f"Changed total of {total_changed} / {len(df)} summaries.")
        return df

    @staticmethod
    def _change_why_what_next_pattern_to_text(
        summaries: pd.Series,
    ) -> pd.Series:
        """Convert 'Why / What / Next' dashed lists to paragraph text.

        Transforms structured list blocks matching the 'Why admitted', 'What
        was done', and 'What next' patterns into flowing prose by replacing
        dashes and line breaks with periods and spaces.

        Args:
            summaries: Series of summary strings with potential 'Why/What/Next'
                list blocks using dashes or other list markers.

        Returns:
            Series of modified summaries with list blocks converted to prose.
        """
        random_string = (
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
            + '\n- '
        )
        summaries = summaries.apply(
            lambda s: WHY_WHAT_NEXT_HEADINGS_DASHED_LIST.sub(random_string, s)
        )
        dash_regex = re.compile(r'(?:\.)?\n-\s{0,4}', re.MULTILINE | re.IGNORECASE)

        def _remove_dashes(s: str) -> str:
            """Replace dashes in list items with periods for prose conversion.

            Args:
                s: Summary text containing random separator markers.

            Returns:
                Text with dashes converted to periods and formatting normalized.
            """
            paragraphs = s.split(random_string)
            res = [paragraphs[0]]
            for p in paragraphs[1:]:
                items = p.split('\n\n', 1)[0]
                items = '. '.join(dash_regex.split(items))
                if '\n\n' in p:
                    items = items + '\n\n' + p.split('\n\n', 1)[1]
                res.append(items.strip())
            return '\n\n'.join(res)

        return summaries.apply(lambda s: _remove_dashes(s) if random_string in s else s)