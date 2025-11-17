# -*- coding: utf-8 -*-
"""
Configuration, constants, and regular expression patterns for PDF → XML pipeline
"""
import re
import tempfile
from pathlib import Path

# ------------------------- Streamlit / temp dirs -------------------------
TEMP_ROOT = Path(tempfile.gettempdir()) / "pdf_xml_converter"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

PDF_INPUT_DIR = Path("temp_pdf_dir")
XML_OUTPUT_DIR = Path("temp_xml_dir")
PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal (add OCR/CV deps in your env if needed)
REQUIREMENTS_TXT = """\
streamlit>=1.28.0
PyMuPDF>=1.23.0
pdfplumber>=0.9.0
Pillow>=10.0.0
"""

# -------------------- Processing thresholds and constants ----------------
HEADER_THRESHOLD = 80
FOOTER_THRESHOLD = 720
MERGE_GAP = 18
LINE_SPACING_THRESHOLD = 15
INDENT_THRESHOLD = 15
Y_CLUSTER_THRESHOLD = 10
FULL_WIDTH_THRESHOLD = 0.45
GAP_THRESHOLD = 95
VERTICAL_TOLERANCE = 3

# Page padding (relative): used by column inference
PAGE_PAD_X_RATIO = 0.04
PAGE_PAD_Y_RATIO = 0.04

# Sentence terminator for soft wrap detection
SENTENCE_TERMINATOR_RE = re.compile(r'[.!?\]["\')\]]?\s*$')

# ----------------------------- List detection ----------------------------
BULLET_RE = re.compile(r'^\s*(\u2022|\*|-|–)\s+')
ORDERED_RE = re.compile(r'^\s*((?:\d+[.\)]|[A-Za-z][.\)]|[ivxlcdmIVXLCDM]+[.\)]))\s+')
BULLET_MARKER_RE = re.compile(r'(?:^\s*)(\u2022|\*|-|–)\s+')
ORDERED_MARKER_RE = re.compile(r'(?:^\s*)((?:\d+[.\)]|[A-Za-z][.\)]|[ivxlcdmIVXLCDM]+[.\)]))\s+')

ORDERED_FALLBACK_RE = re.compile(
    r"""
    ^\s*(
        \(\d{1,3}\)              |   # (1)
        \d{1,3}\)                |   # 1)
        \d{1,3}[.\:\-\–\—]\s+(?=[a-z]) | # 1. foo
        \([ivxlcdmIVXLCDM]+\)    |   # (iv)
        [ivxlcdmIVXLCDM][.\:\-\–\—]\s+(?=[a-z]) |
        \([A-Za-z]\)             |   # (A)
        [A-Za-z][.\:\-\–\—]\s+(?=[a-z])
    )\s*
    """,
    re.VERBOSE,
)

# ------------------------ Model / title / subtitle -----------------------
MODEL_PATTERNS = [
    r'Pro\s*\d{4}\s*[A-Za-z]*',
    r'Pro\d{4}[A-Za-z]*',
    r'\bPro\s*\d{4}\s*[A-Za-z]*',
]

TITLE_PATTERNS = [
    r'^\d+\.\s+[A-Z][a-zA-Z\s]+$',
    r'^[A-Z][A-Z\s]+$',
    r'^Chapter\s+\d+',
    r'^Section\s+\d+',
]

SUBTITLE_PATTERNS = [
    r'^[A-Za-z]+_[A-Za-z0-9]+$',
    r'^[A-Z][a-z]+\s+[A-Z][a-z0-9]+$',
]

# --------------------------- TOC detection --------------------------------
TOC_PATTERN = re.compile(r'(table of contents|contents)', re.IGNORECASE)
DOTTED_PAGE_PATTERN = re.compile(r'\.{2,}\s*\d+$')
NUMBERING_PATTERN = re.compile(r'^(\d+(\.\d+)*)\s+(.+)$')
INDENT_TOLERANCE = 5
MAX_TOC_PAGES = 10
TOC_INDICATORS = [
    'description', 'specification', 'service', 'troubleshooting',
    'component identification', 'function', 'procedure'
]

# ---------------------- Word→line & rotation prefs -----------------------
WORD_LINE_Y_TOLERANCE = 2.5      # points: vertical tolerance to merge words into lines
ROTATED_PAGE_AS_LANDSCAPE = True # treat 90/270 rotated pages as landscape for heuristics