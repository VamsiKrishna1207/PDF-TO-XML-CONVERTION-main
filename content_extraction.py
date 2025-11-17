"""
Content extraction module handling TOC, images, and content detection.

This version:
- Extracts true embedded images (XObjects) and SKIPS page-sized scans.
- For scanned/mixed pages that have no embedded images, it renders the page via
  pdf2image (Poppler) and uses OpenCV to detect and crop TABLE regions from the page.
  Those cropped table images are saved as pageN_tableK.png (no full-page image saved).
- For OpenCV table crops, we also compute their PDF-space bounding boxes and keep a map:
  {page_num: [ {'filename': str, 'bbox_pdf': (x0,y0,x1,y1)}, ... ]}
"""

import os
import re
from statistics import mean
import fitz  # PyMuPDF

# --- NEW imports for scanned/mixed table segmentation ---
import cv2
import numpy as np
from pdf2image import convert_from_path

from config_patterns import (
    TOC_PATTERN, DOTTED_PAGE_PATTERN, TOC_INDICATORS,
    MODEL_PATTERNS, TITLE_PATTERNS, SUBTITLE_PATTERNS,
    BULLET_RE, ORDERED_RE, BULLET_MARKER_RE, ORDERED_MARKER_RE,
    NUMBERING_PATTERN, INDENT_TOLERANCE
)


# ----------------------------- TOC -------------------------------------------
class TOCExtractor:
    """Handles TOC extraction and structure building"""

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.toc_structure = None
        self.section_page_mapping = {}

    def is_table_of_contents_section(self, lines):
        """Heuristic check if a set of lines represents a TOC-like section."""
        if not lines:
            return False

        combined_text = " ".join(line.get("text", "") for line in lines)

        # Keyword indicators
        for indicator in TOC_INDICATORS:
            if indicator.lower() in combined_text.lower():
                return True

        # Dotted leaders pattern prevalence
        dotted_count = sum(1 for line in lines if DOTTED_PAGE_PATTERN.search(line.get("text", "")))
        return (len(lines) > 0) and (dotted_count / len(lines) > 0.3)


def detect_toc_pages(doc, max_pages_to_check=10):
    """Detect pages that likely contain a table of contents using PyMuPDF."""
    toc_pages = []
    limit = min(max_pages_to_check, doc.page_count)
    for i in range(limit):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if TOC_PATTERN.search(text) or DOTTED_PAGE_PATTERN.search(text):
            toc_pages.append(i)
    return sorted(set(toc_pages))


def normalize_indent(x, indent_levels, tolerance=INDENT_TOLERANCE):
    """Normalize indentation to determine hierarchy level."""
    for idx, val in enumerate(indent_levels):
        if abs(x - val) <= tolerance:
            return idx
    indent_levels.append(x)
    indent_levels.sort()
    return indent_levels.index(x)


def extract_toc_entries(doc, toc_pages, page_offset=2):
    """
    Build TOC entries by scanning the specified TOC pages.

    Behavior (as requested):
      - Capture trailing page or page-range (e.g., '... 12' or '... 12-14').
      - Compute real page = (start of range or page) + page_offset.
      - Determine hierarchy from numbering if present, else from indent normalized to the FIRST entry's indent.
      - FIRST valid entry is always level 2 (top-level).
      - Skip empty/meaningless titles (e.g., 'Untitled') so they cannot become Sect2 parents.
    """
    toc = []
    indent_levels = []
    baseline_indent = None

    # Trailing page / page-range
    page_num_re = re.compile(r'(\d+(?:-\d+)?)\s*$')

    # Numbering like "2." or "2.3.4" optionally followed by a separator and title
    # Groups: 1 = numbering, 2 = title
    numbering_re = re.compile(r'^\s*((?:\d+(?:\.[0-9]+)*))\s*(?:[-\.)])?\s*(.*)$')

    for page_num in toc_pages:
        page = doc.load_page(page_num)
        blocks = page.get_text("dict").get("blocks", []) or []

        for block in blocks:
            for line in block.get("lines", []) or []:
                line_text = ""
                x_positions = []
                font_sizes = []

                for span in line.get("spans", []) or []:
                    text = (span.get("text") or "").strip()
                    if text:
                        line_text += text + " "
                        # left X of span
                        x_positions.append(float(span["bbox"][0]))
                        font_sizes.append(float(span.get("size", 0)))

                line_text = line_text.strip()
                if not line_text:
                    continue

                # Trailing page / range
                m = page_num_re.search(line_text)
                if not m:
                    continue

                page_str = m.group(1)
                title = line_text[:m.start()].rstrip(" .")

                # Guard against empty/meaningless titles
                if not title or title.strip().lower() == "untitled":
                    continue

                # Real page number with offset applied
                if "-" in page_str:
                    start_page, _end_page = map(int, page_str.split("-"))
                    real_page = start_page + page_offset
                else:
                    real_page = int(page_str) + page_offset

                # Indent of this line (leftmost span)
                x_indent = min(x_positions) if x_positions else 0.0

                # Numbering-based depth if available; else use indent vs baseline
                nm = numbering_re.match(title)
                if nm:
                    numbering = nm.group(1)
                    title = nm.group(2).strip()
                    depth = len(numbering.split("."))
                    # Depth 1 => top-level; deeper => sub-level
                    level_raw = 0 if depth <= 1 else 1
                else:
                    # Initialize baseline to the first valid entry's indent
                    if baseline_indent is None:
                        baseline_indent = x_indent
                    # Compare current indent to baseline with tolerance
                    level_raw = 0 if abs(x_indent - baseline_indent) <= INDENT_TOLERANCE else 1

                # Final mapping:
                # - first valid entry => level 2 (Sect2)
                # - else top-level => level 2; indented => level 3
                if not toc:
                    level = 2
                    if baseline_indent is None:
                        baseline_indent = x_indent
                    if not indent_levels:
                        indent_levels.append(baseline_indent)
                else:
                    level = 2 if level_raw == 0 else 3

                toc.append({
                    "level": level,
                    "title": title.strip(),
                    "page": int(real_page)
                })

    return toc


# ----------------------------- Images ----------------------------------------
class ImageExtractor:
    """
    Extract embedded images and, for scanned/mixed pages with no embedded images,
    render the page (pdf2image) and crop TABLE regions (OpenCV).
    For OpenCV crops we also store PDF-space bounding boxes for correct placement order.
    """

    def __init__(
        self,
        pdf_path,
        image_output_dir="extracted_images",
        doc_type="mixed",  # 'editable' | 'noneditable' | 'mixed'
        skip_full_page_images=True,
        use_pdf2image_tables=True,  # enable your OpenCV table logic on scanned pages
        poppler_path=None,  # e.g., r"C:\\poppler\\bin"
        render_dpi=300
    ):
        self.pdf_path = pdf_path
        self.image_output_dir = image_output_dir
        self.doc_type = (doc_type or "mixed").lower()
        self.skip_full_page_images = skip_full_page_images
        self.use_pdf2image_tables = use_pdf2image_tables
        self.poppler_path = poppler_path
        self.render_dpi = render_dpi

        # maps
        self.image_map = {}       # {page_num: [(filename, ext), ...]}
        self.image_xref_map = {}  # {page_num: {xref:int -> filename:str}}
        self.cv_table_crops = {}  # {page_num: [ {'filename': str, 'bbox_pdf': (x0,y0,x1,y1)}, ... ]}
        os.makedirs(self.image_output_dir, exist_ok=True)

    # ---- helpers ----
    @staticmethod
    def _is_full_page_image(page, xref, tolerance=0.02):
        """Is the image essentially the whole page?"""
        try:
            rects = page.get_image_rects(xref) or []
        except Exception:
            rects = []
        if not rects:
            return False

        p = page.rect
        page_area = float(p.width * p.height) if p.width and p.height else 1.0
        for r in rects:
            r_area = float((r.x1 - r.x0) * (r.y1 - r.y0))
            if page_area <= 0:
                continue
            coverage = r_area / page_area
            near_left = abs(r.x0 - p.x0) <= p.width * tolerance
            near_right = abs(r.x1 - p.x1) <= p.width * tolerance
            near_top = abs(r.y0 - p.y0) <= p.height * tolerance
            near_bottom = abs(r.y1 - p.y1) <= p.height * tolerance
            if coverage >= (1.0 - tolerance) and near_left and near_right and near_top and near_bottom:
                return True
        return False

    def _extract_tables_cv(self, bgr_image):
        """
        Return list of (BGR crop, (x, y, w, h)) for detected table regions.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 15, -2)

        # horizontal lines
        horizontal = thresh.copy()
        cols = horizontal.shape[1]
        horizontal_size = max(1, cols // 15)
        h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.erode(horizontal, h_struct)
        horizontal = cv2.dilate(horizontal, h_struct)

        # vertical lines
        vertical = thresh.copy()
        rows = vertical.shape[0]
        vertical_size = max(1, rows // 15)
        v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.erode(vertical, v_struct)
        vertical = cv2.dilate(vertical, v_struct)

        # table mask = horizontal + vertical
        mask = cv2.add(horizontal, vertical)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        table_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 50:  # simple noise filter
                crop = bgr_image[y:y + h, x:x + w]
                table_regions.append((crop, (x, y, w, h)))
        return table_regions

    def _render_page_via_pdf2image(self, page_number_1based):
        """
        Render a single page to BGR using pdf2image (Poppler).
        """
        pil_pages = convert_from_path(
            self.pdf_path,
            dpi=self.render_dpi,
            poppler_path=self.poppler_path,
            first_page=page_number_1based,
            last_page=page_number_1based
        )
        if not pil_pages:
            return None
        pil_img = pil_pages[0]  # RGB PIL
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr

    # ---- main API ----
    def extract_images(self):
        """
        - Try to extract real embedded images (skip page-sized scans).
        - If none and (doc_type is scanned/mixed) and pdf2image-tables enabled:
          render page via pdf2image and crop table regions; save each crop and keep PDF-space bbox.
        """
        print("=== EXTRACTING IMAGES ===")
        doc = fitz.open(self.pdf_path)
        try:
            for page_num, page in enumerate(doc, start=1):
                exported = 0

                # 1) Embedded images (XObjects)
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    if self.skip_full_page_images and self._is_full_page_image(page, xref):
                        continue

                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    image_filename = f"page{page_num}_img{img_index + 1}.{image_ext}"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    self.image_map.setdefault(page_num, []).append((image_filename, image_ext))
                    self.image_xref_map.setdefault(page_num, {})[int(xref)] = image_filename
                    exported += 1

                # Also allow fallback on editable pages if no embedded images were exported
                if exported == 0 and self.use_pdf2image_tables and (self.doc_type in ("noneditable", "mixed") or self.doc_type == "editable"):
                    bgr = self._render_page_via_pdf2image(page_num)
                    if bgr is not None:
                        tables = self._extract_tables_cv(bgr)
                        for idx, (tbl, (tx, ty, tw, th)) in enumerate(tables, start=1):
                            out_name = f"page{page_num}_table{idx}.png"
                            out_path = os.path.join(self.image_output_dir, out_name)
                            cv2.imwrite(out_path, tbl)
                            self.image_map.setdefault(page_num, []).append((out_name, "png"))

                            # Compute PDF-space bbox
                            p = page.rect
                            page_w_pt, page_h_pt = float(p.width), float(p.height)
                            img_h_px, img_w_px = bgr.shape[:2]
                            # scale factors pixel -> point
                            sx = img_w_px / max(1.0, page_w_pt)
                            sy = img_h_px / max(1.0, page_h_pt)
                            x0_pt = tx / sx
                            y0_pt = ty / sy
                            x1_pt = (tx + tw) / sx
                            y1_pt = (ty + th) / sy
                            self.cv_table_crops.setdefault(page_num, []).append({
                                'filename': out_name,
                                'bbox_pdf': (x0_pt, y0_pt, x1_pt, y1_pt)
                            })
                            exported += 1

            total = sum(len(v) for v in self.image_map.values())
            print(f"Extracted {total} images (embedded + cv tables; full-page scans skipped).")
        finally:
            doc.close()

    def get_image_map(self):
        return self.image_map

    def get_image_xref_map(self):
        return self.image_xref_map

    def get_cv_table_crops_map(self):
        return self.cv_table_crops


# ----------------------------- Content detection ------------------------------
class ContentDetector:
    """Handles content detection and classification."""

    def __init__(self):
        pass

    # --- models ---
    def extract_models_from_text(self, text):
        found = []
        for pattern in MODEL_PATTERNS:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                cleaned = re.sub(r"\s+", " ", match.strip())
                if cleaned not in found:
                    found.append(cleaned)
        return found

    def is_model_line(self, text_line):
        stripped = (text_line.get("text") or "").strip()
        if not stripped:
            return False
        models = self.extract_models_from_text(stripped)
        if not models:
            return False
        if len(models) >= 2:
            return True
        if len(models) == 1 and ("," in stripped or " and " in stripped.lower()):
            return True
        ratio = sum(len(m) for m in models) / max(1, len(stripped))
        return ratio > 0.3

    # --- titles / subtitles ---
    def is_likely_title_or_subtitle(self, text_line, page):
        stripped = (text_line.get("text") or "").strip()
        if not stripped:
            return None
        if self.is_model_line(text_line):
            return None

        for pattern in TITLE_PATTERNS:
            if re.match(pattern, stripped):
                if not self.extract_models_from_text(stripped):
                    return "title"

        for pattern in SUBTITLE_PATTERNS:
            if re.match(pattern, stripped):
                if not self.extract_models_from_text(stripped) and "Pro" not in stripped:
                    return "subtitle"

        # font-size heuristic (expects pdfplumber Page with chars)
        chars = getattr(page, "chars", []) or []
        if chars:
            x0, top, x1, bottom = (
                text_line.get("x0", 0),
                text_line.get("top", 0),
                text_line.get("x1", 0),
                text_line.get("bottom", 0),
            )
            tol = 2
            fs = [
                ch.get("size")
                for ch in chars
                if (top - tol) <= ch.get("top", 0) <= (bottom + tol)
                and (x0 - tol) <= ch.get("x0", 0) <= (x1 + tol)
                and ch.get("size")
            ]
            avg_size = mean(fs) if fs else None
            if avg_size:
                if 18 <= avg_size <= 40 and not self.extract_models_from_text(stripped):
                    return "title"
                if 12 <= avg_size < 18 and not self.extract_models_from_text(stripped) and "Pro" not in stripped:
                    return "subtitle"

        return None

    # --- lists ---
    def is_list_item(self, text: str):
        stripped = (text or "").strip()
        return bool(BULLET_RE.match(stripped) or ORDERED_RE.match(stripped))

    def get_list_type(self, text: str):
        stripped = (text or "").strip()
        if BULLET_RE.match(stripped):
            return "itemized"
        if ORDERED_RE.match(stripped):
            return "ordered"
        return None

    def has_multiple_list_markers(self, line_text: str):
        bullet_markers = list(BULLET_MARKER_RE.finditer(line_text))
        ordered_markers = list(ORDERED_MARKER_RE.finditer(line_text))
        return bullet_markers, ordered_markers

    def extract_list_segments(self, line_text: str, markers):
        segments = []
        for i, m in enumerate(markers):
            seg_start = m.end()
            next_start = markers[i + 1].start() if i + 1 < len(markers) else len(line_text)
            seg = line_text[seg_start:next_start].strip()
            if seg:
                segments.append(seg)
        return segments

    def clean_list_text(self, text, list_type):
        if list_type == "itemized":
            return BULLET_RE.sub("", text).strip()
        if list_type == "ordered":
            return ORDERED_RE.sub("", text).strip()
        return (text or "").strip()

    # --- bold / para titles ---
    def is_bold_text(self, text_line, page):
        stripped = (text_line.get("text") or "").strip()
        if not stripped:
            return False
        chars = getattr(page, "chars", []) or []
        if not chars:
            return False
        x0, top, x1, bottom = (
            text_line.get("x0", 0),
            text_line.get("top", 0),
            text_line.get("x1", 0),
            text_line.get("bottom", 0),
        )
        tol = 2
        relevant = [
            ch for ch in chars
            if (top - tol) <= ch.get("top", 0) <= (bottom + tol)
            and (x0 - tol) <= ch.get("x0", 0) <= (x1 + tol)
        ]
        if not relevant:
            return False

        bold_count, total = 0, len(relevant)
        sizes = [ch.get("size", 0) for ch in relevant if ch.get("size", 0) > 0]
        line_avg = mean(sizes) if sizes else 0
        for ch in relevant:
            font_name = (ch.get("fontname") or "").lower()
            if any(t in font_name for t in ["bold", "black", "heavy", "demi"]):
                bold_count += 1
                continue
            if line_avg and ch.get("size", 0) > line_avg * 1.10:
                bold_count += 1
        return (bold_count / total) > 0.5 if total else False

    def is_para_title(self, text_line, page):
        stripped = (text_line.get("text") or "").strip()
        if not stripped:
            return False
        if not self.is_bold_text(text_line, page):
            return False
        if self.is_model_line(text_line):
            return False

        ttype = self.is_likely_title_or_subtitle(text_line, page)
        if ttype in ("title", "subtitle"):
            return False

        if self.is_list_item(stripped):
            return False

        if len(stripped) < 5 or len(stripped) > 100:
            return False

        if stripped.isupper() and len(stripped) > 10:
            return False

        return True
# --- relative heading fallback ---
# We compute per-page statistics (median bold ratio) and use layout cues.
# We intentionally DO NOT use per-line font size, to avoid instability when
# line sizes differ by only 1–2 pt.

from statistics import median
import numpy as _np  # safe alias; your file already imports numpy as np above

class _RelativeHeadingHelper:
    BOLD_TOKENS = ("bold", "black", "heavy", "demi")

    @staticmethod
    def compute_page_stats(page):
        """
        Compute simple per-page stats. We keep size percentiles available, but
        we won't use per-line size in scoring; we normalize boldness by page median.
        Returns: {'size_q40','size_q60','size_q80','med_bold'}
        """
        chars = getattr(page, 'chars', []) or []
        sizes = [c.get('size') for c in chars if c.get('size')]
        bold_flags = []
        for c in chars:
            fn = (c.get('fontname') or '').lower()
            is_bold = any(t in fn for t in _RelativeHeadingHelper.BOLD_TOKENS)
            bold_flags.append(1 if is_bold else 0)

        if sizes:
            try:
                q40, q60, q80 = _np.percentile(sizes, [40, 60, 80]).tolist()
            except Exception:
                ss = sorted(sizes)
                def _pct(p):
                    i = max(0, min(len(ss)-1, int(round(p/100.0*(len(ss)-1)))))
                    return ss[i]
                q40, q60, q80 = _pct(40), _pct(60), _pct(80)
        else:
            q40 = q60 = q80 = 0.0
        med_bold = median(bold_flags) if bold_flags else 0.0
        return {'size_q40': float(q40), 'size_q60': float(q60), 'size_q80': float(q80), 'med_bold': float(med_bold)}

    @staticmethod
    def _line_relevant_chars(text_line, page, tol=2):
        x0, top, x1, bottom = (
            text_line.get("x0", 0), text_line.get("top", 0),
            text_line.get("x1", 0), text_line.get("bottom", 0)
        )
        chars = getattr(page, 'chars', []) or []
        relevant = [
            ch for ch in chars
            if (top - tol) <= ch.get('top', 0) <= (bottom + tol)
            and (x0 - tol) <= ch.get('x0', 0) <= (x1 + tol)
        ]
        return relevant

    @staticmethod
    def compute_line_features(text_line, page):
        # Basic geometry
        x0, top, x1, bottom = (
            text_line.get("x0", 0), text_line.get("top", 0),
            text_line.get("x1", 0), text_line.get("bottom", 0)
        )
        page_w = getattr(page, 'width', None) or getattr(page, 'page_width', 595)
        page_h = getattr(page, 'height', None) or getattr(page, 'page_height', 842)
        width = (x1 - x0) if (x1 and x0) else 0.0
        centered = abs(x0 - (page_w - x1)) < max(5, 0.05 * page_w)
        # Prefer upstream-computed inter-line gap if provided
        space_above = text_line.get('space_above', top)

        # We will NOT use per-line avg_size in scoring, but we still compute bold_ratio
        rel = _RelativeHeadingHelper._line_relevant_chars(text_line, page)
        bold_hits = 0
        for c in rel:
            fn = (c.get('fontname') or '').lower()
            if any(t in fn for t in _RelativeHeadingHelper.BOLD_TOKENS):
                bold_hits += 1
        bold_ratio = (bold_hits/len(rel)) if rel else 0.0

        return {
            'x0': x0, 'top': top, 'x1': x1, 'bottom': bottom,
            'bold_ratio': bold_ratio,
            'centered': centered, 'width': width,
            'page_w': page_w, 'page_h': page_h,
            'space_above': space_above
        }

    @staticmethod
    def score_line_for_heading(features, stats, stripped_text, title_patterns, subtitle_patterns):
        score = 0.0
        # NOTE: We intentionally do NOT use per-line font size here.

        # Boldness normalized by page median
        br = features.get('bold_ratio', 0.0)
        med_bold = stats.get('med_bold', 0.0)
        if br >= max(0.1, med_bold + 0.15):
            score += 1.2
        elif br >= max(0.05, med_bold + 0.07):
            score += 0.6

        # Layout cues
        if features.get('centered'):
            score += 0.8
        if features.get('space_above', 0.0) > 0.04 * features.get('page_h', 1.0):
            score += 0.8
        if features.get('width', 0.0) < 0.70 * features.get('page_w', 1.0):
            score += 0.4

        # Regex boosts (reuse your patterns)
        t = stripped_text
        try:
            if any(re.match(p, t) for p in title_patterns):
                score += 1.0
        except Exception:
            pass
        try:
            if any(re.match(p, t) for p in subtitle_patterns):
                score += 0.6
        except Exception:
            pass

        # Penalties
        if len(t) < 3 or len(t) > 120:
            score -= 0.6
        if t.isupper() and len(t) > 12:
            score -= 0.3
        return score

# --- Preserve original ContentDetector behavior and add fallback only ---

if hasattr(ContentDetector, 'is_likely_title_or_subtitle'):
    _original_is_likely_title_or_subtitle = ContentDetector.is_likely_title_or_subtitle
else:
    _original_is_likely_title_or_subtitle = None

_original_cd_init = getattr(ContentDetector, '__init__', None)

def _cd_init_augmented(self):
    if _original_cd_init:
        try:
            _original_cd_init(self)
        except TypeError:
            _original_cd_init(self)
    self._use_relative_heading_fallback = True

ContentDetector.__init__ = _cd_init_augmented

def _is_likely_title_or_subtitle_fallback(self, text_line, page):
    """Fallback called only when original returns None."""
    stripped = (text_line.get("text") or "").strip()
    if not stripped:
        return None
    if self.is_model_line(text_line):
        return None

    stats = _RelativeHeadingHelper.compute_page_stats(page)
    feats = _RelativeHeadingHelper.compute_line_features(text_line, page)

    try:
        title_patterns = TITLE_PATTERNS
    except NameError:
        title_patterns = []
    try:
        subtitle_patterns = SUBTITLE_PATTERNS
    except NameError:
        subtitle_patterns = []

    score = _RelativeHeadingHelper.score_line_for_heading(
        feats, stats, stripped, title_patterns, subtitle_patterns
    )

    if score >= 2.6:
        return "title"
    if score >= 1.6 and "Pro" not in stripped:
        return "subtitle"
    return None

def _is_likely_title_or_subtitle_augmented(self, text_line, page):
    result = _original_is_likely_title_or_subtitle(self, text_line, page)
    if result is not None:
        return result
    if getattr(self, '_use_relative_heading_fallback', True):
        return _is_likely_title_or_subtitle_fallback(self, text_line, page)
    return None