# -*- coding: utf-8 -*-
"""
XML generation, page processing and structure creation module — LANDSCAPE-AWARE
This update changes ONLY the landscape processing path; portrait behavior is unchanged.
Landscape-specific changes:
- Use UNROTATED page frame (width/height) for all layout math on landscape pages only.
- Stricter two-column detection with a single-column guard (avoids false 2-col splits).
- Promote Subtitle/title to page-level before any columns (landscape only).
- Slightly more tolerant list continuation and extra noise filtering (landscape only).

Additional updates in this version:
- Ordered list normalization: handles glued markers like "10.Disconnect" -> "10. Disconnect".
- Ordered list label preservation: numeric/alpha/Roman labels are kept in XML, including multi-marker lines.

Requires constants from config_patterns.py:
 HEADER_THRESHOLD, FOOTER_THRESHOLD, MERGE_GAP, LINE_SPACING_THRESHOLD,
 INDENT_THRESHOLD, ORDERED_FALLBACK_RE, PAGE_PAD_X_RATIO, PAGE_PAD_Y_RATIO,
 SENTENCE_TERMINATOR_RE
"""
import os
import re
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
from xml.dom import minidom

# OCR-table integration (scanned pages only)
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

from layout_processing import LayoutAnalyzer, TextProcessor, TableExtractor
from content_extraction import ContentDetector
from config_patterns import (
    HEADER_THRESHOLD, FOOTER_THRESHOLD, MERGE_GAP,
    LINE_SPACING_THRESHOLD, INDENT_THRESHOLD,
    ORDERED_FALLBACK_RE,
    PAGE_PAD_X_RATIO, PAGE_PAD_Y_RATIO,
    SENTENCE_TERMINATOR_RE,
)

DEBUG = False

# ---------------------- Utilities ----------------------
def is_noise_line(text: str) -> bool:
    """Filter obvious page furniture / serial lines and ultra-short noise."""
    if not text:
        return True
    if text.isdigit():
        return True
    if re.match(r'^DICV-OM926-\d{3}$', text):
        return True
    if len(text.strip()) <= 2:
        return True
    return False

# Extra noise patterns — used ONLY for landscape pages
NOISE_PATTERNS = [
    re.compile(r"\bKnow\s+Your\s+Vehicle\b", re.I),
    re.compile(r"^\s*Page\s+\d+\s*$", re.I),
]
def _is_landscape_noise(s: str) -> bool:
    s = s or ''
    if is_noise_line(s):
        return True
    return any(p.search(s) for p in NOISE_PATTERNS)

def prettify_xml(elem: ET.Element) -> str:
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent=" ")

def _sect_level_from_tag(tag: str):
    """Return integer section level from a tag like 'Sect2'; None if not a section."""
    m = re.match(r'^Sect(\d+)$', tag or '')
    return int(m.group(1)) if m else None

# ---------------------- Section tree builder ----------------------
def build_xml_tree(doc, toc, first_page_is_section1=False, first_title=None):
    """
    Build XML tree structure with nested sections and page assignments.
    Rules:
    - Sect1 starts at page 1
    - Pre-TOC pages (before the first TOC-specified page) are assigned to Sect1
    - Create Sect{level} using RAW entry["level"] (2 = top; 3 = sub) via a stack
    - Compute end pages so parents don't overlap children (no end < start)
    """
    root = ET.Element("Document")
    section_page_map = {}
    toc_valid = [e for e in (toc or []) if isinstance(e.get('page'), int) and e.get('page') >= 1]
    toc_sorted = sorted(toc_valid, key=lambda x: x["page"]) if toc_valid else []

    sect1_elem = ET.SubElement(root, "Sect1", start_page="1")
    if toc_sorted:
        first_toc_page = min(e["page"] for e in toc_sorted)
        if first_toc_page > 1:
            sect1_elem.set("end_page", str(first_toc_page - 1))
        for p in range(1, first_toc_page):
            page_elem = ET.SubElement(sect1_elem, "page", number=str(p))
            section_page_map[p] = {
                "page_element": page_elem,
                "section_element": sect1_elem,
                "title": "Section 1",
                "level": 1,
                "current_section": "Section 1",
                "section_level": 1,
            }

    # Build nested sections from TOC entries
    section_stack = [(sect1_elem, 1)]
    for entry in toc_sorted:
        level = int(entry.get("level", 2))
        title = (entry.get("title") or "Untitled").strip()
        page_num = int(entry.get("page"))
        while len(section_stack) > 1 and section_stack[-1][1] >= level:
            section_stack.pop()
        parent_elem, _ = section_stack[-1]
        section_tag = f"Sect{level}"
        section_elem = ET.SubElement(parent_elem, section_tag)
        section_elem.set("start_page", str(page_num))
        ET.SubElement(section_elem, "Title").text = title
        section_stack.append((section_elem, level))

    # Compute end pages
    def compute_end_pages(elem, start_page, total_pages):
        children = [c for c in elem if c.tag.startswith("Sect")]
        children.sort(key=lambda c: int(c.get("start_page", start_page)))
        if not children:
            elem.set("end_page", str(total_pages))
        else:
            for i, ch in enumerate(children):
                ch_start = int(ch.get("start_page", start_page))
                ch_total = int(children[i + 1].get("start_page")) - 1 if i + 1 < len(children) else total_pages
                compute_end_pages(ch, ch_start, ch_total)
            first_child_start = int(children[0].get("start_page", start_page))
            elem.set("end_page", str(max(start_page, first_child_start - 1)))

    doc_total = getattr(doc, "page_count", None) or getattr(doc, "pageCount", None) or 1
    compute_end_pages(sect1_elem, 1, doc_total)

    # Assign pages to sections
    def assign_pages_to_sections(elem, level):
        s = int(elem.get("start_page", 1))
        e = int(elem.get("end_page", doc_total))
        title_elem = elem.find("Title")
        section_title = title_elem.text if title_elem is not None else f"Section Level {level}"
        children = [c for c in elem if c.tag.startswith("Sect")]
        existing = {int(p.get("number")) for p in elem.findall("page")}
        to_fill = [p for p in range(s, e + 1) if p not in existing]

        if not children:
            for p in to_fill:
                if p not in section_page_map:
                    pe = ET.SubElement(elem, "page", number=str(p))
                    section_page_map[p] = {
                        "page_element": pe,
                        "section_element": elem,
                        "title": section_title,
                        "level": level,
                        "current_section": section_title,
                        "section_level": level,
                    }
        else:
            for ch in children:
                ch_lvl = _sect_level_from_tag(ch.tag) or (level + 1)
                assign_pages_to_sections(ch, ch_lvl)
            covered = set()
            for ch in children:
                cs = int(ch.get("start_page", s)); ce = int(ch.get("end_page", s))
                covered.update(range(cs, ce + 1))
            for p in to_fill:
                if p not in covered and p not in section_page_map:
                    pe = ET.SubElement(elem, "page", number=str(p))
                    section_page_map[p] = {
                        "page_element": pe,
                        "section_element": elem,
                        "title": section_title,
                        "level": level,
                        "current_section": section_title,
                        "section_level": level,
                    }

    assign_pages_to_sections(sect1_elem, 1)
    return root, section_page_map

# ---------------------- XML generator helpers ----------------------
class XMLGenerator:
    """Handles XML structure generation and element creation."""
    def __init__(self, image_output_dir="extracted_images"):
        self.image_output_dir = image_output_dir
        self.root = ET.Element("Document")

    # --- Tables ---
    def create_table_element(self, table, page_number, table_index):
        table_elem = ET.Element("InformalTable", Tableinfo=f"{page_number:02}{table_index:02}")
        tgroup_elem = ET.SubElement(table_elem, "TGroup")
        thead_elem = ET.SubElement(tgroup_elem, "THead")
        header_row_elem = ET.SubElement(thead_elem, "TRow")
        for cell in (table[0] if table else []):
            cell_elem = ET.SubElement(header_row_elem, "TCell")
            cell_elem.text = (cell or "").strip()
        tbody_elem = ET.SubElement(tgroup_elem, "TBody")
        for row in table[1:]:
            trow_elem = ET.SubElement(tbody_elem, "TRow")
            for cell in row:
                cell_elem = ET.SubElement(trow_elem, "TCell")
                cell_elem.text = (cell or "").strip()
        return table_elem

    # --- Images ---
    def create_image_element(self, page_number, img_index, image_filename):
        figure = ET.Element("InformalFigure", Value="0")
        graphic = ET.Element("Graphic", {
            "entityref": os.path.join(self.image_output_dir, image_filename),
            "entity": f"Graphic{page_number}_{img_index+1}",
        })
        figure.append(graphic)
        return figure

    # --- Containers merge ---
    def merge_adjacent_containers(self, containers):
        """Merge adjacent text containers of the same tag within MERGE_GAP."""
        merged, i = [], 0
        while i < len(containers):
            top_i, item_i = containers[i]
            tag_i, payload_i = item_i
            if isinstance(payload_i, list):
                lines_acc = list(payload_i)
                bottom_i = max(L.get('bottom', L.get('top', 0)) for L in lines_acc) if lines_acc else top_i
                j = i + 1
                while j < len(containers):
                    top_j, item_j = containers[j]
                    tag_j, payload_j = item_j
                    if isinstance(payload_j, list) and tag_j == tag_i:
                        lines_j = list(payload_j)
                        top_j_line = min(L.get('top', 0) for L in lines_j) if lines_j else top_j
                        gap = top_j_line - bottom_i
                        if gap <= MERGE_GAP:
                            lines_acc.extend(lines_j)
                            bottom_i = max(bottom_i, max(L.get('bottom', L.get('top', 0)) for L in lines_j))
                            j += 1
                            continue
                    break
                merged.append((min(L.get('top', 0) for L in lines_acc), (tag_i, lines_acc)))
                i = j
            else:
                merged.append((top_i, (tag_i, payload_i)))
                i += 1
        return sorted(merged, key=lambda c: (c[0], 0))

    # --- Lines → semantic XML (lists, titles, paragraphs) ---
    def process_lines_container(self, page, lines, container_elem, page_number, table_bboxes, content_detector, toc_extractorr):
        """Process a list of line dicts into semantic XML under container_elem."""
        paragraph_lines = []
        current_list_items = []
        list_mode = None
        list_indent_x = None
        last_list_y = None
        last_line_y = None
        last_line_x = None
        itemized_list_elem = None
        ordered_list_elem = None

        # landscape flag from pdfplumber page (doesn't change pipeline logic for portrait)
        try:
            is_landscape = (float(getattr(page, 'width', 0)) > float(getattr(page, 'height', 0)))
        except Exception:
            is_landscape = False

        def ends_with_terminator(s: str) -> bool:
            return bool(SENTENCE_TERMINATOR_RE.search((s or "").rstrip()))

        def flush_paragraph():
            nonlocal paragraph_lines, last_line_y, last_line_x
            if paragraph_lines:
                para_elem = ET.SubElement(container_elem, "Para")
                para_elem.text = " ".join(line.strip() for line in paragraph_lines)
                paragraph_lines.clear()
            last_line_y = None
            last_line_x = None

        def flush_list():
            nonlocal current_list_items, list_mode, list_indent_x, last_list_y, itemized_list_elem, ordered_list_elem
            if current_list_items:
                if list_mode == "itemized":
                    if itemized_list_elem is None:
                        itemized_list_elem = ET.SubElement(container_elem, "ItemizedList")
                    for itm in current_list_items:
                        text = " ".join(p.strip() for p in itm["parts"]).strip()
                        if text:
                            li = ET.SubElement(itemized_list_elem, "ListItem")
                            para = ET.SubElement(li, "Para")
                            para.text = text
                elif list_mode == "ordered":
                    if ordered_list_elem is None:
                        ordered_list_elem = ET.SubElement(
                            container_elem, "OrderedList",
                            {"InheritNum": "ignore", "continuation": "continues"}
                        )
                    for itm in current_list_items:
                        text = " ".join(p.strip() for p in itm["parts"]).strip()
                        if text:
                            li = ET.SubElement(ordered_list_elem, "ListItem")
                            para = ET.SubElement(li, "Para")
                            label = itm.get("label")
                            para.text = (f"{label} {text}".strip() if label else text)
            current_list_items.clear()
            list_mode = None
            list_indent_x = None
            last_list_y = None
            itemized_list_elem = None
            ordered_list_elem = None

        # --- helper: ordered list label extraction (numeric/alpha/Roman) ---
        def _extract_order_label(s: str):
            """
            Return (label, rest_text) for common ordered markers.
            Examples: '1.', '1)', '(1)', 'A)', 'ii.', 'IV)'. If none, (None, s).
            """
            import re
            if not s:
                return None, s
            patterns = [
                r'^\s*\((\d{1,3})\)\s+(.*)$',               # (1) foo
                r'^\s*(\d{1,3}[.\):–—-])\s+(.*)$',          # 1. foo / 1) foo / 1: foo / 1– foo
                r'^\s*([ivxlcdmIVXLCDM]+[.\)])\s+(.*)$',    # ii. foo / IV) foo
                r'^\s*([A-Za-z][.\)])\s+(.*)$',             # A) foo / a. foo
            ]
            for pat in patterns:
                m = re.match(pat, s)
                if m:
                    lbl = m.group(1)
                    rest = m.group(2)
                    if lbl.isdigit():
                        lbl = f"{lbl}."
                    return lbl.strip(), rest.strip()
            return None, s.strip()

        for text_line in lines:
            line_text = text_line.get('text') or ""
            line_text = line_text.rstrip("\n\r")
            stripped = line_text.strip()

            # --- NEW: normalize glued markers like "10.Disconnect" -> "10. Disconnect"
            # Numeric with punctuation: 10. / 10) / (10) stuck to word
            stripped = re.sub(r'^(\(?\d{1,3}\)?[.)])(?=\S)', r'\1 ', stripped)
            # Alpha or Roman markers with punctuation: II. / II) / A) stuck to word
            stripped = re.sub(r'^([ivxlcdmIVXLCDM]+[.)]|[A-Za-z][.)])(?=\S)', r'\1 ', stripped)

            # NOISE: portrait uses original filter; landscape uses extended noise
            if (is_landscape and _is_landscape_noise(stripped)) or (not is_landscape and is_noise_line(stripped)):
                continue

            x0 = text_line.get('x0', 0)
            top = text_line.get('top', 0)

            if not stripped:
                flush_paragraph()
                flush_list()
                continue

            # Model detection
            if content_detector.is_model_line(text_line):
                flush_paragraph()
                flush_list()
                model_elem = ET.SubElement(container_elem, "Model")
                model_elem.text = stripped.strip().rstrip(',.').replace(' ', ' ')
                continue

            # Primary list detection
            if content_detector.is_list_item(stripped):
                flush_paragraph()
                current_list_type = content_detector.get_list_type(stripped)
                if list_mode not in (current_list_type, None):
                    flush_list()
                list_mode = current_list_type

                if list_mode == "ordered":
                    # preserve numeric/alpha/Roman label
                    lbl, rest = _extract_order_label(stripped)
                    if lbl is None:
                        cleaned = content_detector.clean_list_text(stripped, list_mode)
                        lbl, rest = None, cleaned
                    current_list_items.append({"parts": [rest], "x0": x0, "y": top, "label": lbl})
                else:
                    # itemized (bullets)
                    cleaned = content_detector.clean_list_text(stripped, list_mode)
                    current_list_items.append({"parts": [cleaned], "x0": x0, "y": top, "label": None})

                list_indent_x = x0
                last_list_y = top
                continue

            # Fallback ordered-list
            if ORDERED_FALLBACK_RE.match(stripped):
                flush_paragraph()
                current_list_type = "ordered"
                if list_mode not in (current_list_type, None):
                    flush_list()
                list_mode = current_list_type

                # retain label even here
                lbl, rest = _extract_order_label(stripped)
                if lbl is None:
                    cleaned = ORDERED_FALLBACK_RE.sub("", stripped, count=1).strip()
                    lbl, rest = None, cleaned

                current_list_items.append({"parts": [rest], "x0": x0, "y": top, "label": lbl})
                list_indent_x = x0
                last_list_y = top
                continue

            # Multiple markers within one visual line
            bullet_markers, ordered_markers = content_detector.has_multiple_list_markers(line_text)
            if bullet_markers or ordered_markers:
                flush_paragraph()
                if list_mode:
                    flush_list()

                if bullet_markers:
                    itemized_list_elem = ET.SubElement(container_elem, "ItemizedList")
                    segments = content_detector.extract_list_segments(line_text, bullet_markers)
                    for seg in segments:
                        li = ET.SubElement(itemized_list_elem, "ListItem")
                        para = ET.SubElement(li, "Para")
                        para.text = seg

                if ordered_markers:
                    ordered_list_elem = ET.SubElement(
                        container_elem, "OrderedList",
                        {"InheritNum": "ignore", "continuation": "continues"}
                    )
                    segments = content_detector.extract_list_segments(line_text, ordered_markers)

                    # derive labels from original matches
                    labels = []
                    for m in ordered_markers:
                        raw_marker = line_text[m.start():m.end()]
                        lbl, _ = _extract_order_label(raw_marker + " X")  # +X helps pattern expect a trailing token
                        labels.append(lbl)

                    if len(labels) != len(segments):
                        labels = [None] * len(segments)

                    for lbl, seg in zip(labels, segments):
                        li = ET.SubElement(ordered_list_elem, "ListItem")
                        para = ET.SubElement(li, "Para")
                        para.text = (f"{lbl} {seg}".strip() if lbl else seg)
                continue

            # Title / Subtitle detection
            title_type = content_detector.is_likely_title_or_subtitle(text_line, page)
            if title_type == 'title':
                flush_paragraph()
                flush_list()
                title_elem = ET.SubElement(container_elem, "title")
                title_elem.text = stripped
                continue
            elif title_type == 'subtitle':
                flush_paragraph()
                flush_list()
                subtitle_elem = ET.SubElement(container_elem, "Subtitle")
                subtitle_elem.text = stripped
                continue

            # List continuation
            if list_mode and current_list_items:
                prev = current_list_items[-1]
                prev_last = prev["parts"][-1] if prev["parts"] else ""
                same_indent = (list_indent_x is not None and abs(x0 - list_indent_x) <= INDENT_THRESHOLD * 3)
                close_in_y = (last_list_y is not None and abs(top - last_list_y) <= (LINE_SPACING_THRESHOLD * 2.0))

                if is_landscape:
                    # landscape-only mild outdent tolerance
                    mild_outdent = (list_indent_x is not None and 0 < (x0 - list_indent_x) <= INDENT_THRESHOLD * 1.2)
                    if not ends_with_terminator(prev_last) and (same_indent or close_in_y or mild_outdent):
                        prev["parts"].append(stripped)
                        last_list_y = top
                        continue
                else:
                    if not ends_with_terminator(prev_last) and (same_indent or close_in_y):
                        prev["parts"].append(stripped)
                        last_list_y = top
                        continue

            is_continuation = False
            if list_indent_x is not None and abs(x0 - list_indent_x) <= INDENT_THRESHOLD * 2:
                is_continuation = True
            if last_list_y is not None and abs(top - last_list_y) <= (LINE_SPACING_THRESHOLD * 1.5):
                is_continuation = True
            if is_continuation:
                current_list_items[-1]["parts"].append(stripped)
                last_list_y = top
                continue

            if current_list_items:
                last_txt = current_list_items[-1]["parts"][-1] if current_list_items[-1]["parts"] else ""
                if ends_with_terminator(last_txt):
                    flush_list()

            # Paragraph accumulation
            current_y = top
            current_x = x0
            is_new_block = False
            if last_line_y is not None and abs(last_line_y - current_y) > (LINE_SPACING_THRESHOLD * 1.2):
                is_new_block = True
            if last_line_x is not None and abs(last_line_x - current_x) > INDENT_THRESHOLD:
                is_new_block = True
            if is_new_block:
                flush_paragraph()
            paragraph_lines.append(stripped)
            last_line_y = current_y
            last_line_x = current_x

        flush_paragraph()
        flush_list()

    def save_xml(self, output_xml_path):
        def indent(elem, level=0):
            i = "\n" + level * " "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + " "
                for e in list(elem):
                    indent(e, level + 1)
                if not elem[-1].tail or not elem[-1].tail.strip():
                    elem[-1].tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        indent(self.root)
        tree = ET.ElementTree(self.root)
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    def get_root_element(self):
        return self.root

# ---------------------- Page processor ----------------------
class PageProcessor:
    """Coordinates processing of individual PDF pages with hybrid single-list approach."""
    def __init__(self, image_map, image_output_dir="extracted_images", image_xref_map=None, cv_table_crops_map=None):
        self.image_map = image_map
        self.image_xref_map = image_xref_map or {}
        self.image_output_dir = image_output_dir
        self.layout_analyzer = LayoutAnalyzer()
        self.content_detector = ContentDetector()
        self.table_extractor = TableExtractor()
        self.text_processor = TextProcessor()
        self.cv_table_crops_map = cv_table_crops_map or {}

        # internal: holds OCR words for scanned pages so column fallback can see them
        self._ocr_words_cache = None

        # OCR table configuration (set from main.py after constructing)
        self._pdf_path = None
        self._poppler_path = None
        self._render_dpi = 300
        self._ocr_lang = "eng"
        self._ocr_tables_enabled = True

    # consistent UNROTATED width/height helper
    def _unrotated_wh(self, fitz_page):
        rot = int(getattr(fitz_page, "rotation", 0)) % 360
        r = fitz_page.rect
        if rot in (90, 270):
            return float(r.height), float(r.width)
        return float(r.width), float(r.height)

    # ---------- OCR TABLE HELPERS (SCANNED ONLY) ----------
    def _render_page_bgr_via_pdf2image(self, page_number_1based: int):
        if not self._pdf_path:
            return None
        pages = convert_from_path(
            self._pdf_path,
            dpi=int(self._render_dpi or 300),
            poppler_path=self._poppler_path,
            first_page=page_number_1based,
            last_page=page_number_1based
        )
        if not pages:
            return None
        rgb = np.array(pages[0])
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _detect_grid_tables(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -4)
        h = thr.copy(); v = thr.copy()
        cols, rows = h.shape[1], v.shape[0]
        h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cols//20), 1))
        v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, rows//20)))
        h = cv2.erode(h, h_struct); h = cv2.dilate(h, h_struct)
        v = cv2.erode(v, v_struct); v = cv2.dilate(v, v_struct)
        mask = cv2.add(h, v)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 80 or h < 80:
                continue
            roi = mask[y:y+h, x:x+w]
            density = (cv2.countNonZero(roi) / float(w*h)) if (w*h) > 0 else 0.0
            if density > 0.02:
                out.append((x, y, w, h))
        return out

    def _extract_cells_rowmajor(self, bgr_table):
        gray = cv2.cvtColor(bgr_table, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -4)
        h = thr.copy(); v = thr.copy()
        cols, rows = h.shape[1], v.shape[0]
        h_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cols//30), 1))
        v_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, rows//30)))
        h = cv2.erode(h, h_struct); h = cv2.dilate(h, h_struct)
        v = cv2.erode(v, v_struct); v = cv2.dilate(v, v_struct)
        h_proj = np.sum(h > 0, axis=1)
        v_proj = np.sum(v > 0, axis=0)
        r_idxs = [i for i, val in enumerate(h_proj) if val > (0.6 * cols)]
        c_idxs = [j for j, val in enumerate(v_proj) if val > (0.6 * rows)]

        def _collapse(idxs, min_gap=3):
            out, cur = [], []
            for i in idxs:
                if not cur or (i - cur[-1]) <= min_gap:
                    cur.append(i)
                else:
                    out.append(int(np.mean(cur))); cur = [i]
            if cur: out.append(int(np.mean(cur)))
            return out

        r_lines = _collapse(r_idxs); c_lines = _collapse(c_idxs)
        if len(r_lines) < 2 or len(c_lines) < 2:
            return None

        cells = []
        for ri in range(len(r_lines)-1):
            r0, r1 = r_lines[ri], r_lines[ri+1]
            for ci in range(len(c_lines)-1):
                c0, c1 = c_lines[ci], c_lines[ci+1]
                pad_y = max(1, int(0.01 * (r1 - r0)))
                pad_x = max(1, int(0.01 * (c1 - c0)))
                y0 = max(0, r0 + pad_y); y1 = max(y0+1, r1 - pad_y)
                x0 = max(0, c0 + pad_x); x1 = max(x0+1, c1 - pad_x)
                cells.append(((y0, y1, x0, x1), (ri, ci)))
        return cells, (len(r_lines)-1), (len(c_lines)-1)

    def _ocr_cell(self, bgr, box):
        (y0, y1, x0, x1) = box
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.fastNlMeansDenoising(g, h=10)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--psm 6"
        txt = pytesseract.image_to_string(bw, lang=(self._ocr_lang or "eng"), config=cfg)
        return (txt or "").strip()

    def _ocr_tables_for_page(self, fitz_page, page_number_1based: int):
        """Return list of {'bbox_pdf':(x0,y0,x1,y1), 'xml': ET.Element('InformalTable')} or []"""
        if not (self._ocr_tables_enabled and self._pdf_path):
            return []
        bgr = self._render_page_bgr_via_pdf2image(page_number_1based)
        if bgr is None:
            return []
        candidates = self._detect_grid_tables(bgr)
        if not candidates:
            return []

        p = fitz_page.rect
        page_w_pt, page_h_pt = float(p.width), float(p.height)
        img_h_px, img_w_px = bgr.shape[:2]
        sx = img_w_px / max(1.0, page_w_pt)
        sy = img_h_px / max(1.0, page_h_pt)

        results = []
        for idx0, (x, y, w, h) in enumerate(sorted(candidates, key=lambda t: (t[1], t[0])), start=1):
            seg = bgr[y:y+h, x:x+w]
            grid = self._extract_cells_rowmajor(seg)
            if not grid:
                continue
            cells, n_rows, n_cols = grid
            grid_text = [["" for _ in range(n_cols)] for __ in range(n_rows)]
            for (box, (ri, ci)) in cells:
                grid_text[ri][ci] = self._ocr_cell(seg, box)

            table_elem = ET.Element("InformalTable", Tableinfo=f"{page_number_1based:02}{idx0:02}")
            tgroup = ET.SubElement(table_elem, "TGroup")
            thead = ET.SubElement(tgroup, "THead")
            hrow = ET.SubElement(thead, "TRow")
            if grid_text:
                for cell_txt in grid_text[0]:
                    ET.SubElement(hrow, "TCell").text = (cell_txt or "").strip()
            tbody = ET.SubElement(tgroup, "TBody")
            for row in (grid_text[1:] if len(grid_text) > 1 else []):
                tr = ET.SubElement(tbody, "TRow")
                for cell_txt in row:
                    ET.SubElement(tr, "TCell").text = (cell_txt or "").strip()

            x0_pt = x / sx; y0_pt = y / sy
            x1_pt = (x + w) / sx; y1_pt = (y + h) / sy
            results.append({"bbox_pdf": (x0_pt, y0_pt, x1_pt, y1_pt), "xml": table_elem})
        return results

    # ---------- Landscape text box extractor ----------
    def _extract_text_boxes_landscape(self, pb_page, fitz_page, table_bboxes):
        """Return list of small text boxes for LANDSCAPE pages (UNROTATED coords)."""
        from config_patterns import HEADER_THRESHOLD, FOOTER_THRESHOLD, GAP_THRESHOLD, VERTICAL_TOLERANCE, WORD_LINE_Y_TOLERANCE
        words = pb_page.extract_words() or []
        lines = self.text_processor.lines_from_words(words, y_tol=WORD_LINE_Y_TOLERANCE)
        lines = [L for L in lines if (L.get('top', 0) >= HEADER_THRESHOLD and L.get('bottom', 0) <= FOOTER_THRESHOLD)]
        chunks = []
        for L in lines:
            chunks.extend(self.text_processor.split_line_on_large_gaps(L, words, gap_threshold=GAP_THRESHOLD, v_tolerance=VERTICAL_TOLERANCE))
        if table_bboxes:
            kept = []
            for C in chunks:
                x0, x1 = C.get('x0', 0), C.get('x1', 0)
                top, bottom = C.get('top', 0), C.get('bottom', 0)
                if any(x0 >= bb[0] and x1 <= bb[2] and top >= bb[1] and bottom <= bb[3] for bb in table_bboxes):
                    continue
                kept.append(C)
            chunks = kept
        chunks.sort(key=lambda c: (round(float(c.get('top', 0)), 1), float(c.get('x0', 0))))
        return chunks

    def _save_landscape_debug_overlay(self, fitz_page, page_number, boxes, table_bboxes, out_path):
        try:
            from PIL import Image, ImageDraw
            import io
            zoom = 2.0
            pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            draw = ImageDraw.Draw(img)
            for b in boxes or []:
                x0 = int(b['x0'] * zoom); y0 = int(b['top'] * zoom)
                x1 = int(b['x1'] * zoom); y1 = int(b['bottom'] * zoom)
                draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 30), width=3)
            for bb in (table_bboxes or []):
                x0 = int(bb[0] * zoom); y0 = int(bb[1] * zoom)
                x1 = int(bb[2] * zoom); y1 = int(bb[3] * zoom)
                draw.rectangle([x0, y0, x1, y1], outline=(40, 120, 240), width=3)
            img.save(out_path)
        except Exception:
            pass

    # synthesize lines from OCR words (for scanned pages)
    def _lines_from_words(self, words, y_tol=3.0):
        if not words:
            return []
        words_sorted = sorted(
            words, key=lambda w: (round(float(w.get('top', 0.0)), 1), float(w.get('x0', 0.0)))
        )
        lines, current = [], []
        def same_line(a, b):
            return abs(float(a.get('top', 0.0)) - float(b.get('top', 0.0))) <= float(y_tol)
        def flush(group):
            if not group:
                return
            group_sorted = sorted(group, key=lambda t: float(t.get('x0', 0.0)))
            x0 = min(float(t['x0']) for t in group_sorted)
            x1 = max(float(t['x1']) for t in group_sorted)
            top = min(float(t.get('top', 0.0)) for t in group_sorted)
            bottom = max(float(t.get('bottom', t.get('top', 0.0) + 10.0)) for t in group_sorted)
            text = " ".join((t.get('text') or "").strip() for t in group_sorted if (t.get('text') or "").strip())
            if text:
                lines.append({'text': text, 'x0': x0, 'x1': x1, 'top': top, 'bottom': bottom})
        for w in words_sorted:
            if not current:
                current = [w]
                continue
            if same_line(current[-1], w):
                current.append(w)
            else:
                flush(current)
                current = [w]
        flush(current)
        return lines

    def _infer_page_columns(self, fitz_page, is_landscape: bool):
        """
        Infer if page has two-column layout.
        - Landscape: use UNROTATED geometry + stronger thresholds + single-column guard.
        - Portrait: keep original behavior (rotated frame, simpler thresholds).
        Returns: (two_col: bool, boundary_x: float)
        """
        if is_landscape:
            page_w, page_h = self._unrotated_wh(fitz_page)
            pad_x = page_w * PAGE_PAD_X_RATIO
            content_l, content_r = pad_x, page_w - pad_x
            blocks = fitz_page.get_text("blocks") or []
            edges = []
            text_block_count = 0
            for b in blocks:
                if len(b) >= 5 and str(b[4]).strip():
                    x0, x1 = float(b[0]), float(b[2])
                    if (x1 >= content_l) and (x0 <= content_r):
                        edges.extend([max(x0, content_l), min(x1, content_r)])
                    text_block_count += 1

            if text_block_count < 6:
                words = fitz_page.get_text("words") or []
                if not words:
                    words = self._ocr_words_cache or []
                norm_words = []
                for w in words:
                    if isinstance(w, dict):
                        x0 = float(w.get('x0', 0.0)); x1 = float(w.get('x1', 0.0))
                    else:
                        if len(w) < 4:
                            continue
                        x0 = float(w[0]); x1 = float(w[2])
                    if (x1 >= content_l) and (x0 <= content_r):
                        norm_words.append(w)
                words = norm_words
                if len(words) >= 40:
                    xs = []
                    x0s, x1s = [], []
                    for w in words:
                        if isinstance(w, dict):
                            x0 = max(float(w.get('x0', 0.0)), content_l)
                            x1 = min(float(w.get('x1', 0.0)), content_r)
                        else:
                            x0 = max(float(w[0]), content_l)
                            x1 = min(float(w[2]), content_r)
                        xs.append((x0 + x1) / 2.0)
                        x0s.append(x0)
                        x1s.append(x1)
                    content_w = max(1.0, content_r - content_l)
                    mid = content_l + content_w * 0.5
                    left = sum(1 for x in xs if x < mid * 0.99)
                    right = sum(1 for x in xs if x > mid * 1.01)
                    try:
                        spans = sorted((x1s[i] - x0s[i]) for i in range(len(x0s)))
                        median_span = spans[len(spans)//2] if spans else 0.0
                        if (median_span / content_w) >= 0.60:
                            return False, content_l + content_w / 2.0
                    except Exception:
                        pass
                    if left >= 0.3 * len(xs) and right >= 0.3 * len(xs):
                        word_edges = sorted(set(x0s + x1s))
                        mid_l = content_l + content_w * 0.25
                        mid_r = content_l + content_w * 0.75
                        max_gap, boundary = 0.0, content_l + content_w / 2.0
                        for i in range(len(word_edges) - 1):
                            g0, g1 = word_edges[i], word_edges[i + 1]
                            center, size = (g0 + g1) / 2.0, (g1 - g0)
                            if mid_l <= center <= mid_r and size > max_gap:
                                max_gap, boundary = size, center
                        return True, boundary
                return False, content_l + (content_r - content_l) / 2.0

            edges = sorted(set(edges))
            content_w = max(1.0, content_r - content_l)
            mid_l = content_l + content_w * 0.25
            mid_r = content_l + content_w * 0.75
            max_gap = 0.0
            boundary = content_l + content_w / 2.0
            for i in range(len(edges) - 1):
                g0, g1 = edges[i], edges[i + 1]
                center = (g0 + g1) / 2.0
                size = g1 - g0
                if mid_l <= center <= mid_r and size > max_gap:
                    max_gap = size
                    boundary = center
            left = right = 0
            for b in blocks:
                if len(b) >= 5 and str(b[4]).strip():
                    x0, x1 = float(b[0]), float(b[2])
                    if x1 < content_l or x0 > content_r:
                        continue
                    cx = (max(x0, content_l) + min(x1, content_r)) / 2.0
                    if cx < boundary:
                        left += 1
                    else:
                        right += 1
            min_blocks = 3
            gap_thresh = content_w * 0.08
            balance_ok = (left >= min_blocks and right >= min_blocks)
            two_col = bool(balance_ok and (max_gap >= gap_thresh))
            return two_col, boundary

        r = fitz_page.rect
        page_w, page_h = float(r.width), float(r.height)
        pad_x = page_w * PAGE_PAD_X_RATIO
        content_l, content_r = pad_x, page_w - pad_x
        blocks = fitz_page.get_text("blocks") or []
        edges = []
        for b in blocks:
            if len(b) >= 5 and str(b[4]).strip():
                x0, x1 = float(b[0]), float(b[2])
                if (x1 >= content_l) and (x0 <= content_r):
                    edges.extend([max(x0, content_l), min(x1, content_r)])
        edges = sorted(set(edges))
        content_w = max(1.0, content_r - content_l)
        mid_l = content_l + content_w * 0.25
        mid_r = content_l + content_w * 0.75
        max_gap = 0.0
        boundary = content_l + content_w / 2.0
        for i in range(len(edges) - 1):
            g0, g1 = edges[i], edges[i + 1]
            center = (g0 + g1) / 2.0
            size = g1 - g0
            if mid_l <= center <= mid_r and size > max_gap:
                max_gap = size
                boundary = center
        left = right = 0
        for b in blocks:
            if len(b) >= 5 and str(b[4]).strip():
                x0, x1 = float(b[0]), float(b[2])
                if x1 < content_l or x0 > content_r:
                    continue
                cx = (max(x0, content_l) + min(x1, content_r)) / 2.0
                if cx < boundary:
                    left += 1
                else:
                    right += 1
        min_blocks = 3
        gap_thresh = content_w * 0.06
        two_col = (left >= min_blocks and right >= min_blocks and max_gap >= gap_thresh)
        return two_col, boundary

    def process_single_page(
        self,
        page,
        page_elem,
        page_number,
        fitz_page,
        section_page_mapping,
        toc_extractorr=None,
        is_scanned: bool = False
    ):
        """
        Process a single page:
        1) Collect all content items (tables, images, text blocks) into a single list
        2) Sort by visual reading order (top-to-bottom, left-to-right)
        3) Append sorted items to XML
        """
        section_info = section_page_mapping.get(page_number, {})
        target_elem = section_info.get('page_element', page_elem)

        u_w, u_h = self._unrotated_wh(fitz_page)
        is_landscape = (u_w > u_h)
        if is_landscape:
            page_w, page_h = u_w, u_h
        else:
            r = fitz_page.rect
            page_w, page_h = float(r.width), float(r.height)

        header_pct = 0.15 if is_landscape else 0.12
        dyn_header = min(HEADER_THRESHOLD, page_h * header_pct)
        dyn_footer = min(page_h - 1.0, max(page_h * 0.88, min(FOOTER_THRESHOLD, page_h * 0.95)))

        xml_gen = XMLGenerator(self.image_output_dir)

        words_all = page.extract_words() or []

        fitz_blocks_probe = fitz_page.get_text("blocks") or []
        self._ocr_words_cache = words_all if (not fitz_blocks_probe and words_all) else None

        page_two_col, page_boundary = self._infer_page_columns(fitz_page, is_landscape)
        self._ocr_words_cache = None

        page_content_items = []

        # === 1) TABLES ===
        had_ocr_tables = False
        ocr_table_bboxes = []
        if is_scanned:
            try:
                ocr_tables = self._ocr_tables_for_page(fitz_page, page_number)
            except Exception:
                ocr_tables = []
            if ocr_tables:
                for res in ocr_tables:
                    table_elem = res.get("xml")
                    x0, top, x1, bottom = res.get("bbox_pdf", (0, 0, 0, 0))
                    x_mid = (x0 + x1) / 2.0
                    if table_elem is not None:
                        page_content_items.append((float(top), float(x_mid), table_elem))
                        ocr_table_bboxes.append((float(x0), float(top), float(x1), float(bottom)))
                had_ocr_tables = True

        if had_ocr_tables:
            table_bboxes = ocr_table_bboxes
        else:
            table_bboxes = self.table_extractor.detect_tables(page)
            table_elements = self.table_extractor.extract_tables(page, page_number)
            for i, table_elem in enumerate(table_elements):
                if i < len(table_bboxes):
                    x0, top, x1, bottom = table_bboxes[i]
                    x_mid = (x0 + x1) / 2.0
                    page_content_items.append((top, x_mid, table_elem))
                else:
                    page_content_items.append((0, page_boundary, table_elem))

        # === 2) IMAGES ===
        page_images = self.image_map.get(page_number, [])
        pymu_images = fitz_page.get_images(full=True) or []
        emitted_any = False

        if page_images and pymu_images:
            filenames = [fn for fn, _ in page_images]
            xref_map = self.image_xref_map.get(page_number, {})

            raw_image_blocks = []
            try:
                raw = fitz_page.get_text("rawdict")
            except Exception:
                raw = fitz_page.get_text("dict")
            if isinstance(raw, dict):
                block_seq = 0
                for block in raw.get("blocks", []):
                    if block.get("type") == 1:  # image
                        bbox = block.get("bbox") or block.get("rect")
                        xref_val = block.get("xref")
                        if xref_val is None:
                            img_obj = block.get("image")
                            if isinstance(img_obj, dict):
                                xref_val = img_obj.get("xref")
                        if bbox and len(bbox) >= 4 and xref_val is not None:
                            try:
                                left = float(min(bbox[0], bbox[2]))
                                right = float(max(bbox[0], bbox[2]))
                                top = float(min(bbox[1], bbox[3]))
                                raw_image_blocks.append({
                                    "xref": int(xref_val),
                                    "x0": left, "x1": right, "y0": top,
                                    "seq": block_seq,
                                })
                                block_seq += 1
                            except Exception:
                                pass

            placed_images = []
            for img_idx, img in enumerate(pymu_images):
                try:
                    xref_val = int(img[0])
                except Exception:
                    xref_val = None
                rects = []
                try:
                    if xref_val is not None:
                        rects = fitz_page.get_image_rects(xref_val) or []
                except Exception:
                    rects = []
                for rect in rects:
                    try:
                        left = float(min(rect.x0, rect.x1))
                        right = float(max(rect.x0, rect.x1))
                        top = float(min(rect.y0, rect.y1))
                        placed_images.append({
                            "xref": xref_val,
                            "x0": left, "x1": right, "y0": top,
                            "img_idx": img_idx,
                        })
                    except Exception:
                        continue

            placed_images.sort(key=lambda p: (p.get("y0", 0.0), ((p.get("x0", 0.0) + p.get("x1", 0.0)) / 2.0)))

            for occ_idx, placed in enumerate(placed_images):
                xref_val = placed["xref"]
                resolved_fname = xref_map.get(xref_val)
                if not resolved_fname:
                    fallback_idx = placed.get("img_idx", 0)
                    if fallback_idx < len(filenames):
                        resolved_fname = filenames[fallback_idx]
                if not resolved_fname:
                    resolved_fname = f"page{page_number}_img{occ_idx+1}.img"

                # skip embedded image overlapping OCR table bbox
                if had_ocr_tables:
                    y_top = placed.get("y0", 0.0)
                    x_mid_center = (placed.get("x0", 0.0) + placed.get("x1", 0.0)) / 2.0
                    skip_img = False
                    for bx0, by0, bx1, by1 in ocr_table_bboxes:
                        if (bx0 - 1) <= x_mid_center <= (bx1 + 1) and (by0 - 1) <= y_top <= (by1 + 1):
                            skip_img = True
                            break
                    if skip_img:
                        continue

                img_elem = xml_gen.create_image_element(page_number, occ_idx, resolved_fname)
                x_mid = (placed.get("x0", 0.0) + placed.get("x1", 0.0)) / 2.0
                page_content_items.append((placed["y0"], x_mid, img_elem))
                emitted_any = True

        # CV fallback images (skip grid-like table images even in mixed)
        if not emitted_any and not had_ocr_tables:
            crops = (self.cv_table_crops_map or {}).get(page_number, [])

            grid_table_bboxes_pdf = []
            try:
                bgr = self._render_page_bgr_via_pdf2image(page_number)
                if bgr is not None:
                    grid_tables_xywh = self._detect_grid_tables(bgr)
                    if grid_tables_xywh:
                        p = fitz_page.rect
                        page_w_pt, page_h_pt = float(p.width), float(p.height)
                        img_h_px, img_w_px = bgr.shape[:2]
                        sx = img_w_px / max(1.0, page_w_pt)
                        sy = img_h_px / max(1.0, page_h_pt)
                        for (gx, gy, gw, gh) in grid_tables_xywh:
                            gx0 = gx / sx; gy0 = gy / sy
                            gx1 = (gx + gw) / sx; gy1 = (gy + gh) / sy
                            grid_table_bboxes_pdf.append((gx0, gy0, gx1, gy1))
            except Exception:
                grid_table_bboxes_pdf = []

            def _overlaps(a, b, tol=1.0):
                ax0, ay0, ax1, ay1 = a
                bx0, by0, bx1, by1 = b
                if (ax1 < bx0 - tol) or (bx1 < ax0 - tol):
                    return False
                if (ay1 < by0 - tol) or (by1 < ay0 - tol):
                    return False
                return True

            for idx, crop in enumerate(crops):
                fname = crop.get('filename')
                cx0, cy0, cx1, cy1 = crop.get('bbox_pdf', (0, 0, 0, 0))

                skip = False
                if grid_table_bboxes_pdf:
                    for (gx0, gy0, gx1, gy1) in grid_table_bboxes_pdf:
                        if _overlaps((cx0, cy0, cx1, cy1), (gx0, gy0, gx1, gy1)):
                            skip = True
                            break
                if skip:
                    continue

                x_mid = (cx0 + cx1) / 2.0
                img_elem = xml_gen.create_image_element(page_number, idx, fname)
                page_content_items.append((float(cy0), float(x_mid), img_elem))
            emitted_any = True

        # === 3) TEXT ===
        fitz_blocks = fitz_page.get_text("blocks") or []
        text_blocks = [b for b in fitz_blocks if len(b) >= 5 and str(b[4]).strip()]
        USE_OCR_LINES = (len(text_blocks) < 3) and (len(words_all or []) >= 20)
        clusters = self.layout_analyzer.cluster_blocks_vertically(fitz_blocks) if not USE_OCR_LINES else []
        if (USE_OCR_LINES or (not clusters)) and words_all:
            ocr_lines = self._lines_from_words(words_all, y_tol=3.0)
            if ocr_lines:
                pseudo_blocks = [(ln['x0'], ln['top'], ln['x1'], ln['bottom'], ln['text']) for ln in ocr_lines]
                clusters = [pseudo_blocks]

        if is_landscape:
            containers = []
            landscape_boxes = self._extract_text_boxes_landscape(page, fitz_page, table_bboxes)
            if landscape_boxes:
                cluster_top = min(L.get('top', 0) for L in landscape_boxes)
                containers.append((cluster_top, ('ParaLines', landscape_boxes)))
            try:
                dbg = os.path.join(self.image_output_dir, f"page{page_number}_lines_boxes.png")
                self._save_landscape_debug_overlay(fitz_page, page_number, landscape_boxes, table_bboxes, dbg)
            except Exception:
                pass
        else:
            containers = []
            for cluster in clusters:
                try:
                    cluster_lines = self.layout_analyzer.convert_blocks_to_lines(cluster)
                    if not cluster_lines:
                        continue
                    filtered_lines = []
                    for L in cluster_lines:
                        line_x0, line_y0 = L.get('x0', 0), L.get('top', 0)
                        line_x1, line_y1 = L.get('x1', 0), L.get('bottom', 0)
                        is_in_table = False
                        for tbx0, tby0, tbx1, tby1 in table_bboxes:
                            cx = (line_x0 + line_x1) / 2.0
                            cy = (line_y0 + line_y1) / 2.0
                            if tbx0 <= cx <= tbx1 and tby0 <= cy <= tby1:
                                is_in_table = True
                                break
                        if not is_in_table:
                            filtered_lines.append(L)
                    if not filtered_lines:
                        continue
                    cluster_top = min(L.get('top', 0) for L in filtered_lines)
                    if page_two_col:
                        boundary = page_boundary
                        left_lines, right_lines = [], []
                        rel_margin = max(10.0, page_w * 0.02)
                        for L in filtered_lines:
                            l0, l1 = L.get('x0', 0), L.get('x1', 0)
                            lc = (l0 + l1) / 2.0
                            if lc < boundary - rel_margin:
                                left_lines.append(L)
                            elif lc > boundary + rel_margin:
                                right_lines.append(L)
                            else:
                                (left_lines if l0 < boundary else right_lines).append(L)
                        if left_lines:
                            left_lines.sort(key=lambda L: (round(L.get('top', 0), 1), round(L.get('x0', 0), 1)))
                            containers.append((min(L.get('top', 0) for L in left_lines), ('LeftColumn', left_lines)))
                        if right_lines:
                            right_lines.sort(key=lambda L: (round(L.get('top', 0), 1), round(L.get('x0', 0), 1)))
                            containers.append((min(L.get('top', 0) for L in right_lines), ('RightColumn', right_lines)))
                    else:
                        filtered_lines.sort(key=lambda L: (round(L.get('top', 0), 1), round(L.get('x0', 0), 1)))
                        containers.append((cluster_top, ('ParaLines', filtered_lines)))
                except Exception as e:
                    if DEBUG:
                        print(f"Error processing cluster on page {page_number}: {e}")

        if containers:
            merged_containers = XMLGenerator(self.image_output_dir).merge_adjacent_containers(containers)
            for top, (tag, payload) in merged_containers:
                try:
                    x_mid = (
                        (min(L.get('x0', 0) for L in payload) + max(L.get('x1', 0) for L in payload)) / 2.0
                    ) if payload else page_boundary
                    temp_parent = ET.Element("temp")
                    if tag == 'ParaLines':
                        self._process_text_container(
                            page, payload, temp_parent, page_number,
                            table_bboxes, words_all, toc_extractorr,
                            max(0.0, dyn_header * 0.90),
                            min(page_h - 1.0, dyn_footer * 1.02)
                        )
                        for child_elem in list(temp_parent):
                            page_content_items.append((top, x_mid, child_elem))
                    else:
                        col_elem = ET.SubElement(temp_parent, tag)
                        self._process_text_container(
                            page, payload, col_elem, page_number,
                            table_bboxes, words_all, toc_extractorr,
                            max(0.0, dyn_header * 0.90),
                            min(page_h - 1.0, dyn_footer * 1.02)
                        )
                        if len(col_elem) > 0:
                            page_content_items.append((top, x_mid, col_elem))
                except Exception as e:
                    if DEBUG:
                        print(f"Error processing text container on page {page_number}: {e}")

        # --- Emit to XML ---
        if is_landscape and page_content_items:
            heads, rest = [], []
            for y, xk, el in page_content_items:
                if el.tag in ('title', 'Subtitle'):
                    heads.append((y, xk, el))
                else:
                    rest.append((y, xk, el))
            heads.sort(key=lambda it: (it[0], it[1]))
            for _, _, el in heads:
                target_elem.append(el)
            page_content_items = rest

        if not page_two_col:
            page_content_items.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
            for _, _, element in page_content_items:
                target_elem.append(element)
        else:
            left_items, right_items = [], []
            rel_margin = max(10.0, page_w * 0.02)
            for y, x_key, elem in page_content_items:
                if elem.tag == 'LeftColumn':
                    left_items.append((y, x_key, elem)); continue
                if elem.tag == 'RightColumn':
                    right_items.append((y, x_key, elem)); continue
                if x_key <= (page_boundary - rel_margin):
                    left_items.append((y, x_key, elem))
                elif x_key >= (page_boundary + rel_margin):
                    right_items.append((y, x_key, elem))
                else:
                    (left_items if abs(x_key - (page_boundary - rel_margin)) <= abs(x_key - (page_boundary + rel_margin))
                     else right_items).append((y, x_key, elem))
            left_items.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
            right_items.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
            left_parent = ET.SubElement(target_elem, "LeftColumn")
            right_parent = ET.SubElement(target_elem, "RightColumn")
            def _append_to(parent, items):
                for _, _, el in items:
                    if el.tag in ("LeftColumn", "RightColumn"):
                        for child in list(el):
                            parent.append(child)
                    else:
                        parent.append(el)
            _append_to(left_parent, left_items)
            _append_to(right_parent, right_items)

    def _process_text_container(
        self,
        page,
        lines,
        container_elem,
        page_number,
        table_bboxes,
        words_all,
        toc_extractorr,
        dyn_header,
        dyn_footer,
    ):
        """Process text container with filtering and content generation."""
        try:
            lines = sorted(lines, key=lambda L: (L.get('top', 0), L.get('x0', 0)))
            lines_expanded = self.text_processor.expand_lines_with_gap_splitting(lines, words_all, table_bboxes)
            lenient_header = dyn_header
            lenient_footer = dyn_footer
            filtered_lines = []
            for line in lines_expanded:
                line_top = line.get('top', 0)
                line_bottom = line.get('bottom', line_top)
                if line_bottom < lenient_header:
                    if DEBUG:
                        print(f"DEBUG: Filtered header line p{page_number}: {line.get('text', '')[:50]}")
                    continue
                if line_top > lenient_footer:
                    if DEBUG:
                        print(f"DEBUG: Filtered footer line p{page_number}: {line.get('text', '')[:50]}")
                    continue
                filtered_lines.append(line)

            if lines_expanded and len(filtered_lines) < max(2, int(0.2 * len(lines_expanded))):
                if isinstance(words_all, list) and len(words_all) >= 50:
                    if DEBUG:
                        print(f"DEBUG: Fail-safe on p{page_number}: kept {len(filtered_lines)} / {len(lines_expanded)}")
                    filtered_lines = list(lines_expanded)

            if filtered_lines:
                XMLGenerator(self.image_output_dir).process_lines_container(
                    page, filtered_lines, container_elem, page_number,
                    table_bboxes, self.content_detector, toc_extractorr
                )
        except Exception as e:
            if DEBUG:
                print(f"Error in text container processing for page {page_number}: {e}")
            import traceback
            traceback.print_exc()