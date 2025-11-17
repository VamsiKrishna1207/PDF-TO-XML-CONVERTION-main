# -*- coding: utf-8 -*-
"""
Layout analysis, text processing, and table extraction module
"""
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from config_patterns import (
    Y_CLUSTER_THRESHOLD, FULL_WIDTH_THRESHOLD,
    GAP_THRESHOLD, VERTICAL_TOLERANCE,
    WORD_LINE_Y_TOLERANCE,
)

class LayoutAnalyzer:
    """Handles page layout analysis and column detection"""
    def __init__(self):
        pass

    def cluster_blocks_vertically(self, blocks, y_thresh=Y_CLUSTER_THRESHOLD):
        """Cluster blocks vertically based on proximity"""
        if not blocks:
            return []
        clusters, current = [], []
        for b in sorted(blocks, key=lambda x: x[1]):
            if not current:
                current = [b]
                continue
            prev_y1 = max(bl[3] for bl in current)
            if b[1] - prev_y1 <= y_thresh:
                current.append(b)
            else:
                clusters.append(current)
                current = [b]
        if current:
            clusters.append(current)
        return clusters

    def classify_cluster(self, cluster, page_width, full_width_thresh=FULL_WIDTH_THRESHOLD):
        if not cluster:
            return {'layout': 'single', 'confidence': 0.0}
        widths = [b[2] - b[0] for b in cluster]
        x_centers = [(b[0] + b[2]) / 2.0 for b in cluster]
        avg_width = sum(widths) / len(widths)
        mid_x = page_width / 2
        left = sum(1 for x in x_centers if x < mid_x * 0.8)
        right = sum(1 for x in x_centers if x > mid_x * 1.2)
        center = len(cluster) - left - right

        if avg_width / page_width > full_width_thresh:
            return {'layout': 'single', 'confidence': 1.0}
        if left > 0.3 * len(cluster) and right > 0.3 * len(cluster) and center < 0.4 * len(cluster):
            return {'layout': 'double', 'confidence': 0.9, 'boundary_x': self.estimate_column_boundary(cluster, page_width)}
        return {'layout': 'mixed', 'confidence': 0.7, 'boundary_x': self.estimate_column_boundary(cluster, page_width)}

    def estimate_column_boundary(self, cluster, page_width):
        if not cluster:
            return page_width / 2
        edges = sorted(set([val for b in cluster for val in (b[0], b[2])]))
        mid_l, mid_r = page_width * 0.25, page_width * 0.75
        max_gap, boundary = 0, page_width / 2
        for i in range(len(edges) - 1):
            g0, g1 = edges[i], edges[i + 1]
            center, size = (g0 + g1) / 2.0, g1 - g0
            if mid_l <= center <= mid_r and size > max_gap:
                max_gap, boundary = size, center
        return boundary

    def convert_blocks_to_lines(self, blocks):
        """Approximate lines from PyMuPDF blocks (kept for fallback)"""
        lines = []
        for block in blocks:
            if len(block) >= 5 and str(block[4]).strip():
                x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                text_lines = str(text).strip().split('\n')
                line_height = (y1 - y0) / max(1, len(text_lines))
                for i, line_text in enumerate(text_lines):
                    if line_text.strip():
                        line_y0 = y0 + i * line_height
                        line_y1 = line_y0 + line_height
                        lines.append({
                            'text': line_text.strip(),
                            'x0': x0, 'x1': x1,
                            'top': line_y0, 'bottom': line_y1
                        })
        return lines


class TextProcessor:
    """Handles text processing and line analysis"""
    def __init__(self):
        pass

    def lines_from_words(self, words, y_tol=WORD_LINE_Y_TOLERANCE):
        """
        Build true line boxes from pdfplumber words in UNROTATED page space.
        Each output line: {'text','x0','x1','top','bottom'}
        """
        if not words:
            return []
        # keep only non-empty texts
        words = [w for w in words if (w.get("text") or "").strip()]
        # stable grouping by (top, x0)
        words_sorted = sorted(
            words,
            key=lambda w: (round(float(w.get("top", 0.0)), 1), float(w.get("x0", 0.0)))
        )
        lines, group = [], []

        def same_line(a, b):
            return abs(float(a.get("top", 0.0)) - float(b.get("top", 0.0))) <= float(y_tol)

        def flush():
            nonlocal group
            if not group:
                return
            x0 = min(float(w["x0"]) for w in group)
            x1 = max(float(w["x1"]) for w in group)
            top = min(float(w.get("top", 0.0)) for w in group)
            bottom = max(float(w.get("bottom", w.get("top", 0.0) + 10.0)) for w in group)
            text = " ".join((w.get("text") or "").strip() for w in group if (w.get("text") or "").strip())
            if text:
                lines.append({"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom})
            group = []

        for w in words_sorted:
            if not group:
                group = [w]; continue
            if same_line(group[-1], w):
                group.append(w)
            else:
                flush(); group = [w]
        flush()
        return lines

    def split_line_on_large_gaps(self, text_line, words, gap_threshold=GAP_THRESHOLD, v_tolerance=VERTICAL_TOLERANCE):
        """(Fallback) Split a block line on large gaps between words"""
        lx0, ltop, lx1, lbottom = text_line.get('x0', 0), text_line.get('top', 0), text_line.get('x1', 0), text_line.get('bottom', 0)
        inside = [w for w in words if
                  (ltop - v_tolerance) <= w.get('top', 0) <= (lbottom + v_tolerance) and
                  (w.get('x0', 0) >= lx0 - 1) and (w.get('x1', 0) <= lx1 + 1)]
        if not inside:
            return [text_line]
        inside = sorted(inside, key=lambda w: w['x0'])
        clusters, current = [], [inside[0]]
        for w in inside[1:]:
            gap = w['x0'] - current[-1]['x1']
            if gap > gap_threshold:
                clusters.append(current); current = [w]
            else:
                current.append(w)
        clusters.append(current)

        if len(clusters) <= 1:
            return [text_line]
        result = []
        for cluster in clusters:
            cx0 = min(w['x0'] for w in cluster)
            cx1 = max(w['x1'] for w in cluster)
            ctop = min(w['top'] for w in cluster)
            cbottom = max(w.get('bottom', w.get('top', 0) + 10) for w in cluster)
            ctext = " ".join((w.get('text') or '').strip() for w in cluster if (w.get('text') or '').strip())
            result.append({'text': ctext, 'x0': cx0, 'x1': cx1, 'top': ctop, 'bottom': cbottom})
        return result

    def filter_lines_by_position(self, lines, header_threshold, footer_threshold):
        return [L for L in lines if not (L.get('top', 0) < header_threshold or L.get('bottom', 0) > footer_threshold)]

    def expand_lines_with_gap_splitting(self, lines, words, table_bboxes):
        """Fallback: expand lines by splitting on large gaps and skip table-overlapping lines"""
        out = []
        for L in lines:
            x0, x1, top, bottom = L.get('x0', 0), L.get('x1', 0), L.get('top', 0), L.get('bottom', 0)
            if any(x0 >= bb[0] and x1 <= bb[2] and top >= bb[1] and bottom <= bb[3] for bb in table_bboxes):
                continue
            out.extend(self.split_line_on_large_gaps(L, words))
        return out


class TableExtractor:
    """Handles table detection and extraction from PDF pages"""
    def __init__(self):
        pass

    def detect_tables(self, page):
        """Detect tables on a page and return bounding boxes"""
        table_bboxes = []
        tables = page.find_tables() or []
        if not tables:
            raw = page.extract_tables()
            if raw:
                words = page.extract_words()
                for table in raw:
                    if not any(any(cell and str(cell).strip() for cell in row) for row in table):
                        continue
                    num_rows, num_cols = len(table), max((len(r) for r in table), default=0)
                    filled = sum(1 for r in table for cell in r if cell and str(cell).strip())
                    if num_rows < 1 or num_cols < 1 or filled <= 2:
                        continue
                    twords = []
                    for row in table:
                        for cell in row:
                            if not cell: continue
                            sc = str(cell).strip()
                            for w in words:
                                if str(w["text"]).strip() in sc:
                                    twords.append(w)
                    if twords:
                        x0 = min(w["x0"] for w in twords); x1 = max(w["x1"] for w in twords)
                        top = min(w["top"] for w in twords); bottom = max(w["bottom"] for w in twords)
                        table_bboxes.append((x0, top, x1, bottom))
        else:
            for t in tables:
                try:
                    tbl = t.extract()
                except Exception:
                    tbl = None
                if not tbl or not any(any(cell and str(cell).strip() for cell in row) for row in tbl):
                    continue
                num_rows, num_cols = len(tbl), max((len(r) for r in tbl), default=0)
                filled = sum(1 for r in tbl for cell in r if cell and str(cell).strip())
                if num_rows < 1 or num_cols < 1 or filled <= 2:
                    continue
                bbox = getattr(t, "bbox", None)
                if bbox:
                    table_bboxes.append(tuple(bbox))
        return table_bboxes

    def extract_tables(self, page, page_number):
        """Extract tables and convert to XML elements"""
        out = []
        objs = page.find_tables() or []
        if objs:
            for idx, t in enumerate(objs, start=1):
                try:
                    table = t.extract()
                except Exception:
                    table = None
                if not table or not any(any(cell and str(cell).strip() for cell in row) for row in table):
                    continue
                num_rows, num_cols = len(table), max((len(r) for r in table), default=0)
                filled = sum(1 for r in table for cell in r if cell and str(cell).strip())
                if num_rows < 1 or num_cols < 1 or filled <= 2:
                    continue
                table_elem = ET.Element("InformalTable", Tableinfo=f"{page_number:02}{idx:02}")
                tgroup = ET.SubElement(table_elem, "TGroup")
                thead = ET.SubElement(tgroup, "THead")
                hrow = ET.SubElement(thead, "TRow")
                for cell in (table[0] if table else []):
                    ET.SubElement(hrow, "TCell").text = (cell or "").strip()
                tbody = ET.SubElement(tgroup, "TBody")
                for row in table[1:]:
                    tr = ET.SubElement(tbody, "TRow")
                    for cell in row:
                        ET.SubElement(tr, "TCell").text = (cell or "").strip()
                out.append(table_elem)
            return out

        # fallback extract_tables
        for idx, table in enumerate(page.extract_tables() or [], start=1):
            if not any(any(cell and str(cell).strip() for cell in row) for row in table):
                continue
            num_rows, num_cols = len(table), max((len(r) for r in table), default=0)
            filled = sum(1 for r in table for cell in r if cell and str(cell).strip())
            if num_rows < 1 or num_cols < 1 or filled <= 2:
                continue
            table_elem = ET.Element("InformalTable", Tableinfo=f"{page_number:02}{idx:02}")
            tgroup = ET.SubElement(table_elem, "TGroup")
            thead = ET.SubElement(tgroup, "THead")
            hrow = ET.SubElement(thead, "TRow")
            for cell in (table[0] if table else []):
                ET.SubElement(hrow, "TCell").text = (cell or "").strip()
            tbody = ET.SubElement(tgroup, "TBody")
            for row in table[1:]:
                tr = ET.SubElement(tbody, "TRow")
                for cell in row:
                    ET.SubElement(tr, "TCell").text = (cell or "").strip()
            out.append(table_elem)
        return out

    def line_overlaps_table(self, line_bbox, table_bboxes):
        lx0, ltop, lx1, lbottom = line_bbox
        for tx0, ttop, tx1, tbottom in table_bboxes:
            if not (lx1 < tx0 + 1 or lx0 > tx1 - 1 or lbottom < ttop + 1 or ltop > tbottom - 1):
                return True
        return False