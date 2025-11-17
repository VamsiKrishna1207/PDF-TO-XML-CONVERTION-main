#!/usr/bin/env python3
# XML → HTML converter (built-in paths, no CLI args)

from xml.etree import ElementTree as ET
import html
from pathlib import Path
from typing import Optional, List

# ---- Edit these paths ----
INPUT_XML = Path(r"C:\Users\vamsik\Desktop\PDF-TO-XML-CONVERTION-main\temp_xml_dir\Sample for XML convertion.xml")      # input XML file
OUTPUT_HTML = Path("Converted_output_final.html")  
# --------------------------

def strip_ns(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag

def attrs_to_data_attrs(attrib: dict) -> str:
    return "".join(
        f' data-{html.escape(strip_ns(k), quote=True)}="{html.escape(v, quote=True)}"'
        for k, v in attrib.items()
    )

def text_to_html(s: Optional[str], preserve_breaks: bool = False) -> str:
    if not s:
        return ""
    t = html.escape(s)
    return t.replace("\n", "<br>") if preserve_breaks else t

def node_inner_text(node: ET.Element) -> str:
    parts: List[str] = []
    if node.text: parts.append(node.text)
    for c in node:
        if c.text: parts.append(c.text)
        if c.tail: parts.append(c.tail)
    return "".join(parts).strip()

# ---------- LISTS: no browser numbering ----------
def render_ordered_list(ol_el: ET.Element) -> str:
    # We render as <ol class="no-num"> with CSS list-style:none so only the
    # existing numbers in the XML text appear (no double “1. 1.”).
    items_html: List[str] = []
    for li in ol_el:
        if strip_ns(li.tag).lower() == "listitem":
            parts = []
            for sub in li:
                sub_name = strip_ns(sub.tag).lower()
                if sub_name == "para":
                    parts.append(f"<p>{text_to_html(node_inner_text(sub), preserve_breaks=True)}</p>")
                else:
                    parts.append(render_element(sub))
            if not parts and li.text:
                parts.append(f"<p>{text_to_html(li.text, preserve_breaks=True)}</p>")
            items_html.append(f"<li>{''.join(parts)}</li>")
    return '<ol class="no-num">' + "".join(items_html) + "</ol>"
# -------------------------------------------------

def render_table(table_el: ET.Element) -> str:
    rows_head, rows_body = [], []
    tgroup = next((c for c in table_el if strip_ns(c.tag).lower() == "tgroup"), None)

    def render_row(row_el, cell_tag: str) -> str:
        cells = []
        for cell in row_el:
            if strip_ns(cell.tag).lower() == "tcell":
                content = text_to_html(node_inner_text(cell), preserve_breaks=True)
                cells.append(f"<{cell_tag}>{content}</{cell_tag}>")
        return f"<tr>{''.join(cells)}</tr>"

    if tgroup is not None:
        for sec in tgroup:
            name = strip_ns(sec.tag).lower()
            if name == "thead":
                for row in sec:
                    if strip_ns(row.tag).lower() == "trow":
                        rows_head.append(render_row(row, "th"))
            elif name == "tbody":
                for row in sec:
                    if strip_ns(row.tag).lower() == "trow":
                        rows_body.append(render_row(row, "td"))

    if not rows_head and not rows_body:
        for row in table_el.iter():
            if strip_ns(row.tag).lower() == "trow":
                rows_body.append(render_row(row, "td"))

    thead = f"<thead>{''.join(rows_head)}</thead>" if rows_head else ""
    tbody = f"<tbody>{''.join(rows_body)}</tbody>" if rows_body else ""
    return f"<table>{thead}{tbody}</table>"

# ---------- PAGES: render single column ----------
def render_page(page_el: ET.Element) -> str:
    number = page_el.attrib.get("number", "")
    left = next((c for c in page_el if strip_ns(c.tag).lower() == "leftcolumn"), None)
    right = next((c for c in page_el if strip_ns(c.tag).lower() == "rightcolumn"), None)

    # Concatenate left then right into ONE flow (no grid)
    if left is not None or right is not None:
        left_html = render_children(left) if left is not None else ""
        right_html = render_children(right) if right is not None else ""
        content = left_html + right_html
        return f'<section class="page" data-page="{html.escape(number, quote=True)}">{content}</section>'

    return f'<section class="page" data-page="{html.escape(number, quote=True)}">{render_children(page_el)}</section>'
# ------------------------------------------------

def render_children(el: Optional[ET.Element]) -> str:
    if el is None: return ""
    return "".join(render_element(child) for child in el)

def render_element(el: ET.Element) -> str:
    name = strip_ns(el.tag).lower()
    data = attrs_to_data_attrs(el.attrib)

    if name in ("document", "sect1"):
        return f'<div class="{name}"{data}>' + render_children(el) + "</div>"
    if name == "page":
        return render_page(el)
    if name in ("leftcolumn", "rightcolumn"):
        return f'<div class="{name}">' + render_children(el) + "</div>"
    if name == "subtitle":
        return f"<h3{data}>" + text_to_html(node_inner_text(el), preserve_breaks=True) + "</h3>"
    if name == "title":
        return f"<h4 class='title'{data}>" + text_to_html(node_inner_text(el), preserve_breaks=True) + "</h4>"
    if name == "para":
        return f"<p{data}>" + text_to_html(node_inner_text(el), preserve_breaks=True) + "</p>"
    if name in ("orderedlist", "ol"):
        return render_ordered_list(el)
    if name in ("informaltable", "table"):
        return render_table(el)
    if name in ("informalfigure", "figure"):
        # very simple figure renderer that keeps any graphic src
        imgs = []
        for ch in el:
            if strip_ns(ch.tag).lower() == "graphic":
                src = ch.attrib.get("entityref", "") or ch.attrib.get("entity", "")
                alt = ch.attrib.get("entity", "") or ""
                imgs.append(f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">')
        return "<figure>" + ("".join(imgs) or render_children(el)) + "</figure>"

    inner = text_to_html(el.text or "", preserve_breaks=True) + render_children(el)
    tail = text_to_html(el.tail or "", preserve_breaks=True)
    return f"<div class='unknown'{data} data-xml-tag='{html.escape(name, quote=True)}'>{inner}</div>{tail}"

def convert_xml_to_html(xml_path: Path) -> str:
    body_html = render_element(ET.parse(xml_path).getroot())
    css = """
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height: 1.4; margin: 0; padding: 1rem; }
.page { padding: 0 0 1rem; }
h3 { margin: 0.6rem 0 0.3rem; }
h4.title { margin: 0.8rem 0 0.3rem; text-transform: uppercase; letter-spacing: 0.02em; }
table { border-collapse: collapse; margin: 0.5rem 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; vertical-align: top; }
figure { margin: 0.5rem 0; }
figure img { max-width: 100%; height: auto; display: block; }
/* NEW: no automatic numbering; keep numbers from XML text only */
ol.no-num { list-style: none; padding-left: 0; margin-left: 0; }
ol.no-num > li { margin-left: 0; }
/* Unknown tags visibly preserved (optional) */
.unknown { outline: 1px dashed #ddd; padding: 0.25rem; margin: 0.25rem 0; }
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Converted XML</title>
<style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>"""

def main():
    print(f"Converting: {INPUT_XML} → {OUTPUT_HTML}")
    html_doc = convert_xml_to_html(INPUT_XML)
    OUTPUT_HTML.write_text(html_doc, encoding="utf-8")
    print("Done.")

if __name__ == "__main__":
    main()
