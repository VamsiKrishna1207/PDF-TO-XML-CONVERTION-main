#!/usr/bin/env python3
"""
Streamlit: FrameMaker-style XML → PDF Converter
-----------------------------------------------
Upload an XML file (produced by your PDF→XML converter),
and generate a readable PDF with Titles, Subtitles, Paragraphs,
Images, and Tables.

Keeps this as a separate tool, not merged with PDF→XML script.
"""

import os
import xml.etree.ElementTree as ET
import streamlit as st
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)

# -----------------------
# Core XML → PDF function
# -----------------------
def xml_to_pdf(xml_path: str, pdf_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                 fontSize=18, spaceAfter=12)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Heading2"],
                                    fontSize=14, spaceAfter=8)
    para_style = ParagraphStyle("Para", parent=styles["Normal"],
                                fontSize=11, leading=14, spaceAfter=6)

    story = []

    def handle_node(node):
        tag = node.tag

        if tag == "Title":
            story.append(Paragraph(node.text or "", title_style))
        elif tag == "Subtitle":
            story.append(Paragraph(node.text or "", subtitle_style))
        elif tag == "Para":
            story.append(Paragraph(node.text or "", para_style))
        elif tag == "InformalFigure":
            graphic = node.find("Graphic")
            if graphic is not None:
                path = graphic.attrib.get("entityref", "")
                if os.path.exists(path):
                    img = Image(path, hAlign="CENTER")

                    # Desired max width (you can adjust this)
                    max_width = 300
                    aspect = img.imageWidth / float(img.imageHeight)

                    if img.imageWidth > max_width:
                        img.drawWidth = max_width
                        img.drawHeight = max_width / aspect
                    else:
                        img.drawWidth = img.imageWidth
                        img.drawHeight = img.imageHeight

                    story.append(img)
                    story.append(Spacer(1, 12))

        elif tag in ("InformalTable", "Table"):
            tgroup = node.find("TGroup")
            if tgroup is not None:
                table_data = []
                # Header
                thead = tgroup.find("THead")
                if thead is not None:
                    for row in thead.findall("TRow"):
                        table_data.append([cell.text or "" for cell in row.findall("TCell")])
                # Body
                tbody = tgroup.find("TBody")
                if tbody is not None:
                    for row in tbody.findall("TRow"):
                        table_data.append([cell.text or "" for cell in row.findall("TCell")])
                if table_data:
                    t = Table(table_data, repeatRows=1)
                    t.setStyle(TableStyle([
                        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                        ("ALIGN", (0,0), (-1,-1), "CENTER"),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 12))

        # Recurse into children
        for child in node:
            handle_node(child)

    handle_node(root)

    doc.build(story)
    return pdf_path

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="XML → PDF", layout="wide")
    st.title("FrameMaker-style XML → PDF Converter")

    st.markdown(
        "Upload a FrameMaker-style XML file and this tool will generate "
        "a PDF version with styled text, tables, and images."
    )

    uploaded_file = st.file_uploader("Choose an XML file", type=["xml"])
    if uploaded_file is None:
        st.info("Upload an XML file to begin.")
        return

    if st.button("Convert to PDF"):
        with st.spinner("Processing..."):
            # Save uploaded XML temporarily
            temp_dir = Path("temp_xml_dir")
            temp_dir.mkdir(exist_ok=True)
            xml_path = temp_dir / uploaded_file.name
            with open(xml_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Output PDF path
            pdf_path = temp_dir / (xml_path.stem + ".pdf")

            try:
                xml_to_pdf(str(xml_path), str(pdf_path))
                st.success("Conversion complete.")

                # Download button
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf"
                    )

                st.subheader("Preview (first page)")
                st.pdf(str(pdf_path))

            except Exception as e:
                st.error(f"Conversion failed: {e}")

    if st.button("Clear temporary files"):
        import shutil
        try:
            shutil.rmtree("temp_xml_dir")
            st.success("Temporary files cleared.")
        except Exception:
            st.warning("No temporary files found.")

if __name__ == "__main__":
    main()
