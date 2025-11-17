# -*- coding: utf-8 -*-
"""
Clean PDF to XML Pipeline with Separated Workflows
- Editable: Original workflow (unchanged)
- Non-editable: OCR → Process through same logic as editable
- Mixed: Smart detection with no duplication
"""
import os
import sys
import io
import fitz
import pdfplumber
import streamlit as st
import shutil
import tempfile
import xml.etree.ElementTree as ET
import logging
import argparse
from pathlib import Path
from PIL import Image
import pytesseract

from content_extraction import detect_toc_pages, extract_toc_entries, ImageExtractor, TOCExtractor
from xml_generation import build_xml_tree, PageProcessor, prettify_xml
from config_patterns import TEMP_ROOT, PDF_INPUT_DIR, XML_OUTPUT_DIR, REQUIREMENTS_TXT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler\poppler-24.02.0\Library\bin")
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", r"C:\Users\vamsik\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ==================== PAGE ANALYSIS ====================
class PageAnalyzer:
    """Analyzes page content to determine extraction strategy"""
    @staticmethod
    def analyze_page(fitz_page, min_text_chars=100):
        """
        Analyze page content comprehensively.
        Returns dict with content metrics.
        """
        result = {
            'extractable_chars': 0,
            'has_text': False,
            'has_images': False,
            'image_coverage': 0.0,
            'text_coverage': 0.0,
            'is_likely_scanned': False,
            'confidence': 0.0
        }

        # Check extractable text
        try:
            text = fitz_page.get_text("text") or ""
            char_count = len(text.strip())
            result['extractable_chars'] = char_count
            result['has_text'] = char_count >= min_text_chars
        except:
            pass

        # Check images
        try:
            images = fitz_page.get_images(full=True) or []
            result['has_images'] = len(images) > 0
            if images:
                page_area = float(fitz_page.rect.width * fitz_page.rect.height)
                total_img_area = 0.0
                for img in images:
                    try:
                        xref = img[0]
                        rects = fitz_page.get_image_rects(xref) or []
                        for rect in rects:
                            img_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                            total_img_area += img_area
                    except:
                        continue
                result['image_coverage'] = total_img_area / max(page_area, 1.0)
        except:
            pass

        # Estimate text coverage
        try:
            blocks = fitz_page.get_text("blocks") or []
            page_area = float(fitz_page.rect.width * fitz_page.rect.height)
            text_area = 0.0
            for block in blocks:
                if len(block) >= 5 and str(block[4]).strip():
                    text_area += (block[2] - block[0]) * (block[3] - block[1])
            result['text_coverage'] = text_area / max(page_area, 1.0)
        except:
            pass

        # Determine if likely scanned
        if result['image_coverage'] > 0.7:
            result['is_likely_scanned'] = True
            result['confidence'] = 0.9
        elif result['image_coverage'] > 0.3 and result['text_coverage'] < 0.2:
            result['is_likely_scanned'] = True
            result['confidence'] = 0.7
        elif not result['has_text'] and result['has_images']:
            result['is_likely_scanned'] = True
            result['confidence'] = 0.8
        else:
            result['is_likely_scanned'] = False
            result['confidence'] = 0.9 if result['has_text'] else 0.3

        return result

    @staticmethod
    def determine_extraction_mode(page_analysis, doc_type):
        """
        Determine extraction mode based on page analysis and doc type.
        Returns: 'editable', 'ocr', or 'mixed'
        """
        if doc_type == "editable":
            return "editable"
        if doc_type == "noneditable":
            return "ocr"
        # Mixed mode decision
        if doc_type == "mixed":
            if page_analysis['is_likely_scanned']:
                return "ocr"
            elif page_analysis['has_text'] and page_analysis['confidence'] > 0.7:
                return "editable"
            else:
                # Uncertain - try editable first, may need OCR supplement
                return "mixed"
        return "editable"

# ==================== OCR TO PDFPLUMBER-LIKE FORMAT ====================
class OCRToPDFPlumberAdapter:
    """Converts OCR results to pdfplumber-compatible format for processing"""
    @staticmethod
    def render_page_to_pil(fitz_page, dpi=300):
        """Render page to PIL image"""
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    @staticmethod
    def ocr_page_to_hocr(fitz_page, lang="eng", dpi=300):
        """Run OCR and get hOCR output"""
        try:
            pil_img = OCRToPDFPlumberAdapter.render_page_to_pil(fitz_page, dpi)
            hocr = pytesseract.image_to_pdf_or_hocr(pil_img, extension='hocr', lang=lang)
            return hocr.decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

    @staticmethod
    def parse_hocr_to_words(hocr_xml, page_width, page_height):
        """
        Parse hOCR XML to extract word-level information.
        Returns list of word dicts compatible with pdfplumber format.
        IMPORTANT: Scales hOCR pixel coords to PDF space (points) using ocr_page bbox.
        """
        if not hocr_xml:
            return []
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(hocr_xml)
        except:
            return []

        # Detect hOCR page bbox (pixel space)
        hocr_w, hocr_h = None, None
        for elem in root.iter():
            cls = elem.attrib.get('class', '')
            if 'ocr_page' in cls:
                title = elem.attrib.get('title', '')
                for part in title.split(';'):
                    part = part.strip()
                    if part.startswith('bbox '):
                        try:
                            _, x0, y0, x1, y1 = part.split()
                            hocr_w = float(x1) - float(x0)
                            hocr_h = float(y1) - float(y0)
                        except:
                            pass
                break

        sx = (float(page_width) / hocr_w) if hocr_w and hocr_w > 0 else 1.0
        sy = (float(page_height) / hocr_h) if hocr_h and hocr_h > 0 else 1.0

        words = []
        # Find all word elements (ocrx_word class)
        for word_elem in root.iter():
            if 'class' in word_elem.attrib and 'ocrx_word' in word_elem.attrib['class']:
                title = word_elem.attrib.get('title', '')
                text = word_elem.text or ''
                if not text.strip():
                    continue
                bbox_match = None
                for part in title.split(';'):
                    if 'bbox' in part:
                        try:
                            coords = part.split()[1:5]
                            bbox_match = [float(c) for c in coords]
                        except:
                            pass
                        break
                if bbox_match and len(bbox_match) == 4:
                    x0, y0, x1, y1 = bbox_match
                    x0, x1 = x0 * sx, x1 * sx
                    y0, y1 = y0 * sy, y1 * sy
                    words.append({
                        'text': text,
                        'x0': x0,
                        'x1': x1,
                        'top': y0,
                        'bottom': y1,
                        'width': x1 - x0,
                        'height': y1 - y0
                    })
        return words

    @staticmethod
    def ocr_page_to_tsv(fitz_page, lang="eng", dpi=300):
        """Run OCR to TSV (fallback) and return (tsv_text, (img_w, img_h))."""
        try:
            pil_img = OCRToPDFPlumberAdapter.render_page_to_pil(fitz_page, dpi)
        except Exception:
            return None, (None, None)
        try:
            tsv = pytesseract.image_to_data(pil_img, lang=lang)
        except Exception:
            tsv = None
        return tsv, pil_img.size if hasattr(pil_img, 'size') else (None, None)

    @staticmethod
    def parse_tsv_to_words(tsv_text, img_size, target_w, target_h):
        """Parse Tesseract TSV to words and scale pixel coords to PDF space."""
        if not tsv_text or not img_size or not img_size[0] or not img_size[1]:
            return []
        try:
            lines = tsv_text.strip().splitlines()
        except Exception:
            return []
        words = []
        img_w, img_h = float(img_size[0]), float(img_size[1])
        sx = float(target_w) / img_w if img_w > 0 else 1.0
        sy = float(target_h) / img_h if img_h > 0 else 1.0
        for line in lines[1:]:
            parts = line.split('\t')
            if len(parts) < 12:
                continue
            try:
                conf = int(parts[10]) if parts[10].strip() != '' else -1
                text = parts[11].strip()
                if conf < 0 or not text:
                    continue
                left = float(parts[6]); top = float(parts[7])
                width = float(parts[8]); height = float(parts[9])
                x0 = left * sx; y0 = top * sy
                x1 = (left + width) * sx; y1 = (top + height) * sy
                words.append({
                    'text': text,
                    'x0': x0, 'x1': x1,
                    'top': y0, 'bottom': y1,
                    'width': x1 - x0, 'height': y1 - y0
                })
            except Exception:
                continue
        return words

# ==================== MAIN PIPELINE ====================
class CleanPDFToXMLPipeline:
    """
    Clean, modular pipeline with separated workflows.
    """
    def __init__(self, pdf_path, output_xml_path, image_output_dir="extracted_images",
                 doc_type="editable", ocr_lang="eng", render_dpi=300):
        self.pdf_path = pdf_path
        self.output_xml_path = output_xml_path
        self.image_output_dir = image_output_dir
        self.doc_type = (doc_type or "editable").lower()
        self.ocr_lang = ocr_lang or "eng"
        self.render_dpi = int(render_dpi) if render_dpi else 300

        self.image_extractor = None
        self.toc = None
        self.toc_extractor = None
        self.xml_root = None
        self.section_page_mapping = {}
        self.page_processor = None
        self.page_analyzer = PageAnalyzer()
        self.ocr_adapter = OCRToPDFPlumberAdapter()

    def run(self):
        """Main pipeline execution"""
        print(f"=== PDF TO XML PIPELINE (Mode: {self.doc_type.upper()}) ===")

        # Step 1: TOC extraction
        self.extract_toc_structure()

        # Step 2: Image extraction (explicit Poppler path)
        self.image_extractor = ImageExtractor(
            self.pdf_path,
            self.image_output_dir,
            poppler_path=POPPLER_PATH
        )
        self.image_extractor.extract_images()

        # Step 3: Initialize page processor (BACKWARD-COMPATIBLE)
        cv_map = getattr(self.image_extractor, "get_cv_table_crops_map", lambda: {})()
        base_kwargs = {
            "image_map": self.image_extractor.get_image_map(),
            "image_output_dir": self.image_output_dir,
            "image_xref_map": self.image_extractor.get_image_xref_map(),
        }
        try:
            self.page_processor = PageProcessor(**base_kwargs, cv_table_crops_map=cv_map)
        except TypeError:
            self.page_processor = PageProcessor(**base_kwargs)
        try:
            setattr(self.page_processor, "cv_table_crops_map", cv_map)
        except Exception:
            pass

        # NEW: provide OCR-table config to the page processor
        try:
            self.page_processor._pdf_path = self.pdf_path
            self.page_processor._poppler_path = POPPLER_PATH
            self.page_processor._render_dpi = self.render_dpi
            self.page_processor._ocr_lang = self.ocr_lang
            self.page_processor._ocr_tables_enabled = True
        except Exception:
            pass

        # Step 4: Build XML structure with sections
        self.extract_pdf_structure_with_sections()

        # Step 5: Process pages based on mode
        self.process_all_pages_content()

        # Step 6: Save XML
        self.save_xml()
        print(f"✓ Pipeline completed: {self.output_xml_path}")

    def extract_toc_structure(self):
        doc = fitz.open(self.pdf_path)
        toc_pages = detect_toc_pages(doc)
        if toc_pages:
            self.toc = extract_toc_entries(doc, toc_pages)
            print(f"✓ TOC: {len(self.toc)} entries from pages {toc_pages}")
        else:
            self.toc = []
            print("! No TOC detected")
        doc.close()

    def extract_pdf_structure_with_sections(self):
        print("=== BUILDING XML STRUCTURE ===")
        doc = fitz.open(self.pdf_path)
        if self.toc:
            self.xml_root, self.section_page_mapping = build_xml_tree(doc, self.toc)
            print(f"✓ Sections: {len(self.section_page_mapping)} page mappings")
        else:
            self.xml_root, self.section_page_mapping = build_xml_tree(doc, [])
            print("! Simple page structure (no sections)")
        doc.close()

    def process_all_pages_content(self):
        print(f"=== PROCESSING PAGES ({self.doc_type.upper()} MODE) ===")
        fitz_doc = fitz.open(self.pdf_path)
        pb_doc = None
        if self.doc_type in ("editable", "mixed"):
            try:
                pb_doc = pdfplumber.open(self.pdf_path)
            except Exception as e:
                logger.warning(f"Could not open with pdfplumber: {e}")

        try:
            for page_num in range(1, fitz_doc.page_count + 1):
                print(f"\nPage {page_num}/{fitz_doc.page_count}")
                section_info = self.section_page_mapping.get(page_num)
                if section_info:
                    page_elem = section_info.get("page_element")
                else:
                    page_elem = self.xml_root.find(f".//page[@number='{page_num}']")
                if page_elem is None:
                    page_elem = ET.SubElement(self.xml_root, "page", number=str(page_num))
                if page_elem is None:
                    logger.warning(f"No page element for page {page_num}")
                    continue

                fitz_page = fitz_doc[page_num - 1]

                if self.doc_type == "editable":
                    self._process_editable_page(page_num, fitz_page, pb_doc, page_elem)
                elif self.doc_type == "noneditable":
                    self._process_noneditable_page(page_num, fitz_page, page_elem)
                elif self.doc_type == "mixed":
                    self._process_mixed_page(page_num, fitz_page, pb_doc, page_elem)
        finally:
            fitz_doc.close()
            if pb_doc:
                pb_doc.close()

    def _process_editable_page(self, page_num, fitz_page, pb_doc, page_elem):
        print(f" → Editable workflow")
        if pb_doc is None or page_num - 1 >= len(pb_doc.pages):
            logger.warning(f"No pdfplumber page for {page_num}")
            return
        pb_page = pb_doc.pages[page_num - 1]
        try:
            self.page_processor.process_single_page(
                page=pb_page,
                page_elem=page_elem,
                page_number=page_num,
                fitz_page=fitz_page,
                section_page_mapping=self.section_page_mapping,
                toc_extractorr=self.toc_extractor
            )
            print(f" ✓ Extracted with original logic")
        except Exception as e:
            logger.error(f"Error processing editable page {page_num}: {e}")
            import traceback
            traceback.print_exc()

    def _process_noneditable_page(self, page_num, fitz_page, page_elem):
        print(f" → OCR workflow")
        try:
            hocr = self.ocr_adapter.ocr_page_to_hocr(
                fitz_page,
                lang=self.ocr_lang,
                dpi=self.render_dpi
            )
            if not hocr:
                logger.warning(f"OCR failed for page {page_num}")
                return
            page_w = float(fitz_page.rect.width)
            page_h = float(fitz_page.rect.height)
            words = self.ocr_adapter.parse_hocr_to_words(hocr, page_w, page_h)
            if not words:
                tsv, img_size = self.ocr_adapter.ocr_page_to_tsv(
                    fitz_page, lang=self.ocr_lang, dpi=self.render_dpi
                )
                words = self.ocr_adapter.parse_tsv_to_words(tsv, img_size, page_w, page_h)
            if not words:
                logger.warning(f"No words extracted from OCR for page {page_num}")
                return
            print(f" ✓ OCR extracted {len(words)} words")

            mock_page = self._create_mock_page_from_words(words, page_w, page_h)
            # NEW: pass is_scanned=True to enable OCR table extraction and suppress table images
            self.page_processor.process_single_page(
                page=mock_page,
                page_elem=page_elem,
                page_number=page_num,
                fitz_page=fitz_page,
                section_page_mapping=self.section_page_mapping,
                toc_extractorr=self.toc_extractor,
                is_scanned=True
            )
            print(f" ✓ Processed with same logic as editable")
        except Exception as e:
            logger.error(f"Error processing non-editable page {page_num}: {e}")
            import traceback
            traceback.print_exc()

    def _create_mock_page_from_words(self, words, page_width, page_height):
        class MockPage:
            def __init__(self, words, width, height):
                self.words = words
                self.chars = []
                self.width = width
                self.height = height
                self.bbox = (0, 0, width, height)
            def extract_words(self):
                return self.words
            def extract_tables(self):
                return []
            def find_tables(self):
                return []
        return MockPage(words, page_width, page_height)

    def _process_mixed_page(self, page_num, fitz_page, pb_doc, page_elem):
        print(f" → Mixed workflow (analyzing...)")
        analysis = self.page_analyzer.analyze_page(fitz_page)
        mode = self.page_analyzer.determine_extraction_mode(analysis, "mixed")
        print(f" → Detected: {mode} (confidence: {analysis['confidence']:.2f})")
        print(f"  Text: {analysis['extractable_chars']} chars, Images: {analysis['image_coverage']:.1%} coverage")

        if mode == "editable":
            self._process_editable_page(page_num, fitz_page, pb_doc, page_elem)
        elif mode == "ocr":
            self._process_noneditable_page(page_num, fitz_page, page_elem)
        else:
            if pb_doc and page_num - 1 < len(pb_doc.pages):
                try:
                    pb_page = pb_doc.pages[page_num - 1]
                    self.page_processor.process_single_page(
                        page=pb_page,
                        page_elem=page_elem,
                        page_number=page_num,
                        fitz_page=fitz_page,
                        section_page_mapping=self.section_page_mapping,
                        toc_extractorr=self.toc_extractor
                    )
                    has_content = any(
                        child.tag in ('Para', 'OrderedList', 'ItemizedList', 'InformalTable', 'LeftColumn', 'RightColumn','InformalFigure')
                        for child in page_elem
                    )
                    if has_content:
                        print(f" ✓ Editable extraction successful")
                    else:
                        print(f"  ! Editable extraction incomplete, trying OCR")
                        for child in list(page_elem):
                            page_elem.remove(child)
                        self._process_noneditable_page(page_num, fitz_page, page_elem)
                except Exception as e:
                    logger.warning(f"Editable extraction failed: {e}, trying OCR")
                    self._process_noneditable_page(page_num, fitz_page, page_elem)
            else:
                self._process_noneditable_page(page_num, fitz_page, page_elem)

    def save_xml(self):
        if self.xml_root is not None:
            with open(self.output_xml_path, "w", encoding="utf-8") as f:
                f.write(prettify_xml(self.xml_root))
            print(f"✓ XML saved: {self.output_xml_path}")
        else:
            logger.error("No XML root to save")

    def get_toc_structure(self):
        return self.toc

    def get_section_mapping(self):
        return self.section_page_mapping

    def get_image_map(self):
        return self.image_extractor.get_image_map() if self.image_extractor else {}

# ==================== STREAMLIT / CLI ====================
def save_uploaded_file(uploaded_file):
    PDF_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = PDF_INPUT_DIR / uploaded_file.name
    with open(fp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(fp)

def build_framemaker_xml(pdf_path, assets_dir, doc_type="editable", ocr_lang="eng", render_dpi=300):
    pdf_name = Path(pdf_path).stem
    XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = XML_OUTPUT_DIR / f"{pdf_name}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_path = XML_OUTPUT_DIR / f"{pdf_name}.xml"

    pipeline = CleanPDFToXMLPipeline(
        pdf_path=pdf_path,
        output_xml_path=str(xml_path),
        image_output_dir=str(image_dir),
        doc_type=doc_type,
        ocr_lang=ocr_lang,
        render_dpi=render_dpi
    )
    pipeline.run()

    with open(xml_path, "rb") as f:
        return f.read()

def cleanup_temp_dir():
    for d in [PDF_INPUT_DIR, XML_OUTPUT_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def main_streamlit():
    st.set_page_config(page_title="PDF → FrameMaker XML", layout="wide")
    st.title("PDF → FrameMaker XML Converter")
    st.markdown(""" 
**Three Separate Workflows:**
- **Editable**: Uses original extraction logic (best for text-based PDFs)
- **Non-editable (Scanned)**: OCR → Processes through same logic (maintains XML structure)
- **Mixed**: Smart detection per page (auto-selects best method)
""")
    with st.expander("Requirements"):
        st.code(REQUIREMENTS_TXT.strip(), language="text")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if not uploaded_file:
        st.info("Upload a PDF to begin")
        return

    st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
    doc_type = st.radio(
        "Document Type",
        ["Editable", "Non-editable (Scanned)", "Mixed (Auto-detect)"],
        index=0,
        horizontal=True
    )
    ocr_lang = st.text_input("OCR Language (if needed)", value="eng")
    render_dpi = st.slider("OCR DPI (if needed)", 200, 400, 300, 25)

    if st.button("Convert to XML"):
        with st.spinner("Processing..."):
            try:
                pdf_path = save_uploaded_file(uploaded_file)
                doc_type_map = {
                    "Editable": "editable",
                    "Non-editable (Scanned)": "noneditable",
                    "Mixed (Auto-detect)": "mixed"
                }
                doc_type_val = doc_type_map[doc_type]
                xml_bytes = build_framemaker_xml(
                    pdf_path,
                    TEMP_ROOT / f"{Path(pdf_path).stem}_assets",
                    doc_type=doc_type_val,
                    ocr_lang=ocr_lang,
                    render_dpi=render_dpi
                )
                xml_filename = f"{Path(pdf_path).stem}_converted.xml"
                st.success("✓ Conversion complete!")
                st.download_button("Download XML", xml_bytes, file_name=xml_filename, mime="text/xml")

                try:
                    xml_text = xml_bytes.decode('utf-8', errors='replace')
                    with st.expander('XML Preview (first 50KB)'):
                        st.code(xml_text[:50000], language='xml')
                except Exception as e:
                    st.warning(f'Preview failed: {e}')

                st.subheader("Extracted Images")
                image_dir = XML_OUTPUT_DIR / f"{Path(pdf_path).stem}_images"
                if image_dir.exists():
                    imgs = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                    if imgs:
                        for img in imgs[:20]:
                            try:
                                st.image(str(img), caption=img.name, width=300)
                            except:
                                pass
                    else:
                        st.info("No images found")
                else:
                    st.info("No images found")
            except Exception as e:
                logger.exception("Failed")
                st.error(f"Error: {e}")

    if st.button("Clear temp files"):
        cleanup_temp_dir()
        st.success("Cleared")

def main_cli():
    parser = argparse.ArgumentParser(description="PDF → XML Converter")
    parser.add_argument("pdf_path", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output XML path")
    parser.add_argument("-i", "--images", default="extracted_images", help="Image directory")
    parser.add_argument("--doc-type", choices=["editable", "noneditable", "mixed"],
                        default="editable", help="Document type")
    parser.add_argument("--ocr-lang", default="eng", help="OCR language")
    parser.add_argument("--render-dpi", type=int, default=300, help="OCR DPI")
    parser.add_argument("--verbose", action="true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    output_path = args.output or f"{pdf_path.stem}_converted.xml"
    try:
        pipeline = CleanPDFToXMLPipeline(
            pdf_path=str(pdf_path),
            output_xml_path=output_path,
            image_output_dir=args.images,
            doc_type=args.doc_type,
            ocr_lang=args.ocr_lang,
            render_dpi=args.render_dpi
        )
        print(f"PDF: {pdf_path}")
        print(f"Mode: {args.doc_type}")
        print(f"Output: {output_path}")
        print("-" * 50)
        pipeline.run()
        print("-" * 50)
        print("✓ Success!")
        toc = pipeline.get_toc_structure()
        if toc:
            print(f"✓ TOC: {len(toc)} sections")
        images = sum(len(v) for v in pipeline.get_image_map().values())
        if images:
            print(f"✓ Images: {images}")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "run":
        main_cli()
    else:
        main_streamlit()