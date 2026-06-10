"""Utilidades de PDF: extraer texto por página y renderizar imágenes (PyMuPDF)."""
import base64
import io

import fitz  # PyMuPDF
from PIL import Image


def extract_pages_text(pdf_path) -> list[dict]:
    """[{"page": 1-based, "text": "..."}] por cada página."""
    doc = fitz.open(pdf_path)
    out = [{"page": i, "text": page.get_text()} for i, page in enumerate(doc, 1)]
    doc.close()
    return out


def render_page_png(pdf_path, page_num: int, dpi: int) -> bytes | None:
    doc = fitz.open(pdf_path)
    idx = page_num - 1
    if idx < 0 or idx >= len(doc):
        doc.close()
        return None
    pix = doc[idx].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_page_data_uri(pdf_path, page_num: int, dpi: int) -> str | None:
    png = render_page_png(pdf_path, page_num, dpi)
    if png is None:
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
