"""
render_template.py
──────────────────
Step 1 of the box-detection pipeline.

Renders the PDF at 300 DPI using PyMuPDF (fitz), saves the full-page PNG,
and optionally lets you crop a sub-region to isolate the digit-box area.

Usage:
    python render_template.py
    python render_template.py --pdf ../bannari_amman_form_template.pdf --dpi 300
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def render_pdf_page(pdf_path: str, page_idx: int = 0, dpi: int = 300) -> "fitz.Pixmap":
    """
    Open *pdf_path* and render *page_idx* at *dpi* dots-per-inch.

    Returns a fitz.Pixmap in RGB colour space.
    """
    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        raise IndexError(f"Page {page_idx} not found in '{pdf_path}' ({len(doc)} pages).")
    page = doc[page_idx]
    zoom = dpi / 72.0          # 72 pt/inch is PDF default
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    doc.close()
    return pix


def save_pixmap(pix: "fitz.Pixmap", out_path: str) -> None:
    """Save a PyMuPDF Pixmap as PNG."""
    pix.save(out_path)
    print(f"[render_template] Saved -> {out_path}  ({pix.width}x{pix.height} px)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render form PDF to PNG at 300 DPI.")
    parser.add_argument(
        "--pdf", default="../bannari_amman_form_template.pdf",
        help="Path to the PDF file."
    )
    parser.add_argument("--dpi",  type=int, default=300, help="Render DPI (default 300).")
    parser.add_argument("--page", type=int, default=0,   help="Page index (0-based).")
    parser.add_argument(
        "--out", default="template_300dpi.png",
        help="Output PNG filename."
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"[ERROR] PDF not found: {pdf_path.resolve()}")

    pix = render_pdf_page(str(pdf_path), page_idx=args.page, dpi=args.dpi)
    save_pixmap(pix, args.out)


if __name__ == "__main__":
    main()
