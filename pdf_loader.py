# pdf_loader.py
import fitz  # PyMuPDF

def load_pdf(path: str):
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        pages.append({
            "text": page.get_text(),
            "page": i + 1
        })

    return pages
