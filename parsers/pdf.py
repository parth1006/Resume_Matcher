import fitz  # PyMuPDF

def pdf_to_text(path: str) -> str:
    """Extract text from PDF using PyMuPDF; safely handle edge cases."""
    try:
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        return text.strip()
    except Exception as e:
        print(f"[WARN] PDF extraction failed for {path}: {e}")
        return ""
