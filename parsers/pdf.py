import fitz  # PyMuPDF
import io

def pdf_to_text(source) -> str:
    """
    Extract text from a PDF.
    Works with both file paths (str) and in-memory file-like objects (Streamlit uploads).
    """

    try:
        # Detect if source is a path (string) or a file-like object
        if isinstance(source, (str, bytes)):
            doc = fitz.open(source)
        else:
            # handle file-like object from Streamlit uploader
            file_bytes = source.read()
            doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")

        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text.strip()

    except Exception as e:
        print(f"[WARN] PDF extraction failed: {e}")
        return ""
