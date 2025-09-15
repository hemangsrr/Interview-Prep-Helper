from io import BytesIO
from typing import Union
from PyPDF2 import PdfReader


def extract_text_from_pdf(file: Union[BytesIO, any]) -> str:
    """Extract text from an uploaded PDF file (werkzeug FileStorage or bytes)."""
    if hasattr(file, 'read'):
        reader = PdfReader(file)
    else:
        reader = PdfReader(BytesIO(file))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)
