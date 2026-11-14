"""Read PDF pages and normalize their text while preserving traceability."""
from pathlib import Path
import re
import unicodedata

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every PDF page with PyMuPDF, retaining physical page numbers.

    PyMuPDF is used because this Vietnamese PDF's pypdf extraction inserted
    spaces inside words (``ng ười``, ``vi ệc``), harming both lexical and dense
    retrieval. The fallback preserves a clear error path for environments where
    PyMuPDF is not installed; it is not used silently in normal runs.
    """
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required. Run: python -m pip install -r requirements.txt") from exc
    try:
        reader = fitz.open(path)
    except Exception as exc:
        raise ValueError(f"Could not read PDF {path}: {exc}") from exc
    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(reader, start=1):
        try:
            raw_text = pdf_page.get_text("text") or ""
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract text from {path}, physical page {page_number}: {exc}"
            ) from exc

        text = normalize_text(raw_text)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "text_extracted": bool(text),
                    "char_count": len(text),
                    "extractor": "pymupdf",
                },
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without flattening paragraph boundaries."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    # Keep line breaks because the structure-aware chunker uses them as paragraph
    # boundaries, while removing indentation and accidental runs of spaces.
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n[ \t]*\n+", "\n\n", normalized)
    return normalized.strip()
