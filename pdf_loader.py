"""
Simple PDF loader for EA-RAG financial documents.
"""

from pathlib import Path
from typing import List

from ea_rag.data_models import Document

MAX_DOCUMENT_TEXT_CHARS = 50000


def load_pdfs(directory: str, max_docs: int = None) -> List[Document]:
    """
    Load PDF files from a directory.

    Args:
        directory: Path to folder containing PDFs
        max_docs: Limit number of documents (optional)

    Returns:
        List of Document objects
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install: pip install --user pdfplumber")

    documents = []
    pdf_path = Path(directory)

    if not pdf_path.exists():
        print(f"Error: Directory '{directory}' not found")
        return []

    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs\n")

    for i, pdf_file in enumerate(pdf_files):
        if max_docs and i >= max_docs:
            break

        print(f"Loading {pdf_file.name}...", end=" ")

        text = _extract_pdf_text(pdfplumber, pdf_file)
        if text is None:
            continue

        if not text:
            print("(empty)")
            continue

        doc = Document(
            id=pdf_file.stem,
            text=text[:MAX_DOCUMENT_TEXT_CHARS],  # Truncate to 50K chars
            source_type=_infer_source_type(pdf_file.name),
            issuer=_infer_issuer(pdf_file.name),
            date=None,
        )

        documents.append(doc)
        print(f"✓ ({len(text)} chars)")

    print(f"\nLoaded {len(documents)} documents\n")
    return documents


def _extract_pdf_text(pdfplumber, pdf_file: Path):
    text_parts = []
    try:
        with pdfplumber.open(str(pdf_file)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
    except Exception as exc:
        print(f"(error: {exc})")
        return None
    return "\n".join(text_parts) + ("\n" if text_parts else "")


def _infer_source_type(filename: str) -> str:
    name_lower = filename.lower()
    if "10-k" in name_lower:
        return "10-K"
    if "8-k" in name_lower:
        return "8-K"
    if "earnings" in name_lower:
        return "earnings"
    if "news" in name_lower:
        return "news"
    return "document"


def _infer_issuer(filename: str):
    name_lower = filename.lower()
    if "tesla" in name_lower:
        return "Tesla, Inc."
    if "panasonic" in name_lower:
        return "Panasonic Corporation"
    return None