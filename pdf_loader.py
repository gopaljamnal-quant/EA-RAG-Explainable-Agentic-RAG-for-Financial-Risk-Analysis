"""
Simple PDF loader for EA-RAG financial documents.
"""

from pathlib import Path
from typing import List
from ea_rag.data_models import Document


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

        # Extract text
        text = ""
        try:
            with pdfplumber.open(str(pdf_file)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            print(f"(error: {e})")
            continue

        if not text:
            print("(empty)")
            continue

        # Auto-detect document type
        name_lower = pdf_file.name.lower()
        source_type = "document"
        issuer = None

        if "10-k" in name_lower:
            source_type = "10-K"
        elif "8-k" in name_lower:
            source_type = "8-K"
        elif "earnings" in name_lower:
            source_type = "earnings"
        elif "news" in name_lower:
            source_type = "news"

        if "tesla" in name_lower:
            issuer = "Tesla, Inc."
        elif "panasonic" in name_lower:
            issuer = "Panasonic Corporation"

        # Create document
        doc = Document(
            id=pdf_file.name.replace(".pdf", ""),
            text=text[:50000],  # Truncate to 50K chars
            source_type=source_type,
            issuer=issuer,
            date=None,
        )

        documents.append(doc)
        print(f"✓ ({len(text)} chars)")

    print(f"\nLoaded {len(documents)} documents\n")
    return documents