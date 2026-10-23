from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf_raw(pdf):
    """
    Fallback: read page text with pypdf directly.

    Some PDFs have broken metadata (e.g. corrupt page labels) that makes
    PyPDFLoader crash before it ever reads the text, even though the text is fine.
    """
    reader = PdfReader(str(pdf))

    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": str(pdf), "page": i, "total_pages": len(reader.pages)},
        )
        for i, page in enumerate(reader.pages)
    ]


def load_pdfs(data_path: str = "data"):

    pdf_files = sorted(Path(data_path).glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs in {Path(data_path).resolve()}")

    documents = []

    for pdf in pdf_files:

        try:
            docs = PyPDFLoader(str(pdf)).load()

        except Exception as e:
            print(f"PyPDFLoader failed on {pdf.name} ({type(e).__name__}), using fallback")
            docs = load_pdf_raw(pdf)

        # Use the same relative path style for every PDF so source matching is consistent
        for doc in docs:
            doc.metadata["source"] = str(Path("data") / pdf.name)

        print(f"Loaded {pdf.name}: {len(docs)} pages")
        documents.extend(docs)

    return documents
