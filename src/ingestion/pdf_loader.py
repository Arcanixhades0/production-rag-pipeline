from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(data_path: str = "data"):

    print(f"\nSearching in: {Path(data_path).resolve()}")
    print(f"Using data_path: {data_path}")

    pdf_files = list(
        Path(data_path).glob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDFs")

    documents = []

    for pdf in pdf_files:

        print(f"Loading: {pdf.name}")

        try:

            loader = PyPDFLoader(
                str(pdf)
            )

            documents.extend(
                loader.load()
            )

        except Exception as e:

            print(
                f"Skipping {pdf.name}: {e}"
            )

    return documents