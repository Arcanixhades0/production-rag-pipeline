from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.web_loader import load_websites


def load_documents():

    documents = []

    pdf_docs = load_pdfs("data")
    web_docs = load_websites("data/web/urls.txt")

    documents.extend(pdf_docs)
    documents.extend(web_docs)

    print(f"\nTotal Documents: {len(documents)}")

    return documents