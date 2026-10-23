from src.config import DATA_DIR, URLS_FILE, SQL_DB
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.web_loader import load_websites
from src.ingestion.sql_loader import load_sql


def tag(documents, source_type):
    # source_type lets us filter results later, e.g. "only search PDFs"
    for doc in documents:
        doc.metadata["source_type"] = source_type
    return documents


def load_documents():

    documents = []

    documents.extend(tag(load_pdfs(str(DATA_DIR)), "pdf"))
    documents.extend(tag(load_websites(str(URLS_FILE)), "web"))

    if SQL_DB.exists():
        documents.extend(load_sql(SQL_DB))

    print(f"\nTotal Documents: {len(documents)}")

    return documents
