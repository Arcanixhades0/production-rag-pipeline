from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_faiss_index(data_folder="data"):

    pdf_files = list(Path(data_folder).glob("*.pdf"))

    documents = []

    for pdf in pdf_files:

        try:
            loader = PyPDFLoader(str(pdf))
            documents.extend(loader.load())

        except Exception as e:
            print(f"Skipping {pdf.name}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local("faiss_db")

    return len(pdf_files), len(documents), len(chunks)