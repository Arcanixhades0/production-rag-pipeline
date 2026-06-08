from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


pdf_files = list(Path("data").glob("*.pdf"))

print(f"Found {len(pdf_files)} PDFs:\n")

for pdf in pdf_files:
    print(pdf.name)


documents = []

for pdf in pdf_files:
    print(f"\nLoading: {pdf.name}")

    try:
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()

        documents.extend(docs)

        print(f"Loaded {len(docs)} pages")

    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nLoaded {len(documents)} pages")

print("\nFirst Document Metadata:")
print(documents[0].metadata)

print("\nFirst Document Page Content:")
print(documents[0].page_content[:500]
      )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"\nTotal Chunks Created: {len(chunks)}")

print("\nFirst Chunk Metadata:")
print(chunks[0].metadata)

print("\nFirst Chunk Length:")
print(len(chunks[0].page_content))


for i in range(20):
    if len(chunks[i].page_content.strip()) > 100:
        print(f"\nChunk {i}:\n")
        print(chunks[i].page_content[:1000])
        break

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(documents = chunks, embedding = embeddings)
vectorstore.save_local("faiss_db")
print("\nVector database created and saved locally as 'faiss_db'")
