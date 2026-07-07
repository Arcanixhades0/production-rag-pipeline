import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama

from embeddings import get_embeddings
from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.chunker import create_chunks


print("Loading PDFs...")
documents = load_pdfs("data")
print(f"Loaded {len(documents)} pages")

chunks = create_chunks(documents)
print(f"Created {len(chunks)} chunks")


# ---------------- FAISS ----------------
embeddings = get_embeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- BM25 ----------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# ---------------- LLM ----------------
llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0
)


def merge_results(faiss_docs, bm25_docs):
    seen = set()
    merged = []

    for doc in faiss_docs + bm25_docs:
        content = doc.page_content.strip()

        if content not in seen:
            seen.add(content)
            merged.append(doc)

    return merged


while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    # FAISS search
    faiss_docs = vectorstore.max_marginal_relevance_search(
    query=query,
    k=3,
    fetch_k=10
    )

    # BM25 search
    bm25_docs = bm25_retriever.invoke(query)

    # Merge both results
    docs = merge_results(faiss_docs, bm25_docs)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources used:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown')} | page {doc.metadata.get('page', 'Unknown')}")