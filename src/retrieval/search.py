import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_community.vectorstores import FAISS

from embeddings import get_embeddings


embeddings = get_embeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = input("Ask a question: ")

results = vectorstore.similarity_search(
    query,
    k=5
)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}\n")

    print(doc.page_content[:500])

    print("\nSource:")

    print(doc.metadata)