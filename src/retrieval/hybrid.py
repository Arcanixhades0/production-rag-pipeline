from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from src.config import FAISS_DIR, FETCH_K, TOP_K
from src.retrieval.embeddings import get_embeddings


def reciprocal_rank_fusion(result_lists, k=60):
    """
    Merge several ranked lists into one.

    Each doc gets 1 / (k + rank) from every list it appears in.
    Docs ranked high by BOTH retrievers float to the top.
    k=60 is the value from the original RRF paper.
    """
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)

    return [docs[key] for key in ranked]


def load_faiss(path=FAISS_DIR, embeddings=None):
    return FAISS.load_local(
        str(path),
        embeddings or get_embeddings(),
        allow_dangerous_deserialization=True
    )


class HybridRetriever:

    def __init__(self, vectorstore=None, fetch_k=FETCH_K):
        self.vectorstore = vectorstore or load_faiss()
        self.fetch_k = fetch_k

        # Build BM25 from the exact same chunks stored in FAISS,
        # so both retrievers search the same corpus
        chunks = list(self.vectorstore.docstore._dict.values())

        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = fetch_k

    def dense(self, query, k=None):
        return self.vectorstore.similarity_search(query, k=k or self.fetch_k)

    def sparse(self, query, k=None):
        self.bm25.k = k or self.fetch_k
        return self.bm25.invoke(query)

    def search(self, query, k=TOP_K):
        fused = reciprocal_rank_fusion([self.dense(query), self.sparse(query)])
        return fused[:k]


if __name__ == "__main__":

    retriever = HybridRetriever()

    while True:
        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        for i, doc in enumerate(retriever.search(query), start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "-")
            print(f"\n{i}. {source} | page {page}")
            print(doc.page_content[:300])
