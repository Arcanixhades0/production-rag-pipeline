import time

from langsmith import traceable

from src.config import FETCH_K, TOP_K
from src.llm import get_llm
from src.ingestion.chunker import expand_windows
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


PROMPT = """Answer the question using only the provided context.
Cite the sources you used by their number, like [1] or [2].

If the answer is not in the context, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:"""


class RAGPipeline:
    """
    The layers, in order:
      1. ingestion  (src/ingestion, run offline by build_index)
      2. chunking   (src/ingestion/chunker.py, offline)
      3. indexing   (FAISS + BM25)
      4. retrieval  (dense + sparse, fused with RRF)
      5. reranking  (cross-encoder)
      6. generation (LLM with a grounded prompt)
    """

    def __init__(self, mode="hybrid_rerank", vectorstore=None, reranker=None):
        # mode lets the evaluation compare: "dense", "hybrid", "hybrid_rerank"
        self.mode = mode
        self.retriever = HybridRetriever(vectorstore)
        self.reranker = None
        if mode == "hybrid_rerank":
            self.reranker = reranker or Reranker()
        self._llm = None

    @property
    def llm(self):
        # Loaded on first use so retrieval-only evaluation never needs an LLM
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @traceable(name="retrieve", run_type="retriever")
    def retrieve(self, query, k=TOP_K, source_type=None):
        if self.mode == "dense":
            docs = self.retriever.dense(query, k=FETCH_K)
        else:
            docs = reciprocal_rank_fusion([
                self.retriever.dense(query),
                self.retriever.sparse(query),
            ])

        # Metadata-aware filtering: e.g. source_type="pdf" searches only books
        if source_type:
            docs = [d for d in docs if d.metadata.get("source_type") == source_type]

        # Sentence-window chunks: swap each matched sentence for its surrounding window
        docs = expand_windows(docs)

        if self.reranker:
            return self.reranker.rerank(query, docs[:FETCH_K * 2], top_k=k)

        return docs[:k]

    @traceable(name="rag_answer")
    def answer(self, question, k=TOP_K, source_type=None):
        timings = {}

        start = time.perf_counter()
        docs = self.retrieve(question, k=k, source_type=source_type)
        timings["retrieval_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        response = self.llm.invoke(build_prompt(question, docs))
        timings["generation_ms"] = (time.perf_counter() - start) * 1000

        return {
            "answer": response.content,
            "contexts": [doc.page_content for doc in docs],
            "sources": sources(docs),
            "timings": timings,
        }

    def stream(self, question, k=TOP_K, source_type=None):
        """Yields the sources first, then the answer token by token."""
        docs = self.retrieve(question, k=k, source_type=source_type)

        yield {"type": "sources", "data": sources(docs)}

        for chunk in self.llm.stream(build_prompt(question, docs)):
            if chunk.content:
                yield {"type": "token", "data": chunk.content}


def build_prompt(question, docs):
    context = "\n\n".join(
        f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1)
    )
    return PROMPT.format(context=context, question=question)


def sources(docs):
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page"),
            "source_type": doc.metadata.get("source_type"),
        }
        for doc in docs
    ]
