from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL


class Reranker:
    """
    A cross-encoder reads (question, chunk) TOGETHER and outputs one relevance score.

    Bi-encoders (our embedding model) embed question and chunk separately,
    which is fast enough to search thousands of chunks but less accurate.
    Cross-encoders are accurate but slow, so we only run them on ~20-40 candidates.
    """

    def __init__(self, model_name=RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        scores = self.model.predict([(query, doc.page_content) for doc in docs])

        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)

        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)

        return [doc for doc, _ in ranked[:top_k]]
