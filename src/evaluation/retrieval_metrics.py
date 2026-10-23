import json
import time

import numpy as np

from src.config import EVAL_DIR


def load_testset(path=EVAL_DIR / "testset.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(doc, row):
    """A retrieved chunk is relevant if it comes from the page the question was written from."""
    if doc.metadata.get("source") != row["source"]:
        return False
    return row.get("page") is None or doc.metadata.get("page") == row["page"]


def evaluate_retrieval(retrieve, testset, k=5):
    """
    retrieve: function(question) -> list of Documents (best first)

    hit_rate  : % of questions where the right page is anywhere in the top k
    mrr       : mean of 1/rank of the first right page (rewards ranking it FIRST)
    precision : % of the k returned chunks that are from the right page
                (low precision = noisy context sent to the LLM)
    """
    hits, reciprocal_ranks, precisions, context_chars, latencies = [], [], [], [], []

    for row in testset:
        start = time.perf_counter()
        docs = retrieve(row["question"])[:k]
        latencies.append((time.perf_counter() - start) * 1000)

        relevant = [is_relevant(d, row) for d in docs]

        hits.append(any(relevant))
        first = next((i for i, r in enumerate(relevant, start=1) if r), None)
        reciprocal_ranks.append(1 / first if first else 0)
        precisions.append(sum(relevant) / k)
        context_chars.append(sum(len(d.page_content) for d in docs))

    return {
        "questions": len(testset),
        "hit_rate": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "precision": float(np.mean(precisions)),
        "avg_context_chars": float(np.mean(context_chars)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }
