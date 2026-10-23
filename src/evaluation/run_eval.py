"""
End-to-end evaluation: baseline pipeline vs final pipeline.

  baseline = dense search only (what the project started with)
  final    = hybrid (FAISS + BM25, RRF) + cross-encoder reranking

Retrieval metrics run on every question (no LLM needed).
Generation metrics need many LLM calls, so --limit controls how many questions.

Run:  python -m src.evaluation.run_eval --limit 20
"""
import argparse
import json

import numpy as np

from src.config import REPORTS_DIR, LLM_PROVIDER
from src.evaluation import generation_metrics as gm
from src.evaluation.retrieval_metrics import evaluate_retrieval, load_testset
from src.llm import get_judge
from src.pipeline import RAGPipeline
from src.retrieval.embeddings import get_embeddings
from src.retrieval.hybrid import load_faiss


def generation_eval(pipeline, testset, judge, embeddings):
    rows = []

    for i, row in enumerate(testset, start=1):
        result = pipeline.answer(row["question"])

        scores = {
            "faithfulness": gm.faithfulness(judge, result["answer"], result["contexts"]),
            "answer_relevancy": gm.answer_relevancy(judge, embeddings, row["question"], result["answer"]),
            "context_recall": gm.context_recall(judge, row["reference_answer"], result["contexts"]),
            "latency_ms": sum(result["timings"].values()),
        }
        rows.append(scores)
        print(f"  {i}/{len(testset)} faith={scores['faithfulness']:.2f} "
              f"rel={scores['answer_relevancy']:.2f} recall={scores['context_recall']:.2f}")

    faith = [r["faithfulness"] for r in rows]
    latency = [r["latency_ms"] for r in rows]

    return {
        "faithfulness": float(np.mean(faith)),
        # Share of answers containing at least one unsupported claim
        "hallucination_rate": float(np.mean([f < 1.0 for f in faith])),
        "answer_relevancy": float(np.mean([r["answer_relevancy"] for r in rows])),
        "context_recall": float(np.mean([r["context_recall"] for r in rows])),
        "p50_latency_ms": float(np.percentile(latency, 50)),
        "p95_latency_ms": float(np.percentile(latency, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="questions for generation metrics (0 = skip)")
    args = parser.parse_args()

    testset = load_testset()
    embeddings = get_embeddings()
    vectorstore = load_faiss(embeddings=embeddings)

    pipelines = {
        "baseline_dense": RAGPipeline(mode="dense", vectorstore=vectorstore),
        "hybrid": RAGPipeline(mode="hybrid", vectorstore=vectorstore),
        "final_hybrid_rerank": RAGPipeline(mode="hybrid_rerank", vectorstore=vectorstore),
    }

    report = {"llm": LLM_PROVIDER, "retrieval": {}, "generation": {}}

    for name, pipeline in pipelines.items():
        report["retrieval"][name] = evaluate_retrieval(pipeline.retrieve, testset)
        m = report["retrieval"][name]
        print(f"{name:>20}: hit@5={m['hit_rate']:.2f} mrr={m['mrr']:.2f} precision={m['precision']:.2f}")

    if args.limit:
        judge = get_judge()
        subset = testset[:args.limit]
        for name in ["baseline_dense", "final_hybrid_rerank"]:
            print(f"\nGeneration eval: {name}")
            report["generation"][name] = generation_eval(pipelines[name], subset, judge, embeddings)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))
    print("\nSaved reports/eval_report.json")


if __name__ == "__main__":
    main()
