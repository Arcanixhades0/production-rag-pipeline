"""
A/B test of chunking strategies.

Same documents, same test set, same retrieval code. Only the chunker changes.
Writes reports/chunking_ab.json and reports/chunking_ab.md

Run:  python -m src.experiments.chunking_ab
      python -m src.experiments.chunking_ab --strategies semantic sentence_window

Results are merged into reports/chunking_ab.json after EACH strategy,
so a crash in one strategy never loses the others.
"""
import argparse
import json
import time

from langchain_community.vectorstores import FAISS

from src.config import REPORTS_DIR
from src.evaluation.retrieval_metrics import evaluate_retrieval, load_testset
from src.ingestion.chunker import STRATEGIES, create_chunks
from src.ingestion.document_loader import load_documents
from src.pipeline import RAGPipeline
from src.retrieval.embeddings import get_embeddings
from src.retrieval.reranker import Reranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    args = parser.parse_args()

    testset = load_testset()
    documents = load_documents()
    embeddings = get_embeddings()
    reranker = Reranker()

    out = REPORTS_DIR / "chunking_ab.json"
    results = json.loads(out.read_text()) if out.exists() else {}

    for strategy in args.strategies:
        print(f"\n=== {strategy} ===")

        start = time.perf_counter()
        chunks = create_chunks(documents, strategy, embeddings)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        build_s = time.perf_counter() - start
        print(f"{len(chunks)} chunks, indexed in {build_s:.0f}s")

        results[strategy] = {"chunks": len(chunks), "build_seconds": round(build_s)}

        for mode in ["dense", "hybrid_rerank"]:
            pipeline = RAGPipeline(mode=mode, vectorstore=vectorstore, reranker=reranker)
            metrics = evaluate_retrieval(pipeline.retrieve, testset)
            results[strategy][mode] = metrics
            print(f"{mode:>14}: hit@5={metrics['hit_rate']:.2f}  mrr={metrics['mrr']:.2f}  "
                  f"precision={metrics['precision']:.2f}  context={metrics['avg_context_chars']:.0f} chars")

        # Save after every strategy, so a crash later never loses finished results
        REPORTS_DIR.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        (REPORTS_DIR / "chunking_ab.md").write_text(to_markdown(results, len(testset)))
        print(f"Saved {strategy} to reports/chunking_ab.md")


def to_markdown(results, n):
    lines = [
        f"# Chunking A/B test ({n} questions, top-5)",
        "",
        "| Strategy | Chunks | Mode | Hit@5 | MRR | Precision@5 | Avg context (chars) | p95 retrieval (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for strategy, r in results.items():
        for mode in ["dense", "hybrid_rerank"]:
            m = r[mode]
            lines.append(
                f"| {strategy} | {r['chunks']} | {mode} | {m['hit_rate']:.2f} | {m['mrr']:.2f} | "
                f"{m['precision']:.2f} | {m['avg_context_chars']:.0f} | {m['p95_ms']:.0f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
