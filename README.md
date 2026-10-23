# Production RAG Pipeline

A retrieval-augmented generation pipeline over PDFs, web pages and a SQL table, with hybrid retrieval,
cross-encoder reranking, and an evaluation suite that measures each change instead of guessing.

```
 PDF ─┐                                   ┌─ FAISS (dense) ─┐
 Web ─┼─▶ load ─▶ chunk ─▶ embed ─▶ index ┤                 ├─▶ RRF fusion ─▶ cross-encoder ─▶ top 5 ─▶ LLM ─▶ answer + [citations]
 SQL ─┘                                   └─ BM25 (sparse) ─┘     (20+20)       rerank
```

1. **Ingestion**: PDFs (with a fallback loader for PDFs whose metadata crashes `PyPDFLoader`), web pages, SQL rows turned into sentences
2. **Chunking**: fixed, recursive, semantic or sentence-window (`src/ingestion/chunker.py`)
3. **Indexing**: FAISS + BM25 built from the same chunks; ChromaDB as a metadata-filterable alternative
4. **Retrieval**: dense + sparse, merged with Reciprocal Rank Fusion
5. **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` over the fused candidates
6. **Generation**: grounded prompt with numbered sources; Groq or local Ollama

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                        # GROQ_API_KEY, or leave empty to use local Ollama
# put PDFs in data/
python -m scripts.seed_sql                  # small SQL source
python -m src.retrieval.build_index         # FAISS index
python -m src.retrieval.rag                 # chat in the terminal
uvicorn src.api.main:app                    # REST API, docs at /docs
python -m pytest -q                         # 14 tests, no models or keys needed
```

## Evaluate

```bash
python -m src.evaluation.generate_testset   # 50 questions, each tied to the page it came from
python -m src.evaluation.run_eval --limit 20
python -m src.experiments.chunking_ab
```

## Results

Corpus: 3 technical books (1,039 pages) + 3 web pages + 12 SQL rows → 2,535 chunks.
Test set: 50 LLM-generated questions; retrieval is correct if the source page is in the top 5.

**Retrieval** (`reports/eval_report.json`)

| Pipeline | Hit@5 | MRR | p95 retrieval |
|---|---|---|---|
| Dense only (baseline) | 0.80 | 0.60 | 25 ms |
| Hybrid (FAISS + BM25, RRF) | 0.80 | 0.68 | 37 ms |
| **Hybrid + reranker** | **0.94** | **0.88** | 1.6 s (CPU) |

**Generation**, 20 questions, generator `qwen/qwen3.8-27b`, judge `openai/gpt-oss-120b` (different family on purpose)

| Pipeline | Context recall | Answer relevancy | Hallucinations |
|---|---|---|---|
| Dense baseline | 0.69 | 0.57 | 0 / 20 |
| Hybrid + reranker | **0.86** | **0.70** | 0 / 20* |

\* The judge flagged 1 answer; manual review showed the answer was correct (`reports/manual_review.md`).

**Chunking A/B** (`reports/chunking_ab.md`), hybrid + reranker

| Strategy | Hit@5 | MRR | Build time |
|---|---|---|---|
| fixed | 0.96 | 0.89 | 2 min |
| recursive | 0.94 | 0.88 | 2.5 min |
| sentence-window | **0.98** | **0.92** | 2 min |
| semantic | 0.96 | 0.90 | 10 min |

## What I learned

- **Reranking mattered far more than chunking**: +14 pts Hit@5 vs ~2 pts. On 50 questions, 2 pts is one question, i.e. noise.
- **A third of the corpus was silently missing**: one PDF's corrupt page labels crashed the loader and the
  `try/except` skipped it. Now there's a fallback loader and per-file page counts.
- **LLM judges are noisy**: the same judge gave opposite verdicts on the same answer. Read failures before reporting a metric.
- **Sentence-window precision is inflated**: neighbouring windows from the same page count as separate hits
  but give the LLM duplicate text. Merging overlapping windows is the next fix.
- **Reranking costs latency**: ~1.6–2.5 s on CPU for 40 candidates. Fewer candidates, a smaller model or a GPU would bring it down.
