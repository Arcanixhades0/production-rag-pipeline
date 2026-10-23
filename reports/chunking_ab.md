# Chunking A/B test (50 questions, top-5)

| Strategy | Chunks | Mode | Hit@5 | MRR | Precision@5 | Avg context (chars) | p95 retrieval (ms) |
|---|---|---|---|---|---|---|---|
| fixed | 2519 | dense | 0.84 | 0.59 | 0.27 | 3992 | 25 |
| fixed | 2519 | hybrid_rerank | 0.96 | 0.89 | 0.32 | 4499 | 2735 |
| recursive | 2535 | dense | 0.80 | 0.60 | 0.27 | 3898 | 28 |
| recursive | 2535 | hybrid_rerank | 0.94 | 0.88 | 0.31 | 4252 | 2576 |
| sentence_window | 12457 | dense | 0.82 | 0.68 | 0.33 | 4175 | 120 |
| sentence_window | 12457 | hybrid_rerank | 0.98 | 0.92 | 0.50 | 4086 | 4696 |
| semantic | 2741 | dense | 0.82 | 0.61 | 0.25 | 3302 | 24 |
| semantic | 2741 | hybrid_rerank | 0.96 | 0.90 | 0.28 | 4278 | 3382 |
