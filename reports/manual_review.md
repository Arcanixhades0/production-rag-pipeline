# Manual review of LLM-judge failures

Judge: `openai/gpt-oss-120b` (a different model family from the generator `qwen/qwen3.8-27b`)
Run: `python -m src.evaluation.run_eval --limit 20` → `eval_report.json`

## Faithfulness failures

| Pipeline | Question | Judge score | Manual verdict |
|---|---|---|---|
| final_hybrid_rerank | #6 "Which paper introduced the concept of using contrastive captioners as image-text foundation models?" | 0.00 | **Judge error.** Answer ("CoCa: Contrastive Captioners Are Image-Text Foundation Models", Jiahui Yu et al.) is correct and supported by context [1]. Re-running the same judge on the same answer marked all 3 claims supported. |

## Corrected numbers

| | Judge-reported | After manual review |
|---|---|---|
| Hallucination rate, dense baseline | 0% | 0% |
| Hallucination rate, hybrid + rerank | 5% | **0%** |

## Takeaways

- LLM judges are noisy: the same judge gave opposite verdicts on the same answer in two runs.
- Always read the failures before reporting a metric. One wrong verdict out of 20 moves the rate by 5 points.
- The baseline's perfect faithfulness is partly because it answers "I could not find that" more often
  (context recall 0.69 vs 0.86). A refusal makes no claims, so it can't be unfaithful.
  Faithfulness must be read together with recall/relevancy, never alone.
