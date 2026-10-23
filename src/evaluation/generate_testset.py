"""
Builds a synthetic QA test set: data/eval/testset.jsonl

For a random sample of pages, the LLM writes one question that page answers,
plus a reference answer. We store WHICH page it came from, so retrieval is
"correct" if that page comes back. Page-level labels stay valid no matter
how we chunk, which is what makes the chunking A/B test fair.
"""
import json
import random
import re

from src.config import EVAL_DIR
from src.ingestion.document_loader import load_documents
from src.llm import get_llm

PER_SOURCE = {"pdf": 45, "web": 3, "sql": 2}
MIN_CHARS = 800
LEAKY = re.compile(r"(passage|this text|the author|this chapter|the text)", re.IGNORECASE)

PROMPT = """You are writing a test question for a search system.

Read the passage and write ONE specific question that this passage answers.
Rules:
- The question must make sense on its own. Never say "the passage", "this text", "the author", "this chapter".
- Ask about a concrete fact, definition or explanation, not about code formatting or page layout.
- Then write a short reference answer (1-3 sentences) using only the passage.

Return ONLY JSON: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""


def page_id(doc):
    return {"source": doc.metadata.get("source"), "page": doc.metadata.get("page")}


def sample_pages(documents, rng):
    picked = []

    for source_type, n in PER_SOURCE.items():
        pool = [d for d in documents if d.metadata.get("source_type") == source_type]

        if source_type == "pdf":
            pool = [d for d in pool if len(d.page_content.strip()) >= MIN_CHARS]
            # Spread questions evenly across the books
            books = sorted({d.metadata["source"] for d in pool})
            for book in books:
                pages = [d for d in pool if d.metadata["source"] == book]
                picked += rng.sample(pages, min(n // len(books), len(pages)))
        else:
            picked += rng.sample(pool, min(n, len(pool)))

    return picked


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def main(seed=42):
    rng = random.Random(seed)
    llm = get_llm()

    pages = sample_pages(load_documents(), rng)
    print(f"Generating questions for {len(pages)} pages...")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "testset.jsonl"

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(pages, start=1):
            try:
                qa = parse_json(llm.invoke(PROMPT.format(passage=doc.page_content[:3000])).content)
            except Exception as e:
                print(f"{i}: failed ({e})")
                continue

            if not qa or not qa.get("question"):
                print(f"{i}: unparseable, skipped")
                continue

            # Questions that point at "the passage" are not answerable on their own
            if LEAKY.search(qa["question"]):
                print(f"{i}: refers to the passage, skipped")
                continue

            row = {
                "question": qa["question"],
                "reference_answer": qa.get("answer", ""),
                "source_type": doc.metadata.get("source_type"),
                **page_id(doc),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(f"{i}/{len(pages)}: {row['question'][:90]}")

    print(f"\nWrote {written} questions to {out_path}")


if __name__ == "__main__":
    main()
