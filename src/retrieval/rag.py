from src.pipeline import RAGPipeline


pipeline = RAGPipeline()

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    result = pipeline.answer(query)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:")
    for i, src in enumerate(result["sources"], start=1):
        print(f"[{i}] {src['source']} | page {src['page']}")

    t = result["timings"]
    print(f"\n(retrieval {t['retrieval_ms']:.0f} ms, generation {t['generation_ms']:.0f} ms)")
