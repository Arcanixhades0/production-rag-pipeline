import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

from embeddings import get_embeddings


embeddings = get_embeddings()

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0
)

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Answer the question using only the provided context.

If the answer is not in the context, say:
'I could not find that information in the knowledge base.'

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)