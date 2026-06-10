from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vectorstore = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0
)

print("Personal Knowledge Base Assistant")
print("Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(query, k=7)

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Prompt
    prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the user's question.

If the answer exists in the context, provide a detailed explanation.

If the answer is not found, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)

    print("\nSources:")
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        print(f"- {source} (Page {page})")

    print("\n" + "=" * 60 + "\n")