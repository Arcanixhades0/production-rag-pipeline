# =========================================
# IMPORTS
# =========================================

# Load PDF documents
from langchain_community.document_loaders import PyPDFLoader

# Split large text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector database
from langchain_community.vectorstores import FAISS

# Ollama local LLM
from langchain_ollama import ChatOllama


# =========================================
# LOAD PDF
# =========================================

print("Loading PDF...")

# IMPORTANT:
# Make sure sample.pdf exists
# inside Chatbot_RAG folder

loader = PyPDFLoader("sample.pdf")

documents = loader.load()

print("PDF loaded successfully!")


# =========================================
# SPLIT DOCUMENTS
# =========================================

print("Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(

    # Maximum chunk size
    chunk_size=1000,

    # Overlap between chunks
    chunk_overlap=150
)

docs = splitter.split_documents(documents)

print(f"Total chunks created: {len(docs)}")


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(

    # Small fast embedding model
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model ready!")


# =========================================
# CREATE VECTOR DATABASE
# =========================================

print("Creating vector database...")

db = FAISS.from_documents(

    docs,

    embeddings
)

print("Vector database ready!")


# =========================================
# CREATE RETRIEVER
# =========================================

retriever = db.as_retriever(

    # Semantic similarity search
    search_type="similarity",

    # Top k chunks
    search_kwargs={"k": 3}
)

print("Retriever ready!")


# =========================================
# LOAD LOCAL GEMMA MODEL
# =========================================

print("Loading Gemma model from Ollama...")

llm = ChatOllama(

    # Your local model
    model="gemma3:4b",

    # Less randomness
    temperature=0
)

print("Gemma model loaded successfully!")


# =========================================
# START CHATBOT
# =========================================

print("\n===================================")
print(" LOCAL PDF RAG CHATBOT READY ")
print(" Type 'exit' to quit ")
print("===================================")


# =========================================
# CHAT LOOP
# =========================================

while True:

    # User input
    question = input("\nAsk Question: ")


    # Exit chatbot
    if question.lower() == "exit":
        break


    # =====================================
    # RETRIEVE RELEVANT CHUNKS
    # =====================================

    print("\nSearching document...")

    retrieved_docs = retriever.invoke(question)


    # =====================================
    # COMBINE CHUNKS INTO CONTEXT
    # =====================================

    context = "\n\n".join(

        [doc.page_content for doc in retrieved_docs]
    )


    # =====================================
    # CREATE FINAL PROMPT
    # =====================================

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.

If the answer is not present in the context,
say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}
"""


    # =====================================
    # ASK GEMMA MODEL
    # =====================================

    print("Thinking...")
    print("Sending prompt to Gemma...")
    response = llm.invoke(prompt)
    print("Response received!")

    # =====================================
    # PRINT ANSWER
    # =====================================

    print("\n===================================")
    print("ANSWER:")
    print("===================================\n")

    print(response.content)
    