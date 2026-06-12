import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from ingest_utils import build_faiss_index
from pathlib import Path

st.set_page_config(
    page_title="Personal Knowledge Base",
    page_icon="📚"
)


st.title("📚 Personal Knowledge Base Assistant")

st.sidebar.header("Knowledge Base Management")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)
if uploaded_file is not None:

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(
        f"Uploaded: {uploaded_file.name}"
    )
    
if st.sidebar.button("Rebuild Knowledge Base"):

    with st.spinner("Rebuilding FAISS index..."):

        pdfs, pages, chunks = build_faiss_index()

    st.sidebar.success(
        f"Done! PDFs={pdfs}, Pages={pages}, Chunks={chunks}"
    )

@st.cache_resource
def load_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(
        model="qwen3.5:9b",
        temperature=0
    )

    return vectorstore, llm


vectorstore, llm = load_components()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)


query = st.chat_input("Ask a question about your PDFs...")


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            docs = vectorstore.similarity_search(
                query,
                k=5
            )

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            prompt = f"""
You are a helpful AI assistant.

Use the provided context to answer the user's question.

If the answer is not found in the context, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)

            answer = response.content

            st.markdown(answer)

            sources = []

            for doc in docs:

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                sources.append(
                    f"{source} (Page {page})"
                )

            with st.expander("Sources"):

                for source in sources:
                    st.write(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )