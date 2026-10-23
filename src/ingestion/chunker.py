import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

STRATEGIES = ["fixed", "recursive", "semantic", "sentence_window"]


def fixed_chunks(documents):
    # Cuts every CHUNK_SIZE characters, ignoring sentences and paragraphs
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def recursive_chunks(documents):
    # Tries to cut at paragraph breaks, then lines, then sentences, then words
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def semantic_chunks(documents, embeddings):
    # Embeds every sentence and cuts where the meaning shifts sharply
    # between neighbouring sentences
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks = splitter.split_documents(documents)

    # Semantic chunks can come out huge; cap them so they fit the embedding model
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 2, chunk_overlap=0
    ).split_documents(chunks)


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentence_window_chunks(documents, window=3):
    """
    Index single sentences (precise matching), but store the surrounding
    `window` sentences on each side in metadata. After retrieval we swap
    in the window, so the LLM sees enough context to answer.
    """
    chunks = []

    for doc in documents:
        text = " ".join(doc.page_content.split())
        sentences = [s for s in SENTENCE_END.split(text) if len(s) > 20]

        for i, sentence in enumerate(sentences):
            start, end = max(0, i - window), i + window + 1
            chunks.append(Document(
                page_content=sentence,
                metadata={**doc.metadata, "window": " ".join(sentences[start:end])},
            ))

    return chunks


def expand_windows(docs):
    # Replace each matched sentence with its surrounding window (sentence-window only)
    return [
        Document(page_content=d.metadata["window"], metadata=d.metadata)
        if "window" in d.metadata else d
        for d in docs
    ]


def create_chunks(documents, strategy="recursive", embeddings=None):

    if strategy == "fixed":
        return fixed_chunks(documents)

    if strategy == "recursive":
        return recursive_chunks(documents)

    if strategy == "semantic":
        return semantic_chunks(documents, embeddings)

    if strategy == "sentence_window":
        return sentence_window_chunks(documents)

    raise ValueError(f"Unknown strategy: {strategy}. Choose from {STRATEGIES}")
