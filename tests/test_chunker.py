import pytest
from langchain_core.documents import Document

from src.config import CHUNK_SIZE
from src.ingestion.chunker import create_chunks, expand_windows

PAGE = Document(
    page_content=" ".join(f"This is sentence number {i} about retrieval." for i in range(300)),
    metadata={"source": "book.pdf", "page": 7},
)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_chunks_respect_size_and_keep_metadata(strategy):
    chunks = create_chunks([PAGE], strategy)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all(c.metadata["page"] == 7 for c in chunks)


def test_sentence_window_indexes_sentences_and_expands():
    chunks = create_chunks([PAGE], "sentence_window")
    middle = chunks[150]
    assert middle.page_content.count(".") == 1          # one sentence is indexed
    expanded = expand_windows([middle])[0]
    assert expanded.page_content.count(".") == 7        # 3 before + itself + 3 after
    assert expanded.metadata["page"] == 7


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        create_chunks([PAGE], "magic")
