from langchain_core.documents import Document

from src.retrieval.hybrid import reciprocal_rank_fusion


def docs(*texts):
    return [Document(page_content=t) for t in texts]


def test_doc_in_both_lists_wins():
    dense = docs("a", "b", "c")
    sparse = docs("c", "d", "e")
    fused = reciprocal_rank_fusion([dense, sparse])
    # "c" is the only doc both retrievers found
    assert fused[0].page_content == "c"


def test_no_duplicates():
    fused = reciprocal_rank_fusion([docs("a", "b"), docs("b", "a")])
    assert [d.page_content for d in fused].count("a") == 1
    assert len(fused) == 2


def test_rank_order_respected_for_single_list():
    fused = reciprocal_rank_fusion([docs("x", "y", "z")])
    assert [d.page_content for d in fused] == ["x", "y", "z"]
