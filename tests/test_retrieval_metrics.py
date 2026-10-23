from langchain_core.documents import Document

from src.evaluation.retrieval_metrics import evaluate_retrieval

TESTSET = [
    {"question": "q1", "source": "a.pdf", "page": 1},
    {"question": "q2", "source": "b.pdf", "page": 2},
]


def doc(source, page):
    return Document(page_content="x", metadata={"source": source, "page": page})


def test_perfect_retriever():
    lookup = {"q1": [doc("a.pdf", 1)], "q2": [doc("b.pdf", 2)]}
    m = evaluate_retrieval(lambda q: lookup[q], TESTSET, k=1)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 1.0


def test_right_page_at_rank_two_gives_half_mrr():
    lookup = {
        "q1": [doc("a.pdf", 9), doc("a.pdf", 1)],
        "q2": [doc("z.pdf", 0), doc("b.pdf", 2)],
    }
    m = evaluate_retrieval(lambda q: lookup[q], TESTSET, k=2)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision"] == 0.5


def test_wrong_page_same_book_is_a_miss():
    m = evaluate_retrieval(lambda q: [doc("a.pdf", 99)], TESTSET[:1], k=1)
    assert m["hit_rate"] == 0.0
