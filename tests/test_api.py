from fastapi.testclient import TestClient

from src.api.main import app, state


class FakePipeline:
    def answer(self, question, k=5, source_type=None):
        return {"answer": f"echo: {question}", "contexts": [], "sources": [], "timings": {}}

    def stream(self, question, k=5, source_type=None):
        yield {"type": "sources", "data": []}
        yield {"type": "token", "data": "hi"}


# No "with TestClient(...)", so the real lifespan (which loads models) never runs
client = TestClient(app)
state["pipeline"] = FakePipeline()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_query():
    r = client.post("/query", json={"question": "what is rag"})
    assert r.status_code == 200
    assert r.json()["answer"] == "echo: what is rag"


def test_rejects_bad_input():
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "what is rag", "source_type": "excel"}).status_code == 422


def test_stream():
    r = client.post("/query/stream", json={"question": "what is rag"})
    assert '"token"' in r.text
    assert r.text.strip().endswith("[DONE]")
