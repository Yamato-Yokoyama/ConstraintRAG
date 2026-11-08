from types import SimpleNamespace

import src.graph_router as gr
from langchain_core.documents import Document


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return SimpleNamespace(text="fake answer")


def test_graph_marks_trap_and_keeps_history(monkeypatch):
    doc = Document(page_content="Chicken Treats", metadata={"product_id": "A", "title": "Chicken Treats"})
    monkeypatch.setattr(gr, "search", lambda *args: [(0.9, doc)])
    graph = gr.build_graph(None, None, FakeLLM())
    config = {"configurable": {"thread_id": "test"}}

    state = graph.invoke({"query": "cat treats without chicken"}, config)
    assert state["negated"] is True
    assert state["results"][0]["trap"] is True
    assert len(state["history"]) == 1

    state = graph.invoke({"query": "and those?"}, config)
    assert state["resolved_query"] == "fake answer"
    assert len(state["history"]) == 2
