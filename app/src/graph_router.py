"""LangGraph flow: contextualize -> inspect -> retrieve -> generate -> critic -> record_history.

The inspect node only records whether the query is negated. v0 does not act on it, so
failures stay visible as a baseline.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.negation import extract_excluded_term, has_unnegated, is_negated
from src.query_rewriting import contextualize_query
from src.rag_pipeline import generate_answer, search


class RagState(TypedDict):
    query: str
    resolved_query: str
    negated: bool
    excluded_term: str | None
    results: list[dict]
    answer: str
    history: Annotated[list[str], operator.add]


def build_graph(collection, embeddings, llm, top_k: int = 10):
    def contextualize(state: RagState) -> dict:
        return {"resolved_query": contextualize_query(state["query"], state["history"], llm)}

    def inspect(state: RagState) -> dict:
        query = state["resolved_query"]
        return {"negated": is_negated(query), "excluded_term": extract_excluded_term(query)}

    def retrieve(state: RagState) -> dict:
        term = state["excluded_term"]
        hits = search(state["resolved_query"], collection, embeddings, top_k)
        results = [
            {
                "rank": rank,
                "score": score,
                "product_id": doc.metadata["product_id"],
                "title": doc.metadata["title"],
                "document": doc.page_content,
                "trap": bool(term) and has_unnegated(doc.metadata["title"], term),
            }
            for rank, (score, doc) in enumerate(hits, start=1)
        ]
        return {"results": results}

    def generate(state: RagState) -> dict:
        from langchain_core.documents import Document

        retrieved = [
            (r["score"], Document(page_content=r["document"], metadata={"product_id": r["product_id"]}))
            for r in state["results"]
        ]
        return {"answer": generate_answer(state["resolved_query"], retrieved, llm)}

    def critic(state: RagState) -> dict:
        context = "\n".join(r["document"] for r in state["results"])
        response = llm.invoke(
            f"Request: {state['resolved_query']}\nAnswer: {state['answer']}\nEvidence:\n{context}\n\n"
            "If the answer is not supported by the evidence, put '⚠️ Check: ' and the reason first, "
            "then return the answer. Otherwise return the answer unchanged."
        )
        return {"answer": response.text.strip()}

    def record_history(state: RagState) -> dict:
        return {"history": [f"Q: {state['resolved_query']}\nA: {state['answer']}"]}

    graph = StateGraph(RagState)
    steps = [contextualize, inspect, retrieve, generate, critic, record_history]
    for step in steps:
        graph.add_node(step.__name__, step)
    graph.add_edge(START, "contextualize")
    for current, following in zip(steps, steps[1:]):
        graph.add_edge(current.__name__, following.__name__)
    graph.add_edge("record_history", END)
    return graph.compile(checkpointer=MemorySaver())
