"""Chat UI. The top 10 table shows where retrieval failed."""
import asyncio
import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.graph_router import build_graph
from src.rag_pipeline import build_index, get_embeddings

load_dotenv()

embeddings = get_embeddings()
collection = build_index(embeddings)
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
graph = build_graph(collection, embeddings, llm)


def format_results(state: dict) -> str:
    term = state["excluded_term"] or "none"
    traps = sum(r["trap"] for r in state["results"])
    lines = [
        f"**Searched:** {state['resolved_query']}",
        f"**Negated:** {state['negated']} | **Excluded term:** {term} | **Traps in top {len(state['results'])}:** {traps}",
        "",
        "| Rank | Trap | Score | Product |",
        "|---|---|---|---|",
    ]
    for r in state["results"]:
        title = r["title"].replace("|", "/")[:90]
        lines.append(f"| {r['rank']} | {'🟥' if r['trap'] else ''} | {r['score']:.3f} | {title} |")
    return "\n".join(lines)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="Negated query", message="cat treats without chicken"),
        cl.Starter(label="Another negated query", message="alarm clock without snooze"),
        cl.Starter(label="Plain query", message="small dog brush for yorkie"),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    state = await asyncio.to_thread(graph.invoke, {"query": message.content}, config)
    await cl.Message(content=f"{state['answer']}\n\n{format_results(state)}").send()
