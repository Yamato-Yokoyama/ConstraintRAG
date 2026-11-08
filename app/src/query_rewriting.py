"""Rewrite a follow-up query so that it can be searched on its own."""
from langchain_core.messages import HumanMessage, SystemMessage


def contextualize_query(query: str, history: list[str], llm) -> str:
    if not history:
        return query
    system_prompt = (
        "Rewrite the latest request so it can be understood without the conversation. "
        "Replace references such as 'that one' or 'those' with the products they refer to. "
        "Keep every exclusion the user stated. If there is nothing to replace, return it unchanged. "
        "Return only the rewritten request."
    )
    history_text = "\n".join(history)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Conversation:\n{history_text}\n\nLatest request: {query}"),
    ])
    return response.text.strip()
