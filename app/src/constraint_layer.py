"""Text -> SQL conditions. Separate from the v0 baseline and used only for comparison.

The LLM returns structured terms. The SQL is built with bound parameters, not written by the LLM.
"""
import json
import re
import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage


def parse_constraints(query: str, llm) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            'Extract product search constraints as JSON: {"include": [words], "exclude": [words]}. '
            "Use single lowercase words or short phrases. Return JSON only."
        )),
        HumanMessage(content=query),
    ])
    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    return {"include": list(data.get("include", [])), "exclude": list(data.get("exclude", []))}


def build_sql(constraints: dict, candidate_count: int) -> tuple[str, list[str]]:
    """Only exclusions filter rows. A short caption rarely contains every include word."""
    placeholders = ",".join("?" * candidate_count)
    sql = f"SELECT product_id FROM captions WHERE product_id IN ({placeholders})"
    params = []
    for term in constraints["exclude"]:
        sql += " AND lower(caption) NOT LIKE ?"
        params.append(f"%{term.lower()}%")
    return sql, params


def filter_candidates(connection: sqlite3.Connection, candidate_ids: list[str], constraints: dict) -> list[str]:
    """Keep ranking order. Products without a caption stay, because nothing is known about them."""
    known = {row[0] for row in connection.execute("SELECT product_id FROM captions")}
    sql, params = build_sql(constraints, len(candidate_ids))
    allowed = {row[0] for row in connection.execute(sql, candidate_ids + params)}
    return [p for p in candidate_ids if p in allowed or p not in known]


def captions_db(captions: dict[str, str]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE captions (product_id TEXT PRIMARY KEY, caption TEXT)")
    connection.executemany("INSERT INTO captions VALUES (?, ?)", captions.items())
    return connection
