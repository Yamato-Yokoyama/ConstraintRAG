"""Compare v0 with and without the caption-based constraint layer on the negated queries."""
import json
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src import metrics
from src.constraint_layer import captions_db, filter_candidates, parse_constraints
from src.eval_retrieval import GAIN, K
from src.negation import has_unnegated
from src.rag_pipeline import build_index, get_embeddings, search
from src.vision.common import CAPTIONS_PATH, negated_products

POOL = 50


def score(ids, labels, titles, term, ideal):
    top = ids[:K]
    gains = [GAIN.get(labels.get(p), 0.0) for p in top]
    traps = sum(has_unnegated(titles.get(p, ""), term) for p in top) / max(len(top), 1)
    return metrics.ndcg(gains, K, ideal), traps


def main() -> None:
    load_dotenv()
    use_llm = "--llm" in sys.argv
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), temperature=0, google_api_key=os.getenv("GEMINI_API_KEY")) if use_llm else None
    captions = {r["product_id"]: r["caption"] for r in map(json.loads, CAPTIONS_PATH.read_text().splitlines())}
    connection = captions_db(captions)
    embeddings = get_embeddings()
    collection = build_index(embeddings)
    table = negated_products()
    rows = []
    for query_id, group in table.groupby("query_id"):
        query, term = group["query"].iloc[0], group["excluded_term"].iloc[0]
        labels = dict(zip(group["product_id"], group["esci_label"]))
        hits = search(query, collection, embeddings, POOL)
        ids = [doc.metadata["product_id"] for _, doc in hits]
        titles = {doc.metadata["product_id"]: doc.metadata["title"] for _, doc in hits}
        # Without --llm, use the hand-labeled excluded term. This tests captions, not query parsing.
        constraints = parse_constraints(query, llm) if use_llm else {"include": [], "exclude": [term]}
        filtered = filter_candidates(connection, ids, constraints)
        ideal = [GAIN[label] for label in group["esci_label"]]
        before = score(ids, labels, titles, term, ideal)
        after = score(filtered, labels, titles, term, ideal)
        rows.append({"query_id": int(query_id), "query": query, "constraints": constraints, "ndcg_before": before[0], "ndcg_after": after[0], "trap_before": before[1], "trap_after": after[1], "removed": len(ids) - len(filtered)})
    summary = {key: statistics.mean(r[key] for r in rows) for key in ("ndcg_before", "ndcg_after", "trap_before", "trap_after", "removed")}
    Path("results/eval_constraint_layer.json").write_text(json.dumps({"summary": summary, "queries": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
