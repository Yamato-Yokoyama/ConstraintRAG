"""Evaluate search() on the eval queries. Runs without an LLM."""
import json
import statistics
from pathlib import Path

from src import metrics
from src.load_esci import clean_text, eval_query_ids, load_eval_queries, load_examples
from src.negation import has_unnegated
from src.rag_pipeline import build_index, get_embeddings, search

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
K = 10


def trap_floor(titles: list[str], term: str, k: int = K) -> float:
    """Lowest top-k trap rate possible using only this query's judged products."""
    slots = min(k, len(titles))
    if slots == 0:
        return 0.0
    clean = sum(not has_unnegated(title, term) for title in titles)
    return max(0, slots - clean) / slots


def main() -> None:
    import pandas as pd

    from src.load_esci import DATA_DIR

    queries = load_eval_queries()
    examples = load_examples()
    examples = examples[examples["query_id"].isin(eval_query_ids(queries))]
    products = pd.read_parquet(DATA_DIR / "products_us.parquet", columns=["product_id", "product_title"]).drop_duplicates("product_id")
    judged = examples.merge(products, on="product_id", how="left")

    embeddings = get_embeddings()
    collection = build_index(embeddings)
    terms = {qid: term for qid, term in queries["negated"]}
    rows = []
    for qid in eval_query_ids(queries):
        group = judged[judged["query_id"] == qid]
        query = group["query"].iloc[0]
        labels = dict(zip(group["product_id"], group["esci_label"]))
        hits = search(query, collection, embeddings, K)
        ids = [doc.metadata["product_id"] for _, doc in hits]
        gains = [GAIN.get(labels.get(product_id), 0.0) for product_id in ids]
        row = {
            "query_id": qid,
            "query": query,
            "group": "negated" if qid in terms else "plain",
            "ndcg@10": metrics.ndcg(gains, K, [GAIN[label] for label in group["esci_label"]]),
            "judged_in_top10": sum(product_id in labels for product_id in ids),
            "top10": [doc.metadata["title"] for _, doc in hits],
        }
        if qid in terms:
            term = terms[qid]
            row["excluded_term"] = term
            row["trap@10"] = sum(has_unnegated(doc.metadata["title"], term) for _, doc in hits) / K
            row["trap_floor"] = trap_floor([clean_text(t) for t in group["product_title"]], term)
        rows.append(row)

    summary = {}
    for name in ("negated", "plain"):
        subset = [r for r in rows if r["group"] == name]
        summary[name] = {
            "n": len(subset),
            "mean_ndcg@10": statistics.mean(r["ndcg@10"] for r in subset),
            "mean_judged_in_top10": statistics.mean(r["judged_in_top10"] for r in subset),
        }
    negated = [r for r in rows if r["group"] == "negated"]
    summary["negated"]["mean_trap@10"] = statistics.mean(r["trap@10"] for r in negated)
    summary["negated"]["mean_trap_floor"] = statistics.mean(r["trap_floor"] for r in negated)
    summary["corpus_size"] = collection.count()

    Path("results").mkdir(exist_ok=True)
    Path("results/eval_v0.json").write_text(json.dumps({"summary": summary, "queries": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
