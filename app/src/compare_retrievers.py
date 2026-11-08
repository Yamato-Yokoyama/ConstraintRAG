"""Compare TF-IDF, BM25, dense and hybrid search over the same index, with significance tests."""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

from src import metrics
from src.eval_retrieval import GAIN, K
from src.load_esci import DATA_DIR, eval_query_ids, load_corpus, load_eval_queries, load_examples
from src.negation import has_unnegated
from src.rag_pipeline import build_index, get_embeddings, search

RRF_K = 60
POOL = 100


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w\w+\b", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, product_id in enumerate(ranking, start=1):
            scores[product_id] = scores.get(product_id, 0.0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def main() -> None:
    queries = load_eval_queries()
    examples = load_examples()
    corpus = load_corpus().reset_index(drop=True)
    ids = corpus["product_id"].tolist()
    titles = dict(zip(ids, corpus["title"]))
    bm25 = BM25Okapi([tokenize(d) for d in corpus["document"]])
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus["document"])
    embeddings = get_embeddings()
    collection = build_index(embeddings)
    terms = {qid: term for qid, term in queries["negated"]}

    rows = []
    for qid in eval_query_ids(queries):
        group = examples[examples["query_id"] == qid]
        query = group["query"].iloc[0]
        labels = dict(zip(group["product_id"], group["esci_label"]))
        ideal = [GAIN[label] for label in group["esci_label"]]
        rankings = {
            "tfidf": [ids[i] for i in np.argsort(-(matrix @ vectorizer.transform([query]).T).toarray().ravel())[:POOL]],
            "bm25": [ids[i] for i in np.argsort(-bm25.get_scores(tokenize(query)))[:POOL]],
            "dense": [doc.metadata["product_id"] for _, doc in search(query, collection, embeddings, POOL)],
        }
        rankings["hybrid"] = reciprocal_rank_fusion([rankings["bm25"], rankings["dense"]])
        for model, ranking in rankings.items():
            top = ranking[:K]
            row = {
                "query_id": qid,
                "group": "negated" if qid in terms else "plain",
                "model": model,
                "ndcg@10": metrics.ndcg([GAIN.get(labels.get(p), 0.0) for p in top], K, ideal),
            }
            if qid in terms:
                row["trap@10"] = sum(has_unnegated(titles[p], terms[qid]) for p in top) / K
            rows.append(row)

    def values(model, group, key="ndcg@10"):
        return [r[key] for r in rows if r["model"] == model and r["group"] == group]

    summary = {}
    for model in ("tfidf", "bm25", "dense", "hybrid"):
        negated, plain = values(model, "negated"), values(model, "plain")
        summary[model] = {
            "ndcg_negated": float(np.mean(negated)),
            "ndcg_plain": float(np.mean(plain)),
            "trap@10": float(np.mean(values(model, "negated", "trap@10"))),
            "mann_whitney_p_negated_vs_plain": float(stats.mannwhitneyu(negated, plain, alternative="two-sided").pvalue),
        }
    pairs = {}
    for model in ("tfidf", "bm25", "hybrid"):
        for group in ("negated", "plain"):
            pairs[f"{model}_vs_dense_{group}_wilcoxon_p"] = float(stats.wilcoxon(values(model, group), values("dense", group), zero_method="zsplit").pvalue)
    output = {"summary": summary, "paired": pairs, "note": "n=20 per group; p values are not corrected for multiple comparisons"}
    Path("results/compare_retrievers.json").write_text(json.dumps({**output, "rows": rows}, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
