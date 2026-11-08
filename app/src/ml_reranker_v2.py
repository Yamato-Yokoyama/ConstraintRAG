"""Second ML experiment: test whether adding random negatives and the dense score helps.

A: first model's features, trained with random products added as negatives.
B: four numeric features including the dense score, on a smaller sample because embedding is slow.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import metrics
from src.eval_retrieval import GAIN, K
from src.load_esci import DATA_DIR, clean_text, eval_query_ids, load_eval_queries, load_examples, make_document
from src.ml_reranker import POOL, SEED, features
from src.negation import has_unnegated
from src.rag_pipeline import build_index, get_embeddings, search

A_ROWS = 100_000
B_ROWS = 2_000


def with_random_negatives(sample: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    positives = sample.merge(products, on="product_id")
    positives["label"] = (positives["esci_label"] == "E").astype(int)
    negatives = sample[["query"]].copy()
    picked = products.sample(n=len(negatives), replace=True, random_state=SEED).reset_index(drop=True)
    negatives = pd.concat([negatives.reset_index(drop=True), picked], axis=1)
    negatives["label"] = 0
    data = pd.concat([positives, negatives], ignore_index=True)
    data["title"] = data["product_title"].map(clean_text)
    data["document"] = [make_document(a, b) for a, b in zip(data["product_title"], data["product_description"])]
    return data


def numeric(queries, titles, documents, dense_scores, vectorizer):
    matrix = features(queries, titles, documents, vectorizer)[:, :3].toarray()
    return np.column_stack([matrix, dense_scores])


def main() -> None:
    examples = load_examples()
    products = pd.read_parquet(DATA_DIR / "products_us.parquet", columns=["product_id", "product_title", "product_description"]).drop_duplicates("product_id")
    train = examples[examples["split"] == "train"]
    data_a = with_random_negatives(train.sample(n=A_ROWS, random_state=SEED), products)
    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    vectorizer.fit(pd.concat([data_a["query"], data_a["title"]]))
    model_a = LogisticRegression(max_iter=300)
    model_a.fit(features(data_a["query"].tolist(), data_a["title"].tolist(), data_a["document"].tolist(), vectorizer), data_a["label"])
    print("trained A", flush=True)

    embeddings = get_embeddings()
    data_b = with_random_negatives(train.sample(n=B_ROWS, random_state=SEED + 1), products)
    query_vectors = np.array(embeddings.embed_documents(data_b["query"].tolist()))
    document_vectors = np.array(embeddings.embed_documents(data_b["document"].tolist()))
    dense_b = (query_vectors * document_vectors).sum(axis=1)
    model_b = LogisticRegression(max_iter=300)
    model_b.fit(numeric(data_b["query"].tolist(), data_b["title"].tolist(), data_b["document"].tolist(), dense_b, vectorizer), data_b["label"])
    print("trained B", dict(zip(["q_title", "q_doc", "overlap", "dense"], model_b.coef_[0].round(3).tolist())), flush=True)

    queries = load_eval_queries()
    terms = {qid: term for qid, term in queries["negated"]}
    collection = build_index(embeddings)
    rows = []
    for qid in eval_query_ids(queries):
        group = examples[examples["query_id"] == qid]
        query = group["query"].iloc[0]
        labels = dict(zip(group["product_id"], group["esci_label"]))
        ideal = [GAIN[label] for label in group["esci_label"]]
        hits = search(query, collection, embeddings, POOL)
        ids = [doc.metadata["product_id"] for _, doc in hits]
        titles = [doc.metadata["title"] for _, doc in hits]
        documents = [doc.page_content for _, doc in hits]
        dense = np.array([score for score, _ in hits])
        repeated = [query] * len(ids)
        rankings = {
            "dense": list(range(len(ids))),
            "A_random_negatives": np.argsort(-model_a.predict_proba(features(repeated, titles, documents, vectorizer))[:, 1]).tolist(),
            "B_dense_feature": np.argsort(-model_b.predict_proba(numeric(repeated, titles, documents, dense, vectorizer))[:, 1]).tolist(),
        }
        for name, ranking in rankings.items():
            top = ranking[:K]
            row = {"query_id": qid, "group": "negated" if qid in terms else "plain", "model": name,
                   "ndcg@10": metrics.ndcg([GAIN.get(labels.get(ids[i]), 0.0) for i in top], K, ideal)}
            if qid in terms:
                row["trap@10"] = sum(has_unnegated(titles[i], terms[qid]) for i in top) / K
            rows.append(row)

    def values(name, group, key="ndcg@10"):
        return [r[key] for r in rows if r["model"] == name and r["group"] == group]

    summary = {}
    for name in ("dense", "A_random_negatives", "B_dense_feature"):
        summary[name] = {"ndcg_negated": float(np.mean(values(name, "negated"))), "ndcg_plain": float(np.mean(values(name, "plain"))), "trap@10": float(np.mean(values(name, "negated", "trap@10")))}
        if name != "dense":
            summary[name]["wilcoxon_p_vs_dense"] = {
                "ndcg_negated": float(stats.wilcoxon(values(name, "negated"), values("dense", "negated"), zero_method="zsplit").pvalue),
                "ndcg_plain": float(stats.wilcoxon(values(name, "plain"), values("dense", "plain"), zero_method="zsplit").pvalue),
                "trap": float(stats.wilcoxon(values(name, "negated", "trap@10"), values("dense", "negated", "trap@10"), zero_method="zsplit").pvalue),
            }
    Path("results/ml_reranker_v2.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
