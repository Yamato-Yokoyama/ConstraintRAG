"""Train logistic regression on the ESCI train split, then rerank dense results for the eval queries."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from src import metrics
from src.eval_retrieval import GAIN, K
from src.load_esci import DATA_DIR, clean_text, eval_query_ids, load_corpus, load_eval_queries, load_examples, make_document
from src.negation import has_unnegated
from src.rag_pipeline import build_index, get_embeddings, search

TRAIN_ROWS = 200_000
POOL = 100
SEED = 42


def features(queries: list[str], titles: list[str], documents: list[str], vectorizer: TfidfVectorizer):
    """Query-document matches, plus words from each side."""
    q = vectorizer.transform(queries)
    t = vectorizer.transform(titles)
    d = vectorizer.transform(documents)
    numeric = np.column_stack([
        np.asarray(q.multiply(t).sum(axis=1)).ravel(),
        np.asarray(q.multiply(d).sum(axis=1)).ravel(),
        np.asarray((q > 0).multiply(t > 0).sum(axis=1)).ravel() / np.maximum(np.asarray((q > 0).sum(axis=1)).ravel(), 1),
    ])
    return sparse.hstack([sparse.csr_matrix(numeric), q, t], format="csr")


def main() -> None:
    examples = load_examples()
    products = pd.read_parquet(DATA_DIR / "products_us.parquet", columns=["product_id", "product_title", "product_description"]).drop_duplicates("product_id")
    train = examples[examples["split"] == "train"].sample(n=TRAIN_ROWS, random_state=SEED).merge(products, on="product_id")
    train["title"] = train["product_title"].map(clean_text)
    train["document"] = [make_document(a, b) for a, b in zip(train["product_title"], train["product_description"])]
    train["label"] = (train["esci_label"] == "E").astype(int)
    fit_index, valid_index = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(train, groups=train["query_id"]))
    fit, valid = train.iloc[fit_index], train.iloc[valid_index]

    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    vectorizer.fit(pd.concat([fit["query"], fit["title"]]))
    model = LogisticRegression(max_iter=300, C=1.0)
    model.fit(features(fit["query"].tolist(), fit["title"].tolist(), fit["document"].tolist(), vectorizer), fit["label"])
    valid_scores = model.predict_proba(features(valid["query"].tolist(), valid["title"].tolist(), valid["document"].tolist(), vectorizer))[:, 1]
    print("validation average precision", average_precision_score(valid["label"], valid_scores), "base rate", float(valid["label"].mean()), flush=True)

    queries = load_eval_queries()
    terms = {qid: term for qid, term in queries["negated"]}
    embeddings = get_embeddings()
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
        scores = model.predict_proba(features([query] * len(ids), titles, documents, vectorizer))[:, 1]
        order = np.argsort(-scores)
        for name, ranking in (("dense", list(range(len(ids)))), ("logreg", order.tolist())):
            top = ranking[:K]
            row = {"query_id": qid, "group": "negated" if qid in terms else "plain", "model": name,
                   "ndcg@10": metrics.ndcg([GAIN.get(labels.get(ids[i]), 0.0) for i in top], K, ideal)}
            if qid in terms:
                row["trap@10"] = sum(has_unnegated(titles[i], terms[qid]) for i in top) / K
            rows.append(row)

    def values(model_name, group, key="ndcg@10"):
        return [r[key] for r in rows if r["model"] == model_name and r["group"] == group]

    summary = {name: {"ndcg_negated": float(np.mean(values(name, "negated"))), "ndcg_plain": float(np.mean(values(name, "plain"))), "trap@10": float(np.mean(values(name, "negated", "trap@10")))} for name in ("dense", "logreg")}
    summary["wilcoxon_p"] = {
        "ndcg_negated": float(stats.wilcoxon(values("logreg", "negated"), values("dense", "negated"), zero_method="zsplit").pvalue),
        "ndcg_plain": float(stats.wilcoxon(values("logreg", "plain"), values("dense", "plain"), zero_method="zsplit").pvalue),
        "trap": float(stats.wilcoxon(values("logreg", "negated", "trap@10"), values("dense", "negated", "trap@10"), zero_method="zsplit").pvalue),
    }
    Path("results/ml_reranker.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
