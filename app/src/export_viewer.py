"""Write the static data for viewer/: a 2D map of the corpus and one recorded search per eval query.

✍️ Claude wrote this file (2026-11-07, trial at Yamato's request). Not reviewed yet.
Trial only: for school use, Yamato re-implements it.

Run from app/: python -m src.export_viewer
"""
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from src import metrics
from src.eval_retrieval import GAIN, K
from src.load_esci import DATA_DIR, clean_text, eval_query_ids, load_eval_queries, load_examples
from src.negation import FREE_AFTER, NEGATOR_BEFORE, TRAP_SYNONYMS, has_unnegated
from src.rag_pipeline import build_index, get_embeddings

OUT = Path("../viewer/data")
REACTED = 300  # how many nearest products the animation lights up
TITLE_CHARS = 90
SNIPPET_CHARS = 400
ANY_NEGATOR = re.compile(r"\b(?:without|no|not|non|except|excluding|free)\b", re.IGNORECASE)
PRODUCTS_PER_SUB = 150
# Names Claude gave after reading each cluster's frequent words and sample titles (KMeans, seed 42).
# These are groups of similar embeddings, not Amazon's categories.
MAJOR_NAMES = [
    "服（メンズ中心）", "工具・日用品", "電子機器", "服（レディース中心）", "スマホケース", "おもちゃ・子ども",
    "本・映像", "文具・美容・アクセ", "家具・寝具", "ライト・計測器", "食品", "マスク・雑貨",
]


def build_map(collection) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    """t-SNE coordinates scaled to 0-1. For display only: search does not use them."""
    data = collection.get(include=["embeddings", "metadatas"])
    ids = data["ids"]
    vectors = np.asarray(data["embeddings"])
    start = PCA(n_components=50, random_state=42).fit_transform(vectors)
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=42).fit_transform(start)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    full_titles = [m["title"] for m in data["metadatas"]]
    return ids, xy, [t[:TITLE_CHARS] for t in full_titles], vectors, full_titles


def build_tree(vectors: np.ndarray, titles: list[str]) -> tuple[list[dict], list[int]]:
    """Two levels of KMeans. Returns the tree and each product's position in the list of subcategories."""
    majors = KMeans(len(MAJOR_NAMES), n_init=10, random_state=42).fit_predict(vectors)
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000, token_pattern=r"[A-Za-z][A-Za-z-]+")
    matrix = tfidf.fit_transform(titles)
    words = np.array(tfidf.get_feature_names_out())
    tree, sub_of, next_sub = [], [0] * len(titles), 0
    for major, name in enumerate(MAJOR_NAMES):
        members = np.where(majors == major)[0]
        count = max(2, round(len(members) / PRODUCTS_PER_SUB))
        subs = KMeans(count, n_init=10, random_state=42).fit_predict(vectors[members])
        branch = {"name": name, "subs": []}
        for sub in range(count):
            inside = members[subs == sub]
            top = np.asarray(matrix[inside].mean(axis=0)).ravel().argsort()[::-1][:2]
            branch["subs"].append({"name": " / ".join(words[top]), "n": len(inside)})
            for i in inside:
                sub_of[i] = next_sub
            next_sub += 1
        tree.append(branch)
    return tree, sub_of


def explain_hit(title: str, document: str, term: str | None, label: str | None) -> tuple[list[dict], str | None]:
    """Spans to highlight in the title and a sentence saying why it is, or is not, a trap."""
    negators = [m for m in ANY_NEGATOR.finditer(title)]
    spans = [{"s": m.start(), "e": m.end(), "kind": "neg"} for m in negators]
    if not term:
        return spans, None
    words = TRAP_SYNONYMS.get(term.lower(), [term])
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\w*", re.IGNORECASE)
    bad, ok = [], []
    for m in pattern.finditer(title):
        denied = NEGATOR_BEFORE.search(title[: m.start()]) or FREE_AFTER.search(title[m.end() :])
        (ok if denied else bad).append(m)
        spans.append({"s": m.start(), "e": m.end(), "kind": "ok" if denied else "trap"})
    if bad:
        why = f"タイトルに「{bad[0].group()}」が否定なしで入っているので罠です。"
        if negators:
            why += f"「{negators[0].group()}」はありますが、除外語にはかかっていません。"
        if label == "E":
            why += "ただし ESCI の判定は Exact です。"
    elif ok:
        why = f"タイトルの「{ok[0].group()}」は否定されているので、罠ではありません。"
    else:
        why = f"タイトルに「{term}」はありません。"
        body = document[len(title):]
        if pattern.search(body):
            why += "説明文には出ますが、罠の判定はタイトルだけで見ています。"
        if negators:
            why += f"タイトルの「{negators[0].group()}」は除外語とは別の話です。"
        if label == "I":
            why += "否定とは別の理由で Irrelevant です。"
    return sorted(spans, key=lambda x: x["s"]), why


def record_query(qid, group, term, collection, embeddings, index_of, xy) -> dict:
    query = group["query"].iloc[0]
    labels = dict(zip(group["product_id"], group["esci_label"]))

    started = time.perf_counter()
    vector = embeddings.embed_query(query)
    embed_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    found = collection.query(query_embeddings=[vector], n_results=REACTED)
    search_ms = (time.perf_counter() - started) * 1000

    ids = found["ids"][0]
    scores = [1 - d for d in found["distances"][0]]
    hits = []
    for rank in range(K):
        product_id = ids[rank]
        title = found["metadatas"][0][rank]["title"]
        label = labels.get(product_id)
        spans, why = explain_hit(title, found["documents"][0][rank], term, label)
        gain = GAIN.get(label, 0.0)
        hits.append({
            "rank": rank + 1,
            "product_id": product_id,
            "title": title,
            "snippet": clean_text(found["documents"][0][rank])[:SNIPPET_CHARS],
            "score": round(scores[rank], 4),
            "label": label,  # None: ESCI has no judgment for this pair
            "gain": gain,
            "dcg_part": round(gain / math.log2(rank + 2), 4),
            "spans": spans,
            "why": why,
            "trap": bool(term) and has_unnegated(title, term),
        })

    gains = [GAIN.get(hit["label"], 0.0) for hit in hits]
    ideal = [GAIN[label] for label in group["esci_label"]]
    # The query has no real 2D position; show it at the centre of its top 10.
    centre = xy[[index_of[p] for p in ids[:K]]].mean(axis=0)
    return {
        "query_id": qid,
        "query": query,
        "group": "negated" if term else "plain",
        "excluded_term": term,
        "position": [round(float(v), 4) for v in centre],
        "timing_ms": {"embed": round(embed_ms, 1), "search": round(search_ms, 1)},
        "ndcg@10": metrics.ndcg(gains, K, ideal),
        "dcg": round(metrics.dcg(gains, K), 4),
        "ideal_dcg": round(metrics.dcg(sorted(ideal, reverse=True), K), 4),
        "exact_in_top10": sum(hit["label"] == "E" for hit in hits),
        "judged_in_top10": sum(hit["label"] is not None for hit in hits),
        "trap_in_top10": sum(hit["trap"] for hit in hits),
        "reacted": [[index_of[p], round(s, 3)] for p, s in zip(ids, scores)],
        "hits": hits,
        "judge": None,  # slot for a later judgment model: verdict and time
    }


def main() -> None:
    queries = load_eval_queries()
    terms = {qid: term for qid, term in queries["negated"]}
    examples = load_examples()
    examples = examples[examples["query_id"].isin(eval_query_ids(queries))]
    products = pd.read_parquet(DATA_DIR / "products_us.parquet", columns=["product_id"]).drop_duplicates()
    judged = examples.merge(products, on="product_id", how="left")

    embeddings = get_embeddings()
    collection = build_index(embeddings)
    ids, xy, titles, vectors, full_titles = build_map(collection)
    tree, sub_of = build_tree(vectors, full_titles)
    index_of = {product_id: i for i, product_id in enumerate(ids)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "map.json").write_text(json.dumps(
        {"xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy], "titles": titles, "tree": tree, "sub": sub_of},
        ensure_ascii=False,
    ))
    runs = [
        record_query(qid, judged[judged["query_id"] == qid], terms.get(qid), collection, embeddings, index_of, xy)
        for qid in eval_query_ids(queries)
    ]
    (OUT / "runs.json").write_text(json.dumps(runs, ensure_ascii=False))
    print(f"map: {len(ids)} products, runs: {len(runs)} queries -> {OUT}")


if __name__ == "__main__":
    main()
