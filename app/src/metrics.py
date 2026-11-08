"""
src/metrics.py

検索の評価指標(nDCG@k、Recall@k)。ML 塾(jp-product-search-eval)の metrics.py と同じ式。

Called by:
    - src/eval_retrieval.py
    - tests/test_metrics.py
"""

import math


def dcg(gains: list[float], k: int) -> float:
    """DCG@k = Σ_{i=1..k} gain_i / log2(i + 1)(i は 1 始まりの順位)"""
    total = 0.0
    top_k = gains[:k]
    for i, gain in enumerate(top_k):
        # enumerate は 0 始まりなので、順位 = i + 1、割る数 = log2(順位 + 1) = log2(i + 2)
        total = total + gain / math.log2(i + 2)
    return total


def ndcg(ranked_gains: list[float], k: int = 10, ideal_gains: list[float] | None = None) -> float:
    """nDCG@k = DCG@k(システムの並び) / DCG@k(理想の並び)。

    ranked_gains: システムが並べた順の gain
    ideal_gains: そのクエリの正解ラベル全部の gain(並び順は問わない)。
        省略すると ranked_gains を並べ替えて理想とみなす。
        検索で取りこぼした正解がある場合はそれも理想に入れないといけないので、評価では必ず渡す。
    理想の DCG が 0 なら 0.0 を返す。
    """
    if ideal_gains is not None:
        ideal = sorted(ideal_gains, reverse=True)
    else:
        ideal = sorted(ranked_gains, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(ranked_gains, k) / idcg


def recall(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k = 上位 k 件に入った正解の数 / 正解の数。正解が無ければ 0.0。"""
    if not relevant_ids:
        return 0.0
    found = 0
    for rid in set(retrieved_ids[:k]):
        if rid in relevant_ids:
            found = found + 1
    return found / len(relevant_ids)
