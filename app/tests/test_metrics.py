"""nDCG と Recall の仕様。値は手計算(ML 塾 A1 の例題と同じ)。"""

import math

import pytest

from src.metrics import dcg, ndcg, recall


def test_dcg_by_hand():
    # 1/log2(2) + 0/log2(3) + 0.1/log2(4) = 1 + 0 + 0.05
    assert dcg([1.0, 0.0, 0.1], k=3) == pytest.approx(1.05)


def test_ndcg_by_hand():
    expected = (1 / math.log2(3) + 0.05) / (1 + 0.1 / math.log2(3))
    assert ndcg([0.0, 1.0, 0.1], k=3) == pytest.approx(expected)


def test_ndcg_uses_given_ideal():
    # 正解が2つあるのに1つしか取れていない。理想は正解2つから作る
    assert ndcg([1.0, 0.0], k=2, ideal_gains=[1.0, 1.0]) == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_ndcg_no_relevant_is_zero():
    assert ndcg([0.0, 0.0], k=2) == 0.0


def test_recall():
    assert recall(["a", "x", "b"], {"a", "b", "c", "d"}, k=2) == 0.25
    assert recall(["a"], set(), k=10) == 0.0
