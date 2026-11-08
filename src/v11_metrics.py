"""v1.1 の物差し。

✍️ このファイルの関数はすべて Claude が作成（2026-11-05 本人依頼・試し用）。本人は未レビュー。
学校用には後で本人が再現する。
考え方の出どころ：has_trap・trap_floor は 03_eval（本人）、修正方針はノート 16・17。

03_eval の関数はそのまま残し、ここには「直した判定」と「足した指標」だけを置く。
"""
import re

from sklearn.metrics import ndcg_score

# 否定の語の直後、最大2語あとまでに除外語があれば「否定されている」とみなす
# 例：without waste basket → basket は否定されている
NEGATOR_BEFORE = re.compile(
    # 間の語に with / and / or などが入ると否定はそこで切れる（Non Ticking with Snooze）
    r"\b(?:without|no|not|non|except|excluding|free\s+of)\b[\s-]+"
    r"(?:(?!(?:with|and|or|plus|but|includes?|including)\b)[\w'-]+\s+){0,2}$",
    re.IGNORECASE,
)
# 例：chicken-free / chicken free
FREE_AFTER = re.compile(r"^[\s-]*free\b", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]+>")
# 同じものの別の書き方（16 の D：waste basket）。語の頭だけでは拾えないものをここに足す
TRAP_SYNONYMS = {"basket": ["basket", "wastebasket"]}


def clean_text(text):
    """HTML タグを消す。欠損は空文字にする。"""
    if not isinstance(text, str):
        return ""
    return HTML_TAG.sub(" ", text)


def has_unnegated(text, trap_word):
    """text に trap_word が「否定されずに」1回でも出てくれば True。

    語の頭だけをそろえる（hood → hooded・hoodie は当たる／gel → angel は当たらない）。
    """
    words = TRAP_SYNONYMS.get(trap_word.lower(), [trap_word])
    alternatives = "|".join(re.escape(word) for word in words)
    pattern = re.compile(r"\b(?:" + alternatives + r")\w*", re.IGNORECASE)
    for match in pattern.finditer(text):
        negated = NEGATOR_BEFORE.search(text[: match.start()])
        free = FREE_AFTER.search(text[match.end() :])
        if not (negated or free):
            return True
    return False


def add_trap_columns(df_q, trap_word):
    """1クエリ分の候補に、罠の列を足した表を返す。

    has_trap_old  : 03 と同じ（タイトルに文字列があれば罠）
    has_trap      : タイトルに否定されていない除外語がある
    has_trap_body : 説明文か箇条書きに否定されていない除外語がある
    hidden_trap   : タイトルは問題なし、本文にだけ除外語がある
    has_body      : 説明文か箇条書きがある
    """
    df_q = df_q.copy()
    titles = df_q["product_title"].fillna("")
    bodies = (
        df_q["product_description"].map(clean_text)
        + " "
        + df_q["product_bullet_point"].map(clean_text)
    )
    df_q["has_trap_old"] = titles.str.contains(trap_word, case=False, regex=False).astype(int)
    df_q["has_trap"] = titles.map(lambda t: has_unnegated(t, trap_word)).astype(int)
    df_q["has_trap_body"] = bodies.map(lambda t: has_unnegated(t, trap_word)).astype(int)
    df_q["hidden_trap"] = ((df_q["has_trap"] == 0) & (df_q["has_trap_body"] == 1)).astype(int)
    df_q["has_body"] = (bodies.str.strip() != "").astype(int)
    return df_q


def rate_at_k(ranked, column, k=10):
    """上位 k 件のうち column が 1 の割合。"""
    return ranked.head(k)[column].sum() / min(k, len(ranked))


def trap_floor(df_q, column="has_trap", k=10):
    """どう並べても避けられない罠の割合（03 の trap_floor と同じ考え方）。"""
    slots = min(k, len(df_q))
    n_clean = (df_q[column] == 0).sum()
    return max(0, slots - n_clean) / slots


def ndcg_binary_at_k(ranked, k=10):
    """E を 1、それ以外を 0 にした nDCG@k。"""
    gain = (ranked["esci_label"] == "E").astype(float).values
    return ndcg_score([gain], [ranked["score"].fillna(0).values], k=k)
