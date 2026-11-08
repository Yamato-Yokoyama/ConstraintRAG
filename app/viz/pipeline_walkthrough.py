# ✍️ Claude が作成（2026-11-07、本人依頼の叩き台）
# 実行：app フォルダで `streamlit run viz/pipeline_walkthrough.py`
# 検索結果は results/eval_v0.json の保存済みデータ。モデルと Gemini は呼ばない。
import json
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from src.negation import extract_excluded_term, has_unnegated, is_negated  # noqa: E402

QUERY_IDS = {
    "否定：cat treats without chicken": 23909,
    "否定（語の取り出しがずれる例）：sweatshirt without hood for men": 99719,
    "普通：small dog brush for yorkie": 94688,
}
STEPS = [
    ("A", "商品を選ぶ"),
    ("A", "文書にする"),
    ("A", "ベクトル化"),
    ("A", "Chroma に保存"),
    ("B", "質問を整える"),
    ("B", "否定を調べる"),
    ("B", "検索"),
    ("B", "回答を作る"),
    ("B", "根拠と照らす"),
    ("B", "履歴に残す"),
]


@st.cache_data
def load_queries() -> dict:
    data = json.loads((APP_DIR / "results" / "eval_v0.json").read_text())
    return {q["query_id"]: q for q in data["queries"]}


def go_to(index: int) -> None:
    st.session_state.step = index


def draw_stage(stage: str, title: str) -> None:
    st.caption(title)
    indexes = [i for i, step in enumerate(STEPS) if step[0] == stage]
    for column, i in zip(st.columns(len(indexes)), indexes):
        kind = "primary" if i == st.session_state.step else "secondary"
        column.button(STEPS[i][1], key=f"step{i}", type=kind, on_click=go_to, args=(i,))


def show_step(index: int, q: dict) -> None:
    text = q["query"]
    term = q.get("excluded_term")
    app_term = extract_excluded_term(text)
    titles = q["top10"]

    if index == 0:
        st.write("ESCI の商品を 5,992 件選びます。質問そのものは保存しません。")
        st.table({"中身": ["評価40クエリの採点済み商品", "ランダムに足した商品"], "件数": ["約1,000", "5,000"]})
        st.info("40件の質問は保存するものではなく、あとで検索を測るためのテストです。")
    elif index == 1:
        st.write("タイトルと説明文をつなぎ、1500文字までで切ります。分割はしません。")
        st.code(f"{titles[0]}\n（続けて説明文。全体で 1500 文字まで）", language=None)
        st.info("1商品＝1文書。分割した場合との比較はまだしていません。")
    elif index == 2:
        st.write("文書を bge-m3 でベクトルにします。意味の近いものが近い位置に来ます。")
        st.code("商品の文書  →  bge-m3  →  [数値の列]", language=None)
    elif index == 3:
        st.write("ベクトルと文書、タイトルを Chroma に保存します。2回目からは保存分を使います。")
        st.code("collection: constraintrag_v0 / 距離: cosine / 件数: 5,992", language=None)
    elif index == 4:
        st.code(f"入力：{text}", language=None)
        st.write("Gemini が「それ」「さっきの」などを具体的な商品名に置き換えます。")
        st.success("会話の履歴がなければ Gemini を呼ばず、質問をそのまま使います。")
        st.info("リライトの役割だけです。学校か LinkedIn かのようなルーティングはありません。")
    elif index == 5:
        if is_negated(text):
            st.error(f"🚨 否定を発見　アプリが取り出した除外語：{app_term}")
            if term and app_term != term:
                st.warning(f"評価で人が付けた除外語は「{term}」です。このずれで trap を見逃すことがあります。")
        else:
            st.success("否定なし。除外語はありません。")
        st.write("正規表現で調べます。ここでは Gemini を呼びません。")
    elif index == 6:
        st.write("質問をベクトルにし、Chroma で類似度の高い上位10件を取ります。")
        negated = bool(term)
        traps = 0
        for rank, title in enumerate(titles, start=1):
            trap = negated and has_unnegated(title, term)
            traps += trap
            st.write(f"{'🟥' if trap else '⬜'} {rank}. {title}")
        if negated:
            st.metric("trap@10（評価で人が付けた除外語）", f"{traps / 10:.1f}")
        st.metric("nDCG@10", f"{q['ndcg@10']:.3f}")
        st.caption("🟥 は印だけで、上位から外しません。類似度の数値は保存していないため出していません。")
    elif index == 7:
        st.write("Gemini が上位10件だけを根拠に商品をすすめます。")
        if term:
            st.warning("🟥 の商品も根拠として Gemini に渡ります。")
        st.caption("回答は保存していないため、未表示です。")
    elif index == 8:
        st.write("Gemini が回答と根拠を見比べ、裏付けがなければ「⚠️ Check」を先頭に付けます。")
        st.info("見ているのは根拠との一致です。否定を守ったかは専用に調べていません。")
    elif index == 9:
        st.write("質問と回答を履歴に残します。次の質問の「質問を整える」で使います。")


st.set_page_config(page_title="ConstraintRAG pipeline", layout="wide")
st.title("ConstraintRAG v0：パイプラインの流れ")
st.session_state.setdefault("step", 0)

label = st.selectbox("質問", QUERY_IDS)
query = load_queries()[QUERY_IDS[label]]

draw_stage("A", "A. 索引を作る（最初の1回）")
draw_stage("B", "B. 質問が来るたび")
st.divider()

current = st.session_state.step
st.subheader(f"{current + 1}. {STEPS[current][1]}")
show_step(current, query)
st.divider()

back, forward = st.columns(2)
back.button("← 戻る", on_click=go_to, args=(current - 1,), disabled=current == 0)
forward.button("次へ →", on_click=go_to, args=(current + 1,), disabled=current == len(STEPS) - 1)
