# ConstraintRAG app (v0)

The v0 RAG baseline for negated product-search queries. See the top-level `README.md` for the design and results.

Started from the skeleton of langchain-rag-explorer (commit 84a200f) on 2026-11-05 (the original repository is not modified) and rebuilt for Amazon ESCI, which lives in `../data`. Prototyped with AI assistance (Claude).

## Run
- `python -m src.eval_retrieval` – v0 retrieval evaluation (no LLM) → `results/eval_v0.json`
- `python -m src.eval_constraint_layer [--llm]` – constraint layer; `--llm` parses constraints with Gemini (rate-limited on the free tier)
- `python -m src.ml_reranker`, `python -m src.ml_reranker_v2` – ML reranker experiments
- `chainlit run src/chainlit_app.py` – chat UI (runs the v0 pipeline live; inherited from the langchain-rag-explorer skeleton)
- `streamlit run viz/pipeline_walkthrough.py` – step-by-step walkthrough of the pipeline on saved results (`results/eval_v0.json`); no model or Gemini calls

Settings come from `.env` (see `.env.example`): `GEMINI_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_MODEL`, `ESCI_DATA_DIR`.
