# ConstraintRAG

Measuring (and later fixing) how retrieval handles **negated and constrained queries** in product search — e.g. "shampoo without sulfate".

Status: **v1 (baseline evaluation)** — work in progress, started Oct 2026. This is a personal learning project; the analysis code in the notebooks is written by me.

## v1 so far
- Data: Amazon Shopping Queries Dataset (ESCI), US locale
- Retrievers: BM25, dense embeddings (all-MiniLM)
- Metrics: nDCG@10, top-10 trap rate (negated items that should have been excluded), split into candidate-pool vs. ranking errors
- Significance tests: Mann-Whitney U, Wilcoxon signed-rank

## Layout
- `notebooks/01_eda.ipynb` – exploring negated queries in ESCI
- `notebooks/02_baseline.ipynb` – BM25 / embedding baselines
- `notebooks/03_eval.ipynb` – evaluation and significance tests
- `scripts/download_esci.py` – downloads ESCI into `data/` (not included in this repo)

## Data and license
The ESCI data is **not redistributed here**. Get it with `python scripts/download_esci.py`.
Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Apache-2.0). Used for personal study only.
