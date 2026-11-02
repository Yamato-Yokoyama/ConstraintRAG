"""ESCI のデータを取得し、指定した言語の部分だけを data/ に保存する（一度だけ実行）。

    python scripts/download_esci.py            # jp と us
    python scripts/download_esci.py --locales jp

元データ: https://github.com/amazon-science/esci-data （Apache-2.0）
products の parquet は約 1.1GB ある。**一時的に約 1.3GB の空き容量が要る**。
言語ごとに、Task 1 で使う商品だけを残した小さいファイル（jp で約 130MB）を保存し、元のファイルは最後に消す（--keep-raw で残す）。
（このスクリプトは道具なので Claude が書いた。学ぶ対象は esci_eval/ の中）
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
FILES = ["shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet"]
DATA = Path(__file__).resolve().parent.parent / "data"


def download(name: str) -> Path:
    raw = DATA / "raw" / name
    if raw.exists():
        print(f"skip (exists): {raw}")
        return raw
    raw.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, raw)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locales", nargs="+", default=["jp", "us"])
    ap.add_argument("--keep-raw", action="store_true", help="元の parquet（約1.2GB）を消さずに残す")
    args = ap.parse_args()

    # data/ を外付け SSD へのリンクにしている場合は、リンク先の空き容量を見る
    DATA.mkdir(exist_ok=True)
    real = DATA.resolve()
    free_gb = shutil.disk_usage(real).free / 1e9
    print(f"data/ の実体: {real}（空き {free_gb:.1f}GB）")
    if not (DATA / "raw").exists() and free_gb < 1.5:
        raise SystemExit(f"空き容量が {free_gb:.1f}GB しかない。約 1.5GB 空けるか、data/ を外付け SSD へのリンクにする（README 参照）")

    ex_path, pr_path = (download(f) for f in FILES)
    for loc in args.locales:
        # Task 1 は small_version == 1 の部分（README の Table 1）
        ex = pd.read_parquet(ex_path, filters=[("product_locale", "==", loc), ("small_version", "==", 1)])
        pr = pd.read_parquet(pr_path, filters=[("product_locale", "==", loc)])
        pr = pr[pr["product_id"].isin(set(ex["product_id"]))]  # Task 1 で使う商品だけ残す
        ex.to_parquet(DATA / f"examples_{loc}.parquet", index=False)
        pr.to_parquet(DATA / f"products_{loc}.parquet", index=False)
        print(f"{loc}: {ex['query_id'].nunique()} queries, {len(ex)} judgements, {len(pr)} products")

    if not args.keep_raw:
        shutil.rmtree(DATA / "raw")
        print("removed data/raw/ (use --keep-raw to keep it)")


if __name__ == "__main__":
    main()
