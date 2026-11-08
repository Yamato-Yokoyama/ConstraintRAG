"""Shared paths and product lists for the image experiment (negated queries only)."""
import os
from pathlib import Path

import pandas as pd

from src.load_esci import DATA_DIR, clean_text, load_eval_queries, load_examples

IMAGE_DIR = DATA_DIR / "sqid" / "images"
CAPTIONS_PATH = Path("results/captions.jsonl")


def image_table() -> pd.DataFrame:
    tables = [pd.read_parquet(DATA_DIR / "sqid" / name) for name in ("product_image_urls.parquet", "supp_product_image_urls.parquet")]
    return pd.concat(tables).dropna(subset=["image_url"]).drop_duplicates("product_id")


def negated_products() -> pd.DataFrame:
    """One row per (query, product) for the negated queries, with title and excluded term."""
    queries = load_eval_queries()["negated"]
    terms = pd.DataFrame(queries, columns=["query_id", "excluded_term"])
    examples = load_examples().merge(terms, on="query_id")
    products = pd.read_parquet(DATA_DIR / "products_us.parquet", columns=["product_id", "product_title"]).drop_duplicates("product_id")
    merged = examples.merge(products, on="product_id", how="left")
    merged["title"] = merged["product_title"].map(clean_text)
    return merged


def image_path(product_id: str) -> Path:
    return IMAGE_DIR / f"{product_id}.jpg"
