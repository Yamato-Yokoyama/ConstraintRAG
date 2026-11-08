"""Download images for the negated-query products at a low rate. Images stay local."""
import time
import urllib.request

from src.vision.common import IMAGE_DIR, image_path, image_table, negated_products


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(negated_products()["product_id"])
    rows = image_table()
    rows = rows[rows["product_id"].isin(wanted)]
    failed = 0
    for product_id, url in zip(rows["product_id"], rows["image_url"]):
        path = image_path(product_id)
        if path.exists():
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "constraintrag-research"})
            path.write_bytes(urllib.request.urlopen(request, timeout=20).read())
        except Exception as error:
            failed += 1
            print(f"failed {product_id}: {error}", flush=True)
        time.sleep(0.5)
    print(f"wanted {len(wanted)}, with url {len(rows)}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
