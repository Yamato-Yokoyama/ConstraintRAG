"""Image -> text with BLIP (ViT encoder, text decoder). Then check captions against titles."""
import json

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.negation import has_unnegated
from src.vision.common import CAPTIONS_PATH, image_path, negated_products

MODEL = "Salesforce/blip-image-captioning-base"


def generate_captions(product_ids: list[str]) -> dict[str, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL)
    model = BlipForConditionalGeneration.from_pretrained(MODEL).to(device).eval()
    captions = {}
    for number, product_id in enumerate(product_ids, start=1):
        image = Image.open(image_path(product_id)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)
        captions[product_id] = processor.decode(output[0], skip_special_tokens=True)
        if number % 50 == 0:
            print(f"captioned {number}/{len(product_ids)}", flush=True)
    return captions


def main() -> None:
    table = negated_products()
    ids = [p for p in table["product_id"].unique() if image_path(p).exists()]
    captions = generate_captions(ids)
    CAPTIONS_PATH.parent.mkdir(exist_ok=True)
    with CAPTIONS_PATH.open("w") as file:
        for product_id, caption in captions.items():
            file.write(json.dumps({"product_id": product_id, "caption": caption}) + "\n")

    rows = table[table["product_id"].isin(captions)].copy()
    rows["caption"] = rows["product_id"].map(captions)
    rows["in_title"] = [has_unnegated(t, e) for t, e in zip(rows["title"], rows["excluded_term"])]
    rows["in_caption"] = [has_unnegated(c, e) for c, e in zip(rows["caption"], rows["excluded_term"])]
    both = int((rows["in_title"] & rows["in_caption"]).sum())
    print(json.dumps({
        "pairs": len(rows),
        "term_in_title": int(rows["in_title"].sum()),
        "term_in_caption": int(rows["in_caption"].sum()),
        "term_in_both": both,
    }, indent=2))


if __name__ == "__main__":
    main()
