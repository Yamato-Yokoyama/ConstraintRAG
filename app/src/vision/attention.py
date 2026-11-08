"""Show where CLIP's ViT looks: CLS attention in the last layer, overlaid on the image."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.vision.common import image_path

MODEL = "openai/clip-vit-base-patch32"
OUTPUT_DIR = Path("results/attention")


def cls_attention(model, processor, image: Image.Image) -> torch.Tensor:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.vision_model(pixel_values=inputs["pixel_values"], output_attentions=True)
    weights = output.attentions[-1][0].mean(0)[0, 1:]
    size = int(weights.numel() ** 0.5)
    return weights.reshape(size, size)


def main(product_ids: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processor = CLIPProcessor.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL, attn_implementation="eager").eval()
    captions = {}
    captions_path = Path("results/captions.jsonl")
    if captions_path.exists():
        captions = {r["product_id"]: r["caption"] for r in map(json.loads, captions_path.read_text().splitlines())}
    for product_id in product_ids:
        image = Image.open(image_path(product_id)).convert("RGB")
        grid = cls_attention(model, processor, image).numpy()
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.imshow(image)
        axis.imshow(grid, cmap="inferno", alpha=0.55, extent=(0, image.width, image.height, 0), interpolation="bilinear")
        axis.set_title(captions.get(product_id, product_id)[:50], fontsize=8)
        axis.axis("off")
        figure.savefig(OUTPUT_DIR / f"{product_id}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main(sys.argv[1:])
