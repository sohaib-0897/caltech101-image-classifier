import argparse
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.data import image_transform, open_rgb
from src.models import SmallCNN, mobilenet


@lru_cache(maxsize=1)
def load_model(checkpoint=Path("checkpoints/final.pt")):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = saved["class_names"]
    if saved["model_name"] == "cnn":
        model = SmallCNN(len(classes))
    elif saved["model_name"] == "mobilenet":
        model = mobilenet(len(classes))
    else:
        raise ValueError("Unknown model in checkpoint")
    model.load_state_dict(saved["state_dict"])
    threshold = saved.get("confidence_threshold")
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("Checkpoint needs a validation confidence threshold; run src.evaluate first")
    return {"model": model.eval(), "class_names": classes, "transform": image_transform(saved["preprocessing"]),
            "threshold": float(threshold), "run": saved["run"], "dataset": saved["dataset"]}


def predict(image, loaded=None, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    loaded = load_model() if loaded is None else loaded
    try:
        if isinstance(image, Image.Image):
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        else:
            rgb = open_rgb(BytesIO(image) if isinstance(image, bytes) else image)
        tensor = loaded["transform"](rgb).unsqueeze(0)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read image; supply a readable image file") from exc
    with torch.inference_mode():
        probabilities = loaded["model"](tensor).softmax(1)[0]
    values, indices = probabilities.topk(min(top_k, len(loaded["class_names"])))
    predictions = [{"label": loaded["class_names"][index], "probability": float(value)} for index, value in zip(indices.tolist(), values.tolist())]
    uncertain = predictions[0]["probability"] < loaded["threshold"]
    return {"label": "uncertain" if uncertain else predictions[0]["label"], "uncertain": uncertain,
            "threshold": loaded["threshold"], "predictions": predictions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final.pt"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        print(json.dumps(predict(args.image, load_model(args.checkpoint), args.top_k), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
