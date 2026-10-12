import argparse
import csv
import hashlib
import json
import random
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Caltech101, OxfordIIITPet


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_dataset(root, name="auto"):
    root = Path(root)
    socket.setdefaulttimeout(60)
    failure = None
    if name != "pets":
        try:
            if not (root / "caltech101/101_ObjectCategories").exists():
                # gdown has no request timeout; bound the whole download instead.
                subprocess.run([sys.executable, "-c",
                                "import sys, socket; socket.setdefaulttimeout(60); from torchvision.datasets import Caltech101; "
                                "Caltech101(sys.argv[1], target_type='category', download=True)", str(root)],
                               timeout=600, check=True)
            dataset = Caltech101(root=str(root), target_type="category", download=False)
            rows = [
                {"path": (Path("caltech101/101_ObjectCategories") / dataset.categories[label]
                          / f"image_{index:04d}.jpg").as_posix(),
                 "label": label, "class_name": dataset.categories[label]}
                for index, label in zip(dataset.index, dataset.y)
            ]
            return rows, dataset.categories, {"name": "caltech101", "download_failure": None}
        except Exception as exc:
            if name == "caltech101":
                raise
            failure = f"{type(exc).__name__}: {exc}"
            print(f"Caltech-101 download failed: {failure}\nUsing Oxford-IIIT Pet instead.", flush=True)
    dataset = OxfordIIITPet(root=str(root), split="trainval", target_types="category", download=True)
    other = OxfordIIITPet(root=str(root), split="test", target_types="category", download=True)
    rows = [
        {"path": Path(path).relative_to(root).as_posix(), "label": int(label), "class_name": dataset.classes[label]}
        for path, label in zip(dataset._images + other._images, dataset._labels + other._labels)
    ]
    return rows, dataset.classes, {"name": "pets", "download_failure": failure}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def preprocessing(size=128):
    return {"image_size": size, "mean": MEAN, "std": STD, "resize": "square_bilinear", "color_mode": "RGB"}


def image_transform(config, train=False):
    size = config["image_size"]
    if train:
        steps = [transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.8, 1.25), antialias=True),
                 transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
    else:
        steps = [transforms.Resize((size, size), antialias=True)]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])


def open_rgb(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def read_split(path, split):
    with Path(path).open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows for {split} in {path}")
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def split_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


class ImageDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["path"])), row["label"]


def make_loader(root, rows, config, batch_size, train=False, seed=42, workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ImageDataset(root, rows, image_transform(config, train)), batch_size=batch_size,
                      shuffle=train, num_workers=workers, worker_init_fn=seed_worker, generator=generator,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def make_split(output, seed):
    report = json.loads((output / "exploration.json").read_text())
    with (output / "image_inventory.csv").open(newline="") as file:
        inventory = list(csv.DictReader(file))
    conflicts = {r["path"] for group in report["conflicting_label_groups"] for r in group}
    seen, rows, excluded = set(), [], []
    for row in sorted(inventory, key=lambda r: r["path"]):
        if row["path"] in conflicts or row["pixel_hash"] in seen:
            excluded.append(row["path"])
            continue
        seen.add(row["pixel_hash"])
        rows.append({"path": row["path"], "label": int(row["label"]), "class_name": row["class_name"]})
    indices = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    train, held = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
    val, test = train_test_split(held, test_size=0.5, stratify=[labels[i] for i in held], random_state=seed)
    result = []
    for name, group in (("train", train), ("val", val), ("test", test)):
        result.extend({**rows[i], "split": name} for i in sorted(group))
    path = output / "splits.csv"
    if path.exists():
        with path.open(newline="") as file:
            previous = [{**r, "label": int(r["label"])} for r in csv.DictReader(file)]
        if previous != result:
            raise ValueError("Existing split differs. Use another output directory for a new experiment.")
    else:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "label", "class_name", "split"])
            writer.writeheader()
            writer.writerows(result)
    counts = {name: dict(Counter(r["class_name"] for r in result if r["split"] == name)) for name in ("train", "val", "test")}
    if any(len(c) != len(report["class_names"]) for c in counts.values()):
        raise ValueError("Every class must occur in every split")
    info = {"dataset": report["name"], "seed": seed, "class_names": report["class_names"], "split_sha256": split_hash(path),
            "split_hash_policy": "UTF-8 CSV with normalized line endings",
            "images": len(rows), "excluded_duplicates": excluded, "excluded_corrupt": report["corrupt_files"],
            "sizes": {name: sum(c.values()) for name, c in counts.items()}, "class_counts": counts}
    (output / "splits.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"dataset": info["dataset"], "sizes": info["sizes"], "excluded_duplicates": len(excluded)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_split(args.output, args.seed)
