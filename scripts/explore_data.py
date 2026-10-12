import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import read_dataset


def explore(root, output, name):
    output.mkdir(parents=True, exist_ok=True)
    rows, classes, info = read_dataset(root, name)
    download_record = output / "dataset_download.json"
    if info["name"] == "pets" and info["download_failure"] is None and download_record.exists():
        info["download_failure"] = json.loads(download_record.read_text()).get("caltech_failure")
    counts = Counter(row["class_name"] for row in rows)
    modes = Counter()
    sizes, corrupt, inventory = [], [], []
    hashes = defaultdict(list)
    for row in rows:
        path = root / row["path"]
        try:
            with Image.open(path) as image:
                image.load()
                modes[image.mode] += 1
                width, height = image.size
                rgb = image.convert("RGB")
                digest = hashlib.sha256(f"{width},{height}:".encode() + rgb.tobytes()).hexdigest()
                sizes.append((width, height))
            inventory.append({**row, "width": width, "height": height, "mode": image.mode, "pixel_hash": digest})
            hashes[digest].append(row)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            corrupt.append({**row, "error": str(exc)})
    duplicates = [group for group in hashes.values() if len(group) > 1]
    label_conflicts = [group for group in duplicates if len({r["label"] for r in group}) > 1]
    values = list(counts.values())
    widths, heights = zip(*sizes)
    report = {
        **info, "images": len(rows), "readable_images": len(inventory), "classes": len(classes),
        "class_names": classes, "label_format": "zero-based integer category ID; sorted torchvision class names",
        "class_counts": dict(sorted(counts.items())), "count_min": min(values), "count_max": max(values),
        "count_median": statistics.median(values), "imbalance_ratio": max(values) / min(values),
        "width_min": min(widths), "width_max": max(widths), "width_median": statistics.median(widths),
        "height_min": min(heights), "height_max": max(heights), "height_median": statistics.median(heights),
        "image_modes": dict(modes), "grayscale_files": sum(v for k, v in modes.items() if k in ("1", "L", "I", "F")),
        "corrupt_files": corrupt, "exact_duplicate_groups": duplicates, "conflicting_label_groups": label_conflicts,
        "duplicate_check": "SHA256 of decoded RGB pixels and dimensions; near-duplicates need manual review",
    }
    (output / "exploration.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "image_inventory.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    ordered = sorted(counts, key=counts.get, reverse=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(ordered)), [counts[c] for c in ordered])
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=6)
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, s=5, alpha=0.15)
    ax.set(xlabel="Width (pixels)", ylabel="Height (pixels)")
    fig.tight_layout()
    fig.savefig(output / "image_sizes.png", dpi=150)
    plt.close(fig)
    review_names = [c for c in classes if c.lower().replace(" ", "_") in ("faces", "faces_easy", "leopards", "cougar_body", "cougar_face",
                     "american_bulldog", "american_pit_bull_terrier", "staffordshire_bull_terrier", "birman", "ragdoll", "siamese")]
    review_rows = []
    for name in review_names:
        review_rows.extend([r for r in rows if r["class_name"] == name][:3])
    if review_rows:
        fig, axes = plt.subplots(len(review_names), 3, figsize=(9, 3 * len(review_names)), squeeze=False)
        for ax, row in zip(axes.flat, review_rows):
            with Image.open(root / row["path"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(row["class_name"])
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "label_review.png", dpi=120)
        plt.close(fig)
    print(json.dumps({k: report[k] for k in ("name", "images", "classes", "count_min", "count_max", "count_median", "imbalance_ratio", "image_modes")}, indent=2))
    print(f"Corrupt files: {len(corrupt)}; duplicate groups: {len(duplicates)}; conflicting labels: {len(label_conflicts)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset", choices=["auto", "caltech101", "pets"], default="auto")
    args = parser.parse_args()
    explore(args.root, args.output, args.dataset)
