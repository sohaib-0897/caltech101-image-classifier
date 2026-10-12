import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.data import image_transform, make_loader, open_rgb, read_split, split_hash
from src.models import SmallCNN, mobilenet
from src.train import validate


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    name, classes = checkpoint["model_name"], checkpoint["class_names"]
    if name == "cnn":
        model = SmallCNN(len(classes))
    elif name == "mobilenet":
        model = mobilenet(len(classes))
    else:
        raise ValueError(f"Unknown checkpoint model: {name}")
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def choose_threshold(labels, scores, target_accuracy=0.9, min_coverage=0.2):
    confidence, correct = scores.max(1), scores.argmax(1) == labels
    curve = []
    for threshold in np.linspace(0, 1, 101):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        curve.append({"threshold": float(threshold), "coverage": float(accepted.mean()), "accepted": count,
                      "accuracy": float(correct[accepted].mean()) if count else None})
    eligible = [row for row in curve if row["coverage"] >= min_coverage and row["accepted"] >= 30]
    feasible = [row for row in eligible if row["accuracy"] >= target_accuracy]
    if feasible:
        chosen = max(feasible, key=lambda row: (row["coverage"], -row["threshold"]))
    else:
        chosen = max(eligible, key=lambda row: (row["accuracy"], row["coverage"], -row["threshold"])) if eligible else curve[0]
    return {**chosen, "target_accuracy": target_accuracy, "min_coverage": min_coverage,
            "target_met": bool(feasible), "curve": curve}


def score_predictions(labels, scores, classes):
    predicted = scores.argmax(1)
    top = np.argsort(scores, axis=1)[:, -min(5, len(classes)):]
    report = classification_report(labels, predicted, labels=list(range(len(classes))), target_names=classes,
                                   output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
            "top5_accuracy": float((top == labels[:, None]).any(1).mean()),
            "images": len(labels), "per_class": {name: report[name] for name in classes}}


def confused_pairs(labels, scores, classes):
    directions = Counter((int(a), int(b)) for a, b in zip(labels, scores.argmax(1)) if a != b)
    pairs = Counter()
    for (a, b), count in directions.items():
        pairs[tuple(sorted((a, b)))] += count
    return [{"class_a": classes[a], "class_b": classes[b], "count": count,
             "a_as_b": directions[(a, b)], "b_as_a": directions[(b, a)]}
            for (a, b), count in pairs.most_common(10)]


def save_errors(output, root, rows, labels, scores, classes):
    predicted = scores.argmax(1)
    pairs = confused_pairs(labels, scores, classes)
    (output / "confused_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    if pairs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([f"{p['class_a']} / {p['class_b']}" for p in reversed(pairs)], [p["count"] for p in reversed(pairs)])
        ax.set_xlabel("Errors in either direction")
        fig.tight_layout()
        fig.savefig(output / "confused_pairs.png", dpi=150)
        plt.close(fig)
    wrong = np.flatnonzero(predicted != labels)
    wrong = sorted(wrong, key=lambda i: -scores[i].max())[:12]
    if wrong:
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, wrong):
            ax.imshow(open_rgb(root / rows[i]["path"]))
            ax.set_title(f"true: {classes[labels[i]]}\npred: {classes[predicted[i]]}\np={scores[i].max():.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / "misclassified.png", dpi=130)
        plt.close(fig)
    with (output / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "true_label", "predicted_label", "confidence"])
        for row, actual, guess, score in zip(rows, labels, predicted, scores.max(1)):
            writer.writerow([row["path"], classes[actual], classes[guess], float(score)])
    return pairs


def benchmark(model, root, rows, config, repetitions=100):
    transform = image_transform(config)
    images = [root / row["path"] for row in rows[:16]]
    tensor = transform(open_rgb(images[0])).unsqueeze(0)
    model_times, full_times = [], []
    with torch.inference_mode():
        for _ in range(20):
            model(tensor)
        for i in range(repetitions):
            start = time.perf_counter()
            model(tensor).softmax(1)
            model_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            sample = transform(open_rgb(images[i % len(images)])).unsqueeze(0)
            model(sample).softmax(1)
            full_times.append((time.perf_counter() - start) * 1000)
    return {"threads": torch.get_num_threads(), "batch_size": 1, "warmup": 20, "repetitions": repetitions,
            "source": "validation images; warm model; no HTTP overhead",
            "model_ms_median": float(np.median(model_times)), "model_ms_p95": float(np.percentile(model_times, 95)),
            "decode_transform_model_ms_median": float(np.median(full_times)),
            "decode_transform_model_ms_p95": float(np.percentile(full_times, 95))}


def evaluate(args):
    lock = args.output / "test_evaluation.json"
    if lock.exists():
        raise ValueError("Test evaluation already started. Read the saved results; do not use test scores for tuning.")
    torch.set_num_threads(1)
    split_file = args.output / "splits.csv"
    info = json.loads((args.output / "splits.json").read_text())
    digest = split_hash(split_file)
    if digest != info["split_sha256"]:
        raise ValueError("Split CSV changed after creation")
    selection = json.loads((args.output / "selection.json").read_text())
    path = args.checkpoints / f"{selection['selected_run']}.pt"
    model, checkpoint = read_checkpoint(path)
    if checkpoint["split_sha256"] != digest:
        raise ValueError("Checkpoint belongs to another split")
    val_rows = read_split(split_file, "val")
    val_loader = make_loader(args.root, val_rows, checkpoint["preprocessing"], args.batch_size)
    _, labels, scores = validate(model, val_loader, "cpu")
    threshold = choose_threshold(labels, scores)
    checkpoint["confidence_threshold"] = threshold["threshold"]
    checkpoint["threshold_selection"] = {key: value for key, value in threshold.items() if key != "curve"}
    final_path = args.checkpoints / "final.pt"
    torch.save(checkpoint, final_path)
    metadata = {key: checkpoint[key] for key in ("model_name", "class_names", "preprocessing", "dataset", "seed", "run", "training", "confidence_threshold")}
    metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    (args.output / "final_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "confidence_threshold.json").write_text(json.dumps(threshold, indent=2) + "\n")
    models = {"final": (model, checkpoint, final_path)}
    if args.baseline:
        baseline_path = args.checkpoints / args.baseline
        baseline, baseline_info = read_checkpoint(baseline_path)
        if baseline_info["preprocessing"] != checkpoint["preprocessing"] or baseline_info["split_sha256"] != digest:
            raise ValueError("Baseline preprocessing or split differs")
        models["baseline"] = (baseline, baseline_info, baseline_path)
    audit = {"status": "started", "started_at": datetime.now(timezone.utc).isoformat(), "selected_run": checkpoint["run"],
             "split_sha256": digest, "final_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
             "selection_metric": "validation macro-F1", "models": list(models)}
    lock.write_text(json.dumps(audit, indent=2) + "\n")
    try:
        test_rows = read_split(split_file, "test")
        loader = make_loader(args.root, test_rows, checkpoint["preprocessing"], args.batch_size)
        labels, collected = [], {name: [] for name in models}
        with torch.inference_mode():
            for images, targets in loader:
                labels.extend(targets.tolist())
                for name, (candidate, _, _) in models.items():
                    collected[name].append(candidate(images).softmax(1).numpy())
        labels = np.array(labels)
        results = {}
        for name, (candidate, metadata, candidate_path) in models.items():
            probabilities = np.concatenate(collected[name])
            metrics = score_predictions(labels, probabilities, checkpoint["class_names"])
            metrics.update(run=metadata["run"], model=metadata["model_name"],
                           model_file_bytes=candidate_path.stat().st_size,
                           latency=benchmark(candidate, args.root, val_rows, metadata["preprocessing"]))
            results[name] = metrics
            with (args.output / f"{name}_per_class.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["class", "precision", "recall", "f1", "support"])
                for label, values in metrics["per_class"].items():
                    writer.writerow([label, values["precision"], values["recall"], values["f1-score"], int(values["support"])])
        final_scores = np.concatenate(collected["final"])
        accepted = final_scores.max(1) >= threshold["threshold"]
        results["final"]["selective_prediction"] = {
            "threshold": threshold["threshold"], "coverage": float(accepted.mean()), "accepted": int(accepted.sum()),
            "accuracy": float((final_scores.argmax(1)[accepted] == labels[accepted]).mean()) if accepted.any() else None}
        results["final"]["confused_pairs"] = save_errors(args.output, args.root, test_rows, labels, final_scores, checkpoint["class_names"])
        (args.output / "test_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        audit.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        print(json.dumps({name: {k: m[k] for k in ("run", "accuracy", "macro_f1", "top5_accuracy")} for name, m in results.items()}, indent=2))
    except Exception as exc:
        audit.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--baseline", default="baseline.pt", help="empty string skips baseline evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
