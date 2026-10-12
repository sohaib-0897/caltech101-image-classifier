import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.data import make_loader, preprocessing, read_split, seed_everything, split_hash
from src.models import SmallCNN, mobilenet, unfreeze_features


RUN_FIELDS = ["run", "model", "seed", "lr", "batch_size", "unfreeze", "head_epochs", "fine_epochs", "weighted",
              "best_epoch", "val_accuracy", "val_macro_f1", "seconds", "parameters", "device", "threads", "status",
              "error", "split_sha256", "started_at"]


def validate(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for images, targets in loader:
            scores.append(model(images.to(device)).softmax(1).cpu().numpy())
            labels.extend(targets.tolist())
    probabilities = np.concatenate(scores)
    predictions = probabilities.argmax(1)
    return {"accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0)}, np.array(labels), probabilities


def train_epoch(model, loader, optimizer, loss_fn, device, freeze_stats=False):
    model.train()
    if freeze_stats:
        model.features.eval()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def write_run(output, summary, history):
    path = output / "run_log.csv"
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: summary.get(key, "") for key in RUN_FIELDS})
    (output / "runs").mkdir(exist_ok=True)
    (output / "runs" / f"{summary['run']}.json").write_text(
        json.dumps({"summary": summary, "epochs": history}, indent=2) + "\n")


def run_training(args):
    if (args.output / "test_evaluation.json").exists():
        raise ValueError("Test results already exist. Use a fresh output directory for another experiment.")
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    path = args.output / "run_log.csv"
    if path.exists():
        with path.open(newline="") as file:
            if any(row["run"] == args.run for row in csv.DictReader(file)):
                raise ValueError(f"Run {args.run} already exists; choose another name")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    split_file = args.output / "splits.csv"
    split_info = json.loads((args.output / "splits.json").read_text())
    if split_hash(split_file) != split_info["split_sha256"]:
        raise ValueError("Split CSV changed after creation")
    config = preprocessing(args.image_size)
    train_rows, val_rows = read_split(split_file, "train"), read_split(split_file, "val")
    train_loader = make_loader(args.root, train_rows, config, args.batch_size, True, args.seed, args.workers)
    val_loader = make_loader(args.root, val_rows, config, args.batch_size, False, args.seed, args.workers)
    classes = split_info["class_names"]
    summary = {"run": args.run, "model": args.model, "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size,
               "unfreeze": args.unfreeze if args.model == "mobilenet" else 0,
               "head_epochs": args.head_epochs if args.model == "mobilenet" else 0,
               "fine_epochs": args.fine_epochs if args.model == "mobilenet" else args.epochs, "weighted": False,
               "device": str(device), "threads": args.threads, "split_sha256": split_info["split_sha256"],
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "failed", "error": ""}
    history, best_f1, started = [], -1.0, time.perf_counter()
    try:
        model = (SmallCNN(len(classes)) if args.model == "cnn" else mobilenet(len(classes), pretrained=True)).to(device)
        summary["parameters"] = sum(p.numel() for p in model.parameters())
        if args.model == "mobilenet":
            unfreeze_features(model, 0)
            optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        epochs = args.epochs if args.model == "cnn" else args.head_epochs + args.fine_epochs
        for epoch in range(1, epochs + 1):
            phase = "scratch" if args.model == "cnn" else "head" if epoch <= args.head_epochs else "fine"
            if args.model == "mobilenet" and epoch == args.head_epochs + 1:
                unfreeze_features(model, args.unfreeze)
                groups = [{"params": model.classifier.parameters(), "lr": args.lr},
                          {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": args.lr / 10}]
                optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            epoch_start = time.perf_counter()
            loss = train_epoch(model, train_loader, optimizer, loss_fn, device, args.model == "mobilenet")
            metrics, _, _ = validate(model, val_loader, device)
            entry = {"epoch": epoch, "phase": phase, "train_loss": loss, "val_accuracy": metrics["accuracy"],
                     "val_macro_f1": metrics["macro_f1"], "seconds": time.perf_counter() - epoch_start}
            history.append(entry)
            print(f"{args.run} epoch {epoch}/{epochs}: loss {loss:.4f}, val f1 {metrics['macro_f1']:.4f}, {entry['seconds']:.1f}s", flush=True)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                summary.update(best_epoch=epoch, val_accuracy=metrics["accuracy"], val_macro_f1=best_f1)
                checkpoint = {"model_name": args.model, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "class_names": classes, "preprocessing": config, "dataset": split_info["dataset"],
                              "split_sha256": split_info["split_sha256"], "seed": args.seed, "run": args.run,
                              "training": vars(args) | {"root": str(args.root), "output": str(args.output), "checkpoints": str(args.checkpoints)},
                              "val_metrics": metrics, "confidence_threshold": None}
                torch.save(checkpoint, args.checkpoints / f"{args.run}.pt")
        summary["status"] = "complete"
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["seconds"] = time.perf_counter() - started
        summary["python"] = platform.python_version()
        summary["torch"] = torch.__version__
        summary["processor"] = platform.processor()
        write_run(args.output, summary, history)
    return summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-epochs", type=int, default=3)
    parser.add_argument("--unfreeze", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.head_epochs, args.fine_epochs, args.batch_size, args.threads) < 1 or args.unfreeze < 0:
        parser.error("epochs, batch size and threads must be positive")
    run_training(args)
