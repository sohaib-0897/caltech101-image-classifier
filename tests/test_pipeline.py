import csv
import json
from argparse import Namespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import image_transform, make_split, open_rgb, preprocessing, split_hash
from src.evaluate import choose_threshold, confused_pairs, read_checkpoint
from src.models import SmallCNN, mobilenet, unfreeze_features
from src.train import train_epoch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_split_keeps_duplicates_out_and_preserves_classes(tmp_path):
    rows = [{"path": f"{label}/{i}.jpg", "label": label, "class_name": str(label), "pixel_hash": f"{label}-{i}"}
            for label in range(2) for i in range(40)]
    rows += [{**rows[0], "path": "0/duplicate.jpg"},
             {"path": "0/conflict.jpg", "label": 0, "class_name": "0", "pixel_hash": "conflict"},
             {"path": "1/conflict.jpg", "label": 1, "class_name": "1", "pixel_hash": "conflict"}]
    with (tmp_path / "image_inventory.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"name": "fixture", "class_names": ["0", "1"], "corrupt_files": [],
              "conflicting_label_groups": [[rows[-2], rows[-1]]]}
    (tmp_path / "exploration.json").write_text(json.dumps(report))
    make_split(tmp_path, 42)
    before = split_hash(tmp_path / "splits.csv")
    make_split(tmp_path, 42)
    assert before == split_hash(tmp_path / "splits.csv")
    info = json.loads((tmp_path / "splits.json").read_text())
    assert info["sizes"] == {"train": 56, "val": 12, "test": 12}
    assert len(info["excluded_duplicates"]) == 3
    with (tmp_path / "splits.csv").open(newline="") as file:
        split = list(csv.DictReader(file))
    lookup = {row["path"]: row["pixel_hash"] for row in rows}
    groups = [{lookup[row["path"]] for row in split if row["split"] == name} for name in ("train", "val", "test")]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
    assert all(set(counts) == {"0", "1"} for counts in info["class_counts"].values())
    with pytest.raises(ValueError, match="Existing split differs"):
        make_split(tmp_path, 43)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_non_rgb_files_use_rgb_preprocessing(tmp_path, mode):
    path = tmp_path / "image.png"
    Image.new(mode, (40, 80)).save(path)
    image = open_rgb(path)
    assert image.mode == "RGB"
    tensor = image_transform(preprocessing())(image)
    assert tensor.shape == (3, 128, 128)
    assert torch.isfinite(tensor).all()


def test_threshold_uses_validation_coverage():
    labels = np.array([0] * 60 + [1] * 40)
    scores = np.array([[0.95, 0.05]] * 60 + [[0.6, 0.4]] * 40)
    result = choose_threshold(labels, scores)
    assert result["threshold"] == pytest.approx(0.61)
    assert result["coverage"] == 0.6
    assert result["accuracy"] == 1.0
    assert result["target_met"]
    failed = choose_threshold(np.ones(100, dtype=int), np.array([[0.7, 0.3]] * 100))
    assert not failed["target_met"]
    assert failed["threshold"] == 0
    labels = np.array([0] * 40 + [1] * 60)
    scores = np.array([[0.9, 0.1]] * 50 + [[0.6, 0.4]] * 50)
    fallback = choose_threshold(labels, scores)
    assert not fallback["target_met"]
    assert fallback["threshold"] == pytest.approx(0.61)
    assert fallback["coverage"] == 0.5
    assert fallback["accuracy"] == 0.8


def test_confusion_pairs_count_both_directions():
    scores = np.array([[0.1, 0.9, 0], [0.1, 0.9, 0], [0.9, 0.1, 0], [0, 0, 1]])
    pairs = confused_pairs(np.array([0, 0, 1, 2]), scores, ["a", "b", "c"])
    assert pairs == [{"class_a": "a", "class_b": "b", "count": 3, "a_as_b": 2, "b_as_a": 1}]


def test_checkpoint_loads_without_pretrained_download(tmp_path):
    model = SmallCNN(2).eval()
    path = tmp_path / "model.pt"
    torch.save({"model_name": "cnn", "class_names": ["a", "b"], "state_dict": model.state_dict()}, path)
    loaded, _ = read_checkpoint(path)
    sample = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        torch.testing.assert_close(model(sample), loaded(sample))


def test_fine_tuning_changes_only_requested_features():
    model = mobilenet(2)
    unfreeze_features(model, 1)
    assert not any(p.requires_grad for layer in model.features[:-2] for p in layer.parameters())
    assert all(p.requires_grad for layer in model.features[-2:] for p in layer.parameters())
    batchnorm = next(layer for layer in model.features.modules() if isinstance(layer, torch.nn.BatchNorm2d))
    before = batchnorm.running_mean.clone()
    frozen = next(model.features[0].parameters()).detach().clone()
    head = model.classifier[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    data = [(torch.rand(2, 3, 128, 128), torch.tensor([0, 1]))]
    train_epoch(model, data, optimizer, torch.nn.CrossEntropyLoss(), "cpu", freeze_stats=True)
    torch.testing.assert_close(before, batchnorm.running_mean)
    torch.testing.assert_close(frozen, next(model.features[0].parameters()))
    assert not torch.equal(head, model.classifier[-1].weight)


def test_split_checksum_survives_checkout_line_endings(tmp_path):
    first, second = tmp_path / "windows.csv", tmp_path / "linux.csv"
    first.write_bytes(b"path,label,split\r\na.jpg,0,train\r\n")
    second.write_bytes(b"path,label,split\na.jpg,0,train\n")
    assert split_hash(first) == split_hash(second)


def test_test_results_lock_training_and_evaluation(tmp_path):
    from src.evaluate import evaluate
    from src.train import run_training
    (tmp_path / "test_evaluation.json").write_text('{"status": "complete"}')
    args = Namespace(output=tmp_path)
    with pytest.raises(ValueError, match="Test evaluation already started"):
        evaluate(args)
    with pytest.raises(ValueError, match="Test results already exist"):
        run_training(args)
