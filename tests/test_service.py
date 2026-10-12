from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import app
from src.data import preprocessing
from src.models import SmallCNN
from src.predict import load_model, predict


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    path = tmp_path / "model.pt"
    torch.save({"model_name": "cnn", "class_names": ["cat", "dog"], "state_dict": SmallCNN(2).state_dict(),
                "preprocessing": preprocessing(), "confidence_threshold": 0.99, "run": "fixture", "dataset": "fixture"}, path)
    return path


@pytest.fixture
def client(checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    with TestClient(app.app) as client:
        yield client


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA"])
def test_image_upload_and_uncertainty(client, mode):
    image = BytesIO()
    Image.new(mode, (64, 80)).save(image, format="PNG")
    result = client.post("/predict?top_k=2", files={"file": ("image.png", image.getvalue(), "image/png")})
    assert result.status_code == 200
    data = result.json()
    assert data["label"] == "uncertain"
    assert data["uncertain"]
    values = [p["probability"] for p in data["predictions"]]
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(1.0)
    assert {p["label"] for p in data["predictions"]} == {"cat", "dog"}


def test_bad_uploads_return_clear_errors(client):
    assert client.post("/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}).status_code == 400
    assert client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/predict", files={"file": ("large.jpg", b"x" * (app.MAX_UPLOAD_BYTES + 1), "image/jpeg")}).status_code == 413
    assert client.post("/predict?top_k=0", files={"file": ("bad.jpg", b"bad", "image/jpeg")}).status_code == 422
    assert client.get("/health").json() == {"status": "ok", "run": "fixture", "dataset": "fixture", "classes": 2}


def test_predict_is_deterministic_and_handles_bad_paths(checkpoint, tmp_path):
    loaded = load_model(checkpoint)
    image = Image.new("L", (50, 40))
    assert predict(image, loaded) == predict(image, loaded)
    with pytest.raises(ValueError, match="Cannot read image"):
        predict(tmp_path / "missing.jpg", loaded)
