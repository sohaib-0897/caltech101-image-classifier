import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.predict import load_model, predict


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.environ.get("INFERENCE_THREADS", "1")))
    path = Path(os.environ.get("MODEL_PATH", "checkpoints/final.pt"))
    app.state.loaded = load_model(path)
    yield


app = FastAPI(title="Image classifier", lifespan=lifespan)


@app.get("/health")
def health():
    loaded = app.state.loaded
    return {"status": "ok", "run": loaded["run"], "dataset": loaded["dataset"], "classes": len(loaded["class_names"])}


@app.post("/predict")
def predict_upload(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10)):
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MiB")
    if not content:
        raise HTTPException(400, "Image is empty")
    try:
        return predict(content, app.state.loaded, top_k)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
