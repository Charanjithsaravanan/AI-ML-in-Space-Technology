"""Read-only image analysis API for the GEOSAT frontend."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cloud
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model import GEOSATCNN, NormalizedModel
from noise import detect_corruption, inject_sensor_noise
from utils import get_device

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs" / "best.pt"
SCREENER = ROOT / "runs" / "cloud_screener.joblib"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEVICE = get_device()

if not CHECKPOINT.exists():
    raise RuntimeError("Missing model checkpoint: " + str(CHECKPOINT))

checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
CLASSES = checkpoint["classes"]
model = NormalizedModel(
    GEOSATCNN(num_classes=len(CLASSES)),
    checkpoint["mean"],
    checkpoint["std"],
).to(DEVICE).eval()
screener = cloud.load_screener(str(SCREENER))

app = FastAPI(title="GEOSAT Analysis API", version="1.0.0")
origins = [
    origin.strip()
    for origin in os.getenv("GEOSAT_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def load_image(contents: bytes) -> torch.Tensor:
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="Upload must be a readable RGB image") from exc
    image = image.resize((64, 64), Image.Resampling.BICUBIC)
    values = torch.from_numpy(np.array(image, copy=True)).float() / 255.0
    return values.permute(2, 0, 1).unsqueeze(0)


def probability_rows(probabilities: torch.Tensor, limit: int = 5) -> list[dict[str, Any]]:
    indices = torch.topk(probabilities, min(limit, len(CLASSES))).indices.tolist()
    return [{"class_name": CLASSES[index], "probability": round(float(probabilities[index]), 6)} for index in indices]


@torch.no_grad()
def predict(patch: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    probabilities = model(patch.to(DEVICE)).softmax(dim=1)[0].cpu()
    return probabilities, probability_rows(probabilities)


def cloud_result(patch: torch.Tensor) -> dict[str, Any]:
    if screener is not None:
        result = cloud.screen(patch, screener)
        return {**result, "source": "saved screener"}
    brightness = float(patch.mean())
    return {
        "cloud_score": round(min(brightness / 0.62, 1.0), 4),
        "cloudy": brightness > 0.62,
        "source": "brightness fallback",
    }


@torch.no_grad()
def robustness(patch: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for mode in ("gaussian", "salt_pepper", "band_dropout"):
        for severity in (0.05, 0.15, 0.30):
            degraded = inject_sensor_noise(patch, mode, severity)
            probabilities, _ = predict(degraded)
            index = int(probabilities.argmax())
            rows.append({
                "mode": mode,
                "severity": severity,
                "prediction": CLASSES[index],
                "confidence": round(float(probabilities[index]), 6),
            })
    return rows


@torch.no_grad()
def occlusion_heatmap(patch: torch.Tensor, target_index: int) -> list[list[float]]:
    """Measure target-confidence loss when each 8x8 patch is occluded."""
    baseline = float(predict(patch)[0][target_index])
    masks = []
    for row in range(8):
        for column in range(8):
            masked = patch.clone()
            masked[:, :, row * 8:(row + 1) * 8, column * 8:(column + 1) * 8] = 0.5
            masks.append(masked)
    scores = []
    for batch_start in range(0, len(masks), 16):
        batch = torch.cat(masks[batch_start:batch_start + 16])
        values = model(batch.to(DEVICE)).softmax(dim=1)[:, target_index].cpu()
        scores.extend((baseline - values).tolist())
    maximum = max(max(scores), 1e-6)
    return [[round(max(0.0, scores[row * 8 + column]) / maximum, 4) for column in range(8)] for row in range(8)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": str(DEVICE), "model": "best.pt"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    contents = await file.read()
    patch = load_image(contents)
    probabilities, top_predictions = predict(patch)
    target_index = int(probabilities.argmax())
    return {
        "filename": file.filename or "upload",
        "image_size": [64, 64],
        "device": str(DEVICE),
        "prediction": {
            "class_name": CLASSES[target_index],
            "confidence": round(float(probabilities[target_index]), 6),
            "top_predictions": top_predictions,
        },
        "cloud": cloud_result(patch),
        "sensor": detect_corruption(patch),
        "robustness": robustness(patch),
        "heatmap": occlusion_heatmap(patch, target_index),
    }
