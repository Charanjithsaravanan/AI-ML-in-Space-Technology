"""Single-patch inference.

    python infer.py --image samples/tile.png --checkpoint runs/best.pt

Prints the predicted class, a cloud flag, a sensor-corruption flag and the
measured latency for one 64x64 tile - the same three outputs the on-board
product would downlink alongside the image.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import cloud
from model import NormalizedModel, TinySatCNN
from noise import detect_corruption


def load_patch(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (64, 64):
        print("[infer] resizing " + str(image.size[0]) + "x" + str(image.size[1]) + " -> 64x64")
        image = image.resize((64, 64), Image.BICUBIC)
    return transforms.ToTensor()(image).unsqueeze(0)


def measure_infer_latency(model, patch: torch.Tensor, repeats: int = 100, warmup: int = 20) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            model(patch)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(patch)
            samples.append((time.perf_counter() - started) * 1000.0)
    samples = np.asarray(samples)
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one 64x64 satellite patch.")
    parser.add_argument("--image", required=True, help="path to a 64x64 RGB patch")
    parser.add_argument("--checkpoint", type=str, default="runs/best.pt")
    parser.add_argument("--scripted", action="store_true",
                        help="load runs/best_torchscript.pt instead of the state dict")
    parser.add_argument("--screener", type=str, default="runs/cloud_screener.joblib")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]

    if args.scripted:
        model = torch.jit.load("runs/best_torchscript.pt", map_location="cpu").eval()
    else:
        raw = TinySatCNN(num_classes=len(classes))
        raw.load_state_dict(checkpoint["state_dict"])
        raw.eval()
        model = NormalizedModel(raw, checkpoint["mean"], checkpoint["std"]).eval()

    patch = load_patch(args.image)

    with torch.no_grad():
        probabilities = model(patch).softmax(dim=1)[0]

    confidence, index = probabilities.max(dim=0)
    label = classes[int(index)]

    sensor = detect_corruption(patch)
    screener = cloud.load_screener(args.screener)
    if screener is not None:
        result = cloud.screen(patch, screener)
        cloud_line = ("CLOUDY" if result["cloudy"] else "CLEAR") + "  (p_cloud " + format(result["cloud_score"], ".3f") + ")"
    else:
        brightness = float(patch.mean())
        cloud_line = (
            ("CLOUDY" if brightness > 0.62 else "CLEAR")
            + "  (heuristic brightness " + format(brightness, ".3f") + " - run cloud.py to train the screener)"
        )

    latency = measure_infer_latency(model, patch)

    top3 = ", ".join(
        classes[int(i)] + " " + format(float(probabilities[int(i)]), ".3f")
        for i in torch.topk(probabilities, 3).indices
    )

    print("")
    print("image        " + Path(args.image).name)
    print("prediction   " + label + "  (p=" + format(float(confidence), ".4f") + ")")
    print("top-3        " + top3)
    print("cloud flag   " + cloud_line)
    sensor_line = "DEGRADED" if sensor["corrupted"] else "CLEAN"
    if sensor["likely_mode"]:
        sensor_line += "  (likely " + sensor["likely_mode"] + ")"
    print(
        "sensor flag  " + sensor_line
        + "  (sigma " + format(sensor["sigma"], ".4f")
        + ", dropout " + format(sensor["dropped_columns"], ".3f")
        + ", saturated " + format(sensor["saturated_fraction"], ".3f") + ")"
    )
    print(
        "latency      " + format(latency["median_ms"], ".3f") + " ms/patch median"
        + " (p95 " + format(latency["p95_ms"], ".3f") + " ms, CPU, batch 1)"
    )


if __name__ == "__main__":
    main()
