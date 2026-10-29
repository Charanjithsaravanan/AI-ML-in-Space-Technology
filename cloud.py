"""Cloud / clean screening, and why the signal is derived from EuroSAT here.

CloudSEN12 (Hugging Face) is the reference dataset for this objective. It is not
fetched automatically: the full release is tens of GB and its download requires
an authenticated Hugging Face session, which a CubeSat ground segment rarely has.

Two paths, both implemented:

  1. --cloudsen12 /path/to/CloudSEN12   (a local extract, ImageFolder layout:
     one folder of clear patches and one of cloudy patches). Used when present.
  2. The substitution actually used by default: EuroSAT supplies the CLEAN class
     directly (every patch in the subset is a cloud-free scene - that is what
     EuroSAT labels), and the CLOUDY class is the same patches with a synthetic
     cloud veil composited on top: 2-4 low-frequency Gaussian blobs, brightened
     toward white and partially desaturated. This reproduces the radiance
     signature a thin-to-thick cloud leaves in RGB without pretending we have
     real cloud masks, and it keeps the screener honest about what it learned:
     "is this patch veiled", not "does this look like CloudSEN12".

The active path is printed when the screener is trained.

    python cloud.py                        # train the default (EuroSAT-derived) screener
    python cloud.py --cloudsen12 ./CloudSEN12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils import SEED, save_json

FEATURE_NAMES = [
    "mean", "std", "sat_mean", "sat_std", "p90",
    "edge_energy", "blue_minus_red", "dynamic_range",
]


def to_hwc(image) -> np.ndarray:
    """(3, 64, 64) float tensor or array -> (64, 64, 3) float32 in [0, 1]."""
    array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def patch_features(image) -> np.ndarray:
    """Eight radiometric features. Cheap enough to run on the flight processor."""
    rgb = to_hwc(image)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)
    edge_energy = (
        np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()
    ) / 2.0
    return np.array(
        [
            float(gray.mean()),
            float(gray.std()),
            float(saturation.mean()),
            float(saturation.std()),
            float(np.percentile(gray, 90)),
            float(edge_energy),
            float((rgb[..., 2] - rgb[..., 0]).mean()),
            float(gray.max() - np.percentile(gray, 10)),
        ],
        dtype=np.float32,
    )


def add_cloud_veil(image, rng, strength=None) -> np.ndarray:
    """Composite a plausible cloud over a clean patch."""
    rgb = to_hwc(image)
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(2, 5))):
        centre_y = rng.uniform(0, height)
        centre_x = rng.uniform(0, width)
        sigma_y = rng.uniform(0.15, 0.45) * height
        sigma_x = rng.uniform(0.15, 0.50) * width
        mask += np.exp(
            -(((yy - centre_y) ** 2) / (2 * sigma_y ** 2) + ((xx - centre_x) ** 2) / (2 * sigma_x ** 2))
        )
    mask = np.clip(mask / max(float(mask.max()), 1e-6), 0.0, 1.0)

    alpha = float(strength if strength is not None else rng.uniform(0.35, 0.90))
    veil = (mask * alpha)[..., None]
    brightened = rgb * (1.0 - veil) + veil
    local_mean = brightened.mean(axis=2, keepdims=True)
    desaturated = brightened * (1.0 - 0.4 * veil) + local_mean * (0.4 * veil)
    return np.clip(desaturated, 0.0, 1.0)


def build_from_eurosat(limit=None):
    """Clean EuroSAT patches as class 0, the same patches veiled as class 1."""
    from data import build_splits

    splits = build_splits()
    rng = np.random.default_rng(SEED)
    features, labels = [], []

    for index in range(len(splits.train)):
        patch, _ = splits.train[index]
        features.append(patch_features(patch))
        labels.append(0)
        if index % 2 == 0:
            features.append(patch_features(add_cloud_veil(patch, rng)))
            labels.append(1)
        if limit and index >= limit:
            break

    return np.stack(features), np.asarray(labels), "eurosat-synthetic"


def build_from_cloudsen12(root: str):
    """Local CloudSEN12 extract (ImageFolder layout: clear / cloudy subfolders)."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root, transform=transforms.ToTensor())
    features = [patch_features(dataset[i][0]) for i in range(len(dataset))]
    labels = np.asarray([label for _, label in dataset])
    return np.stack(features), labels, "cloudsen12:" + root


def train_screener(cloudsen12_root=None, out_path: str = "runs/cloud_screener.joblib") -> dict:
    if cloudsen12_root and Path(cloudsen12_root).exists():
        features, labels, source = build_from_cloudsen12(cloudsen12_root)
    else:
        if cloudsen12_root:
            print("[cloud] " + cloudsen12_root + " not found - falling back to the EuroSAT-derived signal")
        features, labels, source = build_from_eurosat()

    print("[cloud] training on " + str(len(labels)) + " patches from " + source)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=SEED, stratify=labels
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "source": source,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_names": FEATURE_NAMES,
        "coefficients": [float(v) for v in model.coef_[0]],
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    save_json(Path(out_path).with_name("cloud_screener.json"), metrics)

    print("[cloud] accuracy " + format(metrics["accuracy"], ".4f")
          + "  macro F1 " + format(metrics["macro_f1"], ".4f"))
    print("[cloud] saved " + out_path)
    return metrics


def load_screener(path: str = "runs/cloud_screener.joblib"):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def screen(image, screener) -> dict:
    """Probability that a patch is veiled, from the eight radiometric features."""
    probability = float(screener.predict_proba(patch_features(image)[None, :])[0][1])
    return {"cloud_score": round(probability, 4), "cloudy": probability >= 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cloud/clean screener.")
    parser.add_argument("--cloudsen12", type=str, default=None,
                        help="path to a local CloudSEN12 extract (clear/ and cloudy/ subfolders)")
    parser.add_argument("--out", type=str, default="runs/cloud_screener.joblib")
    args = parser.parse_args()
    train_screener(args.cloudsen12, args.out)


if __name__ == "__main__":
    main()
