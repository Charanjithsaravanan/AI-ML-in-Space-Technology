"""Robustness evaluation on the 400 held-out patches.

    python evaluate.py --checkpoint runs/best.pt

Every row is the same 400 patches: once clean, then degraded by each noise mode
at severity 0.05 / 0.15 / 0.30. Nothing here trains or re-splits - the split is
rebuilt from the same seeds, so the test patches are identical to the ones
train.py never saw.

Writes results/robustness.json and results/robustness.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import f1_score

from data import build_splits, make_loader
from model import NormalizedModel, TinySatCNN
from noise import inject_sensor_noise
from utils import CLASSES, NOISE_MODES, SEVERITIES, save_json


def load_model(checkpoint_path: str, device: torch.device) -> NormalizedModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TinySatCNN(num_classes=len(checkpoint.get("classes", CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return NormalizedModel(model, checkpoint["mean"], checkpoint["std"]).to(device).eval()


@torch.no_grad()
def evaluate_condition(model, loader, device, mode: str, severity: float) -> dict:
    """Macro F1 (+ per-class F1) for one noise condition over the whole test split."""
    predictions, targets = [], []
    for images, labels in loader:
        if mode != "clean":
            images = inject_sensor_noise(images, mode, severity)
        logits = model(images.to(device))
        predictions.append(logits.argmax(1).cpu())
        targets.append(labels)

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()

    return {
        "mode": mode,
        "severity": float(severity),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "accuracy": float((predictions == targets).mean()),
        "per_class_f1": [
            float(v) for v in f1_score(targets, predictions, average=None, labels=list(range(10)))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean + noisy macro F1 on the held-out patches.")
    parser.add_argument("--checkpoint", type=str, default="runs/best.pt")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cpu")
    splits = build_splits(root=args.data_root)
    loader = make_loader(splits.test, args.batch_size)
    model = load_model(args.checkpoint, device)

    conditions = [("clean", 0.0)]
    conditions += [(mode, severity) for mode in NOISE_MODES for severity in SEVERITIES]

    rows = [evaluate_condition(model, loader, device, mode, severity) for mode, severity in conditions]

    lines = ["| condition | macro F1 | accuracy | delta vs clean |", "|---|---|---|---|"]
    clean_f1 = rows[0]["macro_f1"]
    for row in rows:
        name = "clean" if row["mode"] == "clean" else row["mode"] + " @ " + format(row["severity"], ".2f")
        delta = "—" if row["mode"] == "clean" else format(row["macro_f1"] - clean_f1, "+.4f")
        lines.append("| " + name + " | " + format(row["macro_f1"], ".4f") + " | "
                     + format(row["accuracy"], ".4f") + " | " + delta + " |")

    table = "\n".join(lines)
    print(table)

    worst_index = min(range(len(rows)), key=lambda i: rows[i]["macro_f1"])
    worst = rows[worst_index]

    per_class_lines = ["| class | clean F1 | worst-case F1 |", "|---|---|---|"]
    for index, name in enumerate(splits.classes):
        per_class_lines.append(
            "| " + name + " | " + format(rows[0]["per_class_f1"][index], ".4f")
            + " | " + format(worst["per_class_f1"][index], ".4f") + " |"
        )
    per_class_table = "\n".join(per_class_lines)
    print("\n" + per_class_table)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "robustness.json", {"conditions": rows, "classes": splits.classes})

    heading = (
        "\n\n### Per-class F1 (clean vs worst condition: "
        + str(worst["mode"]) + " @ " + format(worst["severity"], ".2f") + ")\n\n"
    )
    (out_dir / "robustness.md").write_text(
        "# Robustness\n\n" + table + heading + per_class_table + "\n", encoding="utf-8"
    )
    print("\nwrote " + str(out_dir / "robustness.json") + " and " + str(out_dir / "robustness.md"))


if __name__ == "__main__":
    main()
