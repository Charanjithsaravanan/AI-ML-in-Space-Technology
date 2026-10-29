"""Export, size and CPU latency - the third axis of the trade-off.

    python benchmark.py --checkpoint runs/best.pt

Measures:
    * parameter count and file size of the exported TorchScript graph (a real
      file on disk, not an estimate)
    * median / p95 per-patch latency, single thread, batch 1, after warm-up

Then merges with results/robustness.json, when present, to print the trade-off.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import NormalizedModel, TinySatCNN
from utils import CLASSES, count_parameters, human_mb, load_json, parameters_to_mb, save_json


def export_torchscript(model: NormalizedModel, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), example)
    traced.save(str(out_path))
    return out_path


def export_onnx(model: NormalizedModel, out_path: Path):
    """Optional. Returns None when the onnx package is not installed."""
    try:
        import onnx  # noqa: F401
    except Exception:
        print("[benchmark] onnx not installed - skipping ONNX export (TorchScript is the deliverable)")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(),
        torch.rand(1, 3, 64, 64),
        str(out_path),
        input_names=["patch"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    return out_path


def measure_latency(model, repeats: int = 200, warmup: int = 50, threads: int = 1) -> dict:
    """Median per-patch wall time on CPU, batch 1, single thread.

    Single thread and batch 1 are the honest numbers for a flight processor:
    there is no batch to amortise and no SIMD-heavy host to hide behind.
    """
    torch.set_num_threads(threads)
    model.eval()
    example = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        for _ in range(warmup):
            model(example)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(example)
            samples.append((time.perf_counter() - started) * 1000.0)

    samples = np.asarray(samples)
    return {
        "median_ms": float(np.median(samples)),
        "mean_ms": float(samples.mean()),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(samples.min()),
        "repeats": repeats,
        "warmup": warmup,
        "threads": threads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Size + latency benchmark and trade-off table.")
    parser.add_argument("--checkpoint", type=str, default="runs/best.pt")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TinySatCNN(num_classes=len(checkpoint.get("classes", CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    wrapped = NormalizedModel(model, checkpoint["mean"], checkpoint["std"]).eval()

    n_params = count_parameters(model)
    ts_path = export_torchscript(wrapped, Path("runs/model_ts.pt"))
    onnx_path = export_onnx(wrapped, Path("runs/model.onnx"))

    ts_mb = human_mb(ts_path.stat().st_size)
    report = {
        "parameters": n_params,
        "parameter_mb_fp32": parameters_to_mb(n_params),
        "torchscript_mb": ts_mb,
        "onnx_mb": human_mb(onnx_path.stat().st_size) if onnx_path else None,
        "latency": measure_latency(wrapped, args.repeats, args.warmup, args.threads),
    }
    save_json(Path(args.out_dir) / "benchmark.json", report)

    print("parameters        " + format(n_params, ","))
    print("fp32 weights      " + format(report["parameter_mb_fp32"], ".4f") + " MiB")
    print("torchscript file  " + format(ts_mb, ".4f") + " MiB  (" + str(ts_path) + ")")
    if onnx_path:
        print("onnx file         " + format(report["onnx_mb"], ".4f") + " MiB  (" + str(onnx_path) + ")")
    print(
        "latency (CPU x" + str(args.threads) + ", batch 1)  median "
        + format(report["latency"]["median_ms"], ".3f") + " ms  p95 "
        + format(report["latency"]["p95_ms"], ".3f") + " ms  over " + str(args.repeats) + " runs"
    )

    robustness_path = Path(args.out_dir) / "robustness.json"
    clean_f1 = worst_f1 = None
    if robustness_path.exists():
        conditions = load_json(robustness_path)["conditions"]
        clean_f1 = conditions[0]["macro_f1"]
        worst_f1 = min(row["macro_f1"] for row in conditions)

    tradeoff = [
        "| metric | value |",
        "|---|---|",
        "| macro F1 (clean) | " + ("—" if clean_f1 is None else format(clean_f1, ".4f")) + " |",
        "| macro F1 (worst noise) | " + ("—" if worst_f1 is None else format(worst_f1, ".4f")) + " |",
        "| parameters | " + format(n_params, ",") + " |",
        "| exported size | " + format(ts_mb, ".3f") + " MiB |",
        "| latency (median) | " + format(report["latency"]["median_ms"], ".3f") + " ms |",
        "",
        "Run evaluate.py first to populate the F1 rows.",
    ]
    tradeoff_md = "\n".join(tradeoff)
    (Path(args.out_dir) / "tradeoff.md").write_text(tradeoff_md + "\n", encoding="utf-8")
    print("\n" + tradeoff_md)


if __name__ == "__main__":
    main()
