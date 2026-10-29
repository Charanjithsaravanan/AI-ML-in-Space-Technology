"""Train the compact classifier on the fixed 1,600-patch EuroSAT subset.

    python train.py --epochs 30

Writes:
    runs/best.pt             state dict + normalisation stats + class list + args
    runs/best_torchscript.pt portable graph (normalisation included)
    runs/history.json        per-epoch loss / validation F1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import BATCH_SIZE, TrainAugment, build_splits, channel_stats, make_loader
from model import NormalizedModel, TinySatCNN
from utils import (
    SEED,
    count_parameters,
    get_device,
    parameters_to_mb,
    save_json,
    seed_everything,
)


def macro_f1(model: nn.Module, loader, mean, std, device) -> float:
    """Macro-averaged F1 - the reported headline metric, so it also selects the checkpoint."""
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(((images - mean) / std).to(device))
            predictions.append(logits.argmax(1).cpu())
            targets.append(labels)
    return float(f1_score(torch.cat(targets), torch.cat(predictions), average="macro"))


def train_one_epoch(model, loader, optimizer, criterion, mean, std, device) -> float:
    model.train()
    running, seen = 0.0, 0
    for images, labels in loader:
        images = ((images - mean) / std).to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        running += loss.item() * labels.size(0)
        seen += labels.size(0)
    return running / max(seen, 1)


def save_checkpoint(model, mean, std, classes, args, epoch, val_f1, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "classes": classes,
            "args": vars(args),
            "epoch": epoch,
            "val_macro_f1": val_f1,
        },
        path,
    )

    # Portable graph: trace once, on CPU, with the normalisation folded in.
    wrapped = NormalizedModel(model, mean, std).eval()
    example = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
    traced.save(str(path.with_name("best_torchscript.pt")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinySat on the fixed EuroSAT subset.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--noise-prob", type=float, default=0.5,
                        help="probability that a training patch gets synthetic sensor noise")
    parser.add_argument("--val-size", type=int, default=200,
                        help="patches held out of the 1,600 training patches for checkpoint selection")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--out", type=str, default="runs/best.pt")
    args = parser.parse_args()

    seed_everything(SEED)
    device = get_device()

    splits = build_splits(root=args.data_root, val_size=args.val_size)
    mean, std = channel_stats(splits.train)

    train_loader = make_loader(
        TrainAugment(splits.train, p_noise=args.noise_prob), args.batch_size, shuffle=True
    )
    val_loader = make_loader(splits.val, args.batch_size)

    model = TinySatCNN(num_classes=len(splits.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    n_params = count_parameters(model)
    print("device=" + str(device) + "  params=" + format(n_params, ",") + "  fp32=" + format(parameters_to_mb(n_params), ".3f") + " MiB")
    print("train=" + str(len(splits.train)) + "  val=" + str(len(splits.val)) + "  test=" + str(len(splits.test)))
    print("channel mean=" + str(mean.flatten().tolist()) + "  std=" + str(std.flatten().tolist()))

    best_f1, best_epoch = -1.0, -1
    history = []
    out_path = Path(args.out)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, mean, std, device)
        val_f1 = macro_f1(model, val_loader, mean, std, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(loss, 4),
                "val_macro_f1": round(val_f1, 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            "epoch " + str(epoch).zfill(2) + "/" + str(args.epochs)
            + "  loss " + format(loss, ".4f")
            + "  val_macro_f1 " + format(val_f1, ".4f")
            + "  lr " + format(optimizer.param_groups[0]["lr"], ".2e")
            + "  " + format(time.time() - started, ".1f") + "s"
        )

        # Checkpoint selection uses the validation carve only - never the test set.
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            save_checkpoint(model, mean, std, splits.classes, args, epoch, best_f1, out_path)
            print("  -> new best (val macro F1 " + format(best_f1, ".4f") + "), saved " + str(out_path))

    save_json(out_path.with_name("history.json"), history)
    print("\nbest epoch " + str(best_epoch) + "  val macro F1 " + format(best_f1, ".4f"))
    print("checkpoint " + str(out_path) + "  scripted " + str(out_path.with_name("best_torchscript.pt")))


if __name__ == "__main__":
    main()
