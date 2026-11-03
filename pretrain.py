"""Self-supervised Landsat representation pretraining.

This learns from unlabeled RGB tiles with a SimCLR-style contrastive objective.
The resulting encoder can initialize ``train.py``; EuroSAT labels still provide
the supervised objective and the final class names.

    python pretrain.py --tiles data/landsat/tiles --epochs 20 --out runs/landsat_encoder.pt
    python train.py --pretrained runs/landsat_encoder.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import GEOSATCNN
from utils import SEED, get_device, seed_everything


class TileDataset(Dataset):
    def __init__(self, root: Path):
        self.files = sorted(root.glob("*.png"))
        if not self.files:
            raise FileNotFoundError("No PNG tiles found in " + str(root))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), self.transform(image)


class ProjectionHead(nn.Module):
    def __init__(self, features: int, projection: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(inplace=True),
            nn.Linear(features, projection),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=1)


def contrastive_loss(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    representations = torch.cat([left, right], dim=0)
    logits = representations @ representations.T / temperature
    mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, -torch.inf)
    batch = left.size(0)
    targets = torch.cat([
        torch.arange(batch, 2 * batch, device=logits.device),
        torch.arange(batch, device=logits.device),
    ])
    return F.cross_entropy(logits, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GEOSAT on unlabeled Landsat tiles.")
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("runs/landsat_encoder.pt"))
    args = parser.parse_args()

    seed_everything(SEED)
    device = get_device()
    loader = DataLoader(TileDataset(args.tiles), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
    encoder = GEOSATCNN(num_classes=10).to(device)
    projection = ProjectionHead(encoder.classifier.in_features).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection.parameters()), lr=args.lr, weight_decay=1e-4
    )

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        encoder.train()
        projection.train()
        total = 0.0
        for left, right in loader:
            left, right = left.to(device), right.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(
                projection(encoder.encode(left)),
                projection(encoder.encode(right)),
                args.temperature,
            )
            loss.backward()
            optimizer.step()
            total += loss.item()
        print("epoch " + str(epoch) + "/" + str(args.epochs)
              + "  loss " + format(total / max(len(loader), 1), ".4f")
              + "  " + format(time.time() - started, ".1f") + "s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "args": vars(args)}, args.out)
    print("saved " + str(args.out))


if __name__ == "__main__":
    main()
