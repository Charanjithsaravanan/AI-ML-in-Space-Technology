"""EuroSAT loading, the fixed 2,000-patch subset, and training-time augmentation.

Split protocol - identical for every run and every team, do not edit:

    torch.manual_seed(42)
    perm = torch.randperm(len(full_dataset))[:2000]
    random_split(that subset, [1600, 400], generator=torch.Generator().manual_seed(42))

Noise augmentation happens here and only here, inside TrainAugment, which wraps
the training split exclusively. The test split is never wrapped, so every
reported number is measured on clean patches plus deliberately injected noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import EuroSAT

from noise import inject_sensor_noise
from utils import NOISE_MODES, NUM_CLASSES, SEED

BATCH_SIZE = 32
SUBSET_SIZE = 2000
N_TRAIN = 1600
N_TEST = 400
VAL_SIZE = 200


@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset
    classes: List[str]


def build_full_dataset(root: str = "./data") -> EuroSAT:
    """EuroSAT RGB, downloaded on first use, patches as float tensors in [0, 1].

    torchvision's EuroSAT downloader reaches its mirror through an unverified
    SSL context. Keep this call exactly as it is - replacing it with a custom
    downloader changes both the patch order and the integrity check.
    """
    return EuroSAT(root=root, download=True, transform=transforms.ToTensor())


def build_splits(
    root: str = "./data",
    val_size: int = VAL_SIZE,
    seed: int = SEED,
    full: bool = False,
) -> Splits:
    """2,000 fixed patches -> 1,600 train / 400 test, then 200 of the train set held out.

    The validation carve uses seed + 1, so it can never collide with the
    permutation that produced the subset itself.
    """
    torch.manual_seed(seed)
    full_dataset = build_full_dataset(root)
    if full:
        train, val = random_split(
            full_dataset,
            [len(full_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(seed + 1),
        )
        return Splits(
            train=train,
            val=val,
            test=Subset(full_dataset, []),
            classes=list(full_dataset.classes),
        )

    perm = torch.randperm(len(full_dataset))[:SUBSET_SIZE]
    subset = Subset(full_dataset, perm.tolist())

    train_pool, test = random_split(
        subset, [N_TRAIN, N_TEST], generator=torch.Generator().manual_seed(seed)
    )
    train, val = random_split(
        train_pool,
        [N_TRAIN - val_size, val_size],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    classes = list(full_dataset.classes)
    assert len(classes) == NUM_CLASSES, "expected " + str(NUM_CLASSES) + " classes, got " + str(len(classes))
    return Splits(train=train, val=val, test=test, classes=classes)


class TrainAugment(Dataset):
    """Flip / 90-degree rotation / on-the-fly sensor noise. TRAIN SPLIT ONLY.

    The augmentation distribution deliberately overlaps the evaluation grid:
    noise is drawn uniformly from severity 0.02 up to 0.30 across all three
    modes, so the model sees the same corruptions it will be measured on -
    that is what makes the robustness table a measurement rather than a surprise.
    """

    def __init__(self, base: Dataset, p_noise: float = 0.5, max_severity: float = 0.30, seed: int = SEED):
        self.base = base
        self.p_noise = p_noise
        self.max_severity = max_severity
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base[index]

        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=[2])
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=[1])
        turns = self.rng.randint(0, 3)
        if turns:
            image = torch.rot90(image, turns, dims=[1, 2])

        if self.rng.random() < self.p_noise:
            mode = self.rng.choice(list(NOISE_MODES))
            severity = self.rng.uniform(0.02, self.max_severity)
            image = inject_sensor_noise(image.unsqueeze(0), mode, severity).squeeze(0)

        return image.contiguous().clamp(0.0, 1.0), label


@torch.no_grad()
def channel_stats(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over CLEAN training patches, as 1x3x1x1 tensors.

    Computed from the training split only (no validation, no test) so the
    normalisation carries no information from the held-out patches.
    """
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for i in range(len(dataset)):
        image, _ = dataset[i]
        total += image.sum(dim=(1, 2)).double()
        total_sq += (image.double() ** 2).sum(dim=(1, 2))
        count += image.shape[1] * image.shape[2]
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp_min(1e-8).sqrt()
    return mean.float().view(1, 3, 1, 1), std.float().view(1, 3, 1, 1)


def make_loader(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    seed: int = SEED,
) -> DataLoader:
    """Single-process loader: one RNG, one order, no worker non-determinism."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
