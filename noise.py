"""Synthetic sensor-noise models.

inject_sensor_noise(images, mode, severity) is the evaluation contract: identical
signature, identical behaviour, used both by the training augmentation and by
every robustness measurement. Do not change its semantics - a changed helper
would make the reported table incomparable with the training distribution.

Input is a float tensor in [0, 1], shape (B, C, H, W) or (C, H, W).
Output has the same shape and stays in [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MODES = ("gaussian", "salt_pepper", "band_dropout")


def _as_batch(images: torch.Tensor):
    if images.dim() == 3:
        return images.unsqueeze(0), True
    if images.dim() != 4:
        raise ValueError("expected (B, C, H, W) or (C, H, W), got " + str(tuple(images.shape)))
    return images, False


def _restore(batch: torch.Tensor, was_unbatched: bool) -> torch.Tensor:
    return batch[0] if was_unbatched else batch


def gaussian_noise(images: torch.Tensor, severity: float) -> torch.Tensor:
    """Additive white Gaussian noise with sigma equal to the severity."""
    batch, unbatched = _as_batch(images)
    noisy = batch + torch.randn_like(batch) * float(severity)
    return _restore(noisy.clamp(0.0, 1.0), unbatched)


def salt_pepper_noise(images: torch.Tensor, severity: float) -> torch.Tensor:
    """A fraction of elements equal to the severity is forced to 0 or 1, split evenly."""
    batch, unbatched = _as_batch(images)
    prob = float(severity)
    draw = torch.rand_like(batch)
    out = batch.clone()
    out = torch.where(draw < prob / 2.0, torch.zeros_like(out), out)
    out = torch.where(draw > 1.0 - prob / 2.0, torch.ones_like(out), out)
    return _restore(out.clamp(0.0, 1.0), unbatched)


def band_dropout_noise(images: torch.Tensor, severity: float) -> torch.Tensor:
    """Contiguous stripe dropout.

    For 3-band RGB input a true spectral band loss leaves only 2 channels, which
    is the same failure seen from orbit as a dropped detector line. We model the
    visible signature: one contiguous column stripe, severity x width wide, is
    zeroed. It is per-patch random so a model cannot memorise a fixed location.
    """
    batch, unbatched = _as_batch(images)
    out = batch.clone()
    width = out.shape[-1]
    drop = max(1, int(round(float(severity) * width)))
    for i in range(out.shape[0]):
        if drop >= width:
            out[i] = 0.0
            continue
        start = int(torch.randint(0, width - drop + 1, (1,)).item())
        out[i, :, :, start:start + drop] = 0.0
    return _restore(out, unbatched)


def inject_sensor_noise(images: torch.Tensor, mode: str, severity: float) -> torch.Tensor:
    """Apply one named sensor-degradation mode at the given severity."""
    if mode == "clean" or float(severity) == 0.0:
        return images.clone()
    if mode == "gaussian":
        return gaussian_noise(images, severity)
    if mode == "salt_pepper":
        return salt_pepper_noise(images, severity)
    if mode == "band_dropout":
        return band_dropout_noise(images, severity)
    raise ValueError("unknown noise mode: " + repr(mode) + " (expected one of " + str(MODES) + ")")


@torch.no_grad()
def estimate_noise_sigma(images: torch.Tensor) -> torch.Tensor:
    """Immerkaer's fast noise estimate: mean |I * M| / 6 with M a 3x3 Laplacian.

    Returns one sigma per image, in [0, 1] pixel units.
    """
    batch, unbatched = _as_batch(images)
    channels = batch.shape[1]
    kernel = torch.tensor(
        [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
        dtype=batch.dtype,
        device=batch.device,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    response = F.conv2d(batch, kernel, groups=channels, padding=1)
    sigma = response.abs().mean(dim=(1, 2, 3)) * math.sqrt(math.pi / 2.0) / 6.0
    return sigma[0] if unbatched else sigma


@torch.no_grad()
def detect_corruption(image: torch.Tensor, sigma_threshold: float = 0.08) -> dict:
    """Cheap on-board triage: is this tile degraded, and by what?

    Used by infer.py to print a flag next to the prediction. Deliberately
    rule-based - it costs microseconds and needs no second network.
    """
    batch, _ = _as_batch(image)
    sigma = float(estimate_noise_sigma(batch)[0])

    pixels = batch[0]
    saturated = float(((pixels <= 1e-6) | (pixels >= 1.0 - 1e-6)).float().mean())
    column_energy = pixels.amax(dim=(0, 1))
    dropped_columns = float((column_energy <= 1e-6).float().mean())

    if dropped_columns > 0.03:
        hint = "band_dropout"
    elif saturated > 0.02:
        hint = "salt_pepper"
    elif sigma > sigma_threshold:
        hint = "gaussian"
    else:
        hint = None

    return {
        "corrupted": hint is not None,
        "likely_mode": hint,
        "sigma": round(sigma, 4),
        "dropped_columns": round(dropped_columns, 4),
        "saturated_fraction": round(saturated, 4),
    }
