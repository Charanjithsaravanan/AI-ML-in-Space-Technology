"""Prepare Landsat GeoTIFF scenes for GEOSAT patch inference.

The Kaggle Landsat scene is unlabeled and multispectral, so it is not mixed
into the supervised EuroSAT training split. This utility converts non-overlapping
scene windows into 64x64 RGB PNGs that can be passed to ``infer.py``.

For the dataset's usual band order (B2, B3, B4, B5, B6, B7), RGB is bands
4, 3, 2 in Landsat names and raster indexes 3, 2, 1 respectively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _rasterio():
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "Landsat support requires rasterio. Install it with "
            "`python -m pip install -r requirements-landsat.txt`."
        ) from exc
    return rasterio, Window


def _image_dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Landsat support requires NumPy and Pillow. Install the project "
            "requirements before preparing tiles."
        ) from exc
    return np, Image


def _limits(dataset, bands: Iterable[int]) -> list[tuple[float, float]]:
    np, _ = _image_dependencies()
    limits = []
    for band in bands:
        sample = dataset.read(
            band,
            out_shape=(min(dataset.height, 512), min(dataset.width, 512)),
            masked=True,
        )
        values = np.asarray(sample.compressed(), dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Band " + str(band) + " contains no valid pixels")
        low, high = np.percentile(values, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Band " + str(band) + " has unusable radiometric limits")
        limits.append((float(low), float(high)))
    return limits


def _to_rgb(values: np.ndarray, limits: list[tuple[float, float]]) -> np.ndarray:
    np, _ = _image_dependencies()
    channels = []
    for channel, (low, high) in zip(values, limits):
        scaled = (channel.astype(np.float32) - low) / (high - low)
        channels.append(np.clip(scaled, 0.0, 1.0))
    return (np.stack(channels, axis=-1) * 255.0).round().astype(np.uint8)


def prepare_scene(
    source: Path,
    output_dir: Path,
    bands: tuple[int, int, int] = (3, 2, 1),
    tile_size: int = 64,
    stride: int | None = None,
    limit: int | None = None,
) -> int:
    rasterio, Window = _rasterio()
    _, Image = _image_dependencies()
    if tile_size <= 0:
        raise ValueError("tile-size must be positive")
    if stride is None:
        stride = tile_size
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(set(bands)) != 3 or min(bands) < 1:
        raise ValueError("bands must contain three distinct 1-based raster indexes")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    with rasterio.open(source) as dataset:
        if max(bands) > dataset.count:
            raise ValueError(
                "Requested band " + str(max(bands))
                + " but the GeoTIFF contains only " + str(dataset.count) + " bands"
            )
        limits = _limits(dataset, bands)
        tile_number = 0
        for row in range(0, dataset.height, stride):
            for column in range(0, dataset.width, stride):
                if limit is not None and tile_number >= limit:
                    break
                window = Window(column, row, tile_size, tile_size)
                values = dataset.read(
                    list(bands),
                    window=window,
                    boundless=True,
                    fill_value=0,
                )
                valid = dataset.read_masks(
                    1,
                    window=window,
                    boundless=True,
                ) > 0
                if not valid.any():
                    continue
                name = "tile_" + str(tile_number).zfill(6) + ".png"
                Image.fromarray(_to_rgb(values, limits), mode="RGB").save(output_dir / name)
                metadata.append(
                    {
                        "file": name,
                        "row": row,
                        "column": column,
                        "height": min(tile_size, dataset.height - row),
                        "width": min(tile_size, dataset.width - column),
                        "bands": list(bands),
                        "transform": list(dataset.window_transform(window)),
                    }
                )
                tile_number += 1
            if limit is not None and tile_number >= limit:
                break

        manifest = {
            "source": str(source),
            "crs": str(dataset.crs) if dataset.crs else None,
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "tile_size": tile_size,
            "stride": stride,
            "band_limits": limits,
            "tiles": metadata,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Landsat GeoTIFF into GEOSAT-ready RGB tiles."
    )
    parser.add_argument("--image", required=True, type=Path, help="input GeoTIFF")
    parser.add_argument("--output-dir", type=Path, default=Path("data/landsat/tiles"))
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        default=(3, 2, 1),
        metavar=("RED", "GREEN", "BLUE"),
        help="1-based raster indexes; default assumes TIFF order B2,B3,B4,...",
    )
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int, help="window stride; defaults to tile-size")
    parser.add_argument("--limit", type=int, help="maximum number of tiles to write")
    args = parser.parse_args()
    count = prepare_scene(
        args.image,
        args.output_dir,
        tuple(args.bands),
        args.tile_size,
        args.stride,
        args.limit,
    )
    print("wrote " + str(count) + " tiles to " + str(args.output_dir))


if __name__ == "__main__":
    main()
