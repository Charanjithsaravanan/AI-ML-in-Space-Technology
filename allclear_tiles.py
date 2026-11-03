"""Extract RGB self-supervised tiles from the AllClear train split.

AllClear is a cloud-removal dataset, not a ten-class land-cover dataset. This
utility uses only Sentinel-2 optical files belonging to the official train ROI
list, converts bands B4/B3/B2 to RGB, and writes tiles for pretrain.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _image_dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "AllClear tile extraction requires NumPy and Pillow. "
            "Install the project requirements first."
        ) from exc
    return np, Image


def prepare(
    data_root: Path,
    roi_list: Path,
    output_dir: Path,
    tile_size: int = 64,
    stride: int | None = None,
    max_files: int | None = None,
    limit: int | None = None,
) -> int:
    np, Image = _image_dependencies()
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError(
            "AllClear tile extraction requires rasterio. Install "
            "requirements-landsat.txt first."
        ) from exc

    if tile_size <= 0:
        raise ValueError("tile-size must be positive")
    stride = tile_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    train_rois = {
        line.strip() for line in roi_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    files = []
    for path in sorted(data_root.rglob("*.tif")):
        relative = path.relative_to(data_root)
        if "s2_toa" in relative.parts and relative.parts[0] in train_rois:
            files.append(path)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            "No AllClear train Sentinel-2 files found under " + str(data_root)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for source in files:
        with rasterio.open(source) as dataset:
            if dataset.count < 4:
                raise ValueError(str(source) + " has fewer than four Sentinel-2 bands")
            sample = dataset.read([4, 3, 2], masked=True)
            limits = []
            for channel in sample:
                values = np.asarray(channel.compressed(), dtype=np.float32)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    raise ValueError(str(source) + " contains no valid RGB pixels")
                limits.append(np.percentile(values, [2, 98]))

            for row in range(0, dataset.height, stride):
                for column in range(0, dataset.width, stride):
                    if limit is not None and written >= limit:
                        return written
                    window = Window(column, row, tile_size, tile_size)
                    values = dataset.read([4, 3, 2], window=window, boundless=True, fill_value=0)
                    valid = dataset.read_masks(1, window=window, boundless=True, fill_value=0) > 0
                    if not valid.any():
                        continue
                    channels = []
                    for channel, (low, high) in zip(values, limits):
                        channels.append(np.clip((channel - low) / max(high - low, 1e-8), 0, 1))
                    image = (np.stack(channels, axis=-1) * 255).round().astype(np.uint8)
                    Image.fromarray(image, mode="RGB").save(
                        output_dir / ("tile_" + str(written).zfill(7) + ".png")
                    )
                    written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AllClear train RGB tiles.")
    parser.add_argument("--data-root", type=Path, default=Path("data/allclear/data"))
    parser.add_argument("--roi-list", type=Path,
                        default=Path("data/allclear/metadata/rois/train_rois_19k.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/allclear/tiles"))
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-files", type=int, help="limit source GeoTIFFs")
    parser.add_argument("--limit", type=int, help="limit output tiles")
    args = parser.parse_args()
    count = prepare(args.data_root, args.roi_list, args.output_dir, args.tile_size,
                    args.stride, args.max_files, args.limit)
    print("wrote " + str(count) + " AllClear train tiles to " + str(args.output_dir))


if __name__ == "__main__":
    main()
