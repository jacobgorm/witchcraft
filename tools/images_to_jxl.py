#!/usr/bin/env python3
"""Recursively convert images into a flat directory of JPEG XL files.

Requires cjxl. HEIC/HEIF and formats not read directly by cjxl also require
ImageMagick's `magick` command with the relevant decoder delegate.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".apng",
    ".avif",
    ".bmp",
    ".dib",
    ".exr",
    ".gif",
    ".heic",
    ".heif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".pam",
    ".pfm",
    ".pgm",
    ".pgx",
    ".png",
    ".pnm",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

# These formats are accepted directly by the reference cjxl command. Passing
# JPEG directly also preserves libjxl's lossless JPEG reconstruction path.
CJXL_INPUT_EXTENSIONS = {
    ".apng",
    ".exr",
    ".gif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".pam",
    ".pfm",
    ".pgm",
    ".pgx",
    ".png",
    ".pnm",
    ".ppm",
}


def flattened_name(relative: Path) -> str:
    parts = list(relative.parts)
    parts[-1] = Path(parts[-1]).stem
    label = "--".join(parts)
    label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-") or "image"
    digest = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]
    return f"{label[:180]}--{digest}.jxl"


def discover_images(source: Path, destination: Path) -> list[Path]:
    images = []
    destination = destination.resolve()
    for directory, child_directories, filenames in os.walk(source):
        directory_path = Path(directory)
        child_directories.sort()
        child_directories[:] = [
            child
            for child in child_directories
            if (directory_path / child).resolve() != destination
        ]
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return images


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode == 0:
        return
    detail = completed.stderr.strip() or completed.stdout.strip()
    raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")


def temporary_output(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".jxl", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def encode_lossless(source: Path, destination: Path, cjxl: str, magick: str | None) -> None:
    temporary = temporary_output(destination)
    try:
        extension = source.suffix.lower()
        if extension == ".jxl":
            shutil.copyfile(source, temporary)
        elif extension in CJXL_INPUT_EXTENSIONS:
            run([cjxl, str(source), str(temporary), "-d", "0", "-e", "9", "--quiet"])
        else:
            if magick is None:
                raise RuntimeError(f"ImageMagick is required to decode {source}")
            with tempfile.TemporaryDirectory(prefix="images-to-jxl-") as work:
                decoded = Path(work) / "decoded.png"
                # Bake orientation into the pixels and retain the decoder's
                # precision and embedded color profile in a 16-bit PNG.
                run([
                    magick,
                    f"{source}[0]",
                    "-auto-orient",
                    "-depth",
                    "16",
                    f"png:{decoded}",
                ])
                run([cjxl, str(decoded), str(temporary), "-d", "0", "-e", "9", "--quiet"])
        os.replace(temporary, destination)
        shutil.copystat(source, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert images to a collision-safe flat directory of lossless JPEG XL files."
    )
    parser.add_argument("source", type=Path, help="directory subtree to scan")
    parser.add_argument("destination", type=Path, help="flat output directory")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output files"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source is not a directory: {source}")
    if source == destination:
        raise SystemExit("source and destination must be different directories")

    cjxl = shutil.which("cjxl")
    if cjxl is None:
        raise SystemExit("cjxl was not found; on macOS install it with: brew install jpeg-xl")

    images = discover_images(source, destination)
    needs_magick = any(
        path.suffix.lower() not in CJXL_INPUT_EXTENSIONS | {".jxl"} for path in images
    )
    magick = shutil.which("magick") if needs_magick else None
    if needs_magick and magick is None:
        raise SystemExit(
            "ImageMagick was not found; it is required for HEIC/HEIF and other non-cjxl inputs"
        )

    destination.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    failed = 0
    for index, source_path in enumerate(images, 1):
        relative = source_path.relative_to(source)
        output = destination / flattened_name(relative)
        if output.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(images)}] skip {relative} -> {output.name}")
            continue
        print(f"[{index}/{len(images)}] {relative} -> {output.name}")
        try:
            encode_lossless(source_path, output, cjxl, magick)
        except Exception as error:
            print(f"error: {relative}: {error}", file=sys.stderr)
            failed += 1
            continue
        converted += 1

    print(
        f"converted={converted} skipped={skipped} failed={failed} discovered={len(images)}"
    )
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
