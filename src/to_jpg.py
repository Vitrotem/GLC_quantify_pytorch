"""Convert all images under a folder (recursively) to max-quality JPEG."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.constants import IMAGE_EXTENSIONS

# Broader than quantify inputs — convert common formats found in folders.
CONVERT_EXTENSIONS = IMAGE_EXTENSIONS | {".bmp", ".webp", ".gif"}


def coalesce_folder_path(parts: list[str]) -> Path:
    """Join argv fragments that were split by spaces in folder names."""
    for j in range(len(parts), 0, -1):
        candidate = Path(" ".join(parts[:j]))
        if candidate.exists():
            return candidate
    return Path(" ".join(parts))


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in CONVERT_EXTENSIONS
    )


def jpg_destination(path: Path) -> Path:
    return path.with_suffix(".jpg")


def convert_to_jpg(path: Path, *, overwrite: bool) -> str:
    """Convert one image to max-quality JPG. Returns status: converted|skipped|error."""
    dest = jpg_destination(path)
    same_path = path.resolve() == dest.resolve()

    if same_path and path.suffix.lower() == ".jpg" and not overwrite:
        return "skipped"

    if not same_path and dest.exists() and not overwrite:
        return "skipped"

    try:
        # Read into memory first so Windows can delete the source afterward.
        raw = path.read_bytes()
        with Image.open(BytesIO(raw)) as image:
            image.load()
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                rgb = background
            else:
                rgb = image.convert("RGB")

            temp_dest = dest.with_name(dest.stem + ".__tmp__.jpg")
            rgb.save(
                temp_dest,
                format="JPEG",
                quality=100,
                optimize=False,
                subsampling=0,
            )
    except OSError as exc:
        return f"error: {exc}"

    temp_dest.replace(dest)
    if not same_path:
        path.unlink()

    return "converted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all images in a folder (including subfolders) to max-quality JPG"
    )
    parser.add_argument(
        "folder",
        nargs="+",
        help="Folder to convert (quote paths that contain spaces)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode existing .jpg files and replace existing .jpg outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = coalesce_folder_path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    images = collect_images(folder)
    if not images:
        print(f"No images found under {folder}")
        return

    converted = skipped = errors = 0
    total = len(images)
    for i, path in enumerate(images, start=1):
        print(f"[{i}/{total}] {path.name}", end="\r")
        status = convert_to_jpg(path, overwrite=args.overwrite)
        if status == "converted":
            converted += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1
            print(f"\n{path}: {status}")

    print(
        f"\nDone: {converted} converted, {skipped} skipped, {errors} errors "
        f"(folder: {folder})"
    )


if __name__ == "__main__":
    main()
