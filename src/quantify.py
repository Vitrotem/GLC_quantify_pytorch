"""Quantify GLC coverage inside TEM grid circles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.circle_tune_ui import QuantifySession, run_quantify_wizard
from src.constants import IMAGE_EXTENSIONS
from src.preprocessing import (
    CircleDetectParams,
    LevelsParams,
    SegParams,
    apply_levels_params,
    coverage_stats,
    crop_metadata,
    detect_circles,
    extract_circle_crops_from_gray,
    extract_circle_overlay_crops,
    filter_skipped_circles,
    inset_circles,
    mean_coverage,
    render_segmentation_overlay,
    segment_glc_params,
)

DEFAULT_OUTPUT_DIR = Path("outputs/quantify")
DEFAULT_CSV_NAME = "quantification_report.csv"
CSV_FIELDS = (
    "image",
    "circle_id",
    "x",
    "y",
    "r",
    "glc_area",
    "circle_area",
    "coverage_ratio",
    "status",
    "error",
)


def coalesce_image_paths(parts: list[str]) -> list[Path]:
    """Join argv fragments that were split by spaces in paths."""
    joined = " ".join(parts)
    # Prefer the longest existing prefix path, then remaining tokens as extra paths.
    tokens = parts
    paths: list[Path] = []
    i = 0
    while i < len(tokens):
        found = None
        for j in range(len(tokens), i, -1):
            candidate = Path(" ".join(tokens[i:j]))
            if candidate.exists():
                found = candidate
                i = j
                break
        if found is None:
            paths.append(Path(joined if not paths else " ".join(tokens[i:])))
            break
        paths.append(found)
    return paths or [Path(joined)]


def collect_images(paths: list[Path]) -> list[Path]:
    """Expand files/folders into a sorted list of image paths."""
    images: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        elif path.is_dir():
            images.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
    return images


def default_session() -> QuantifySession:
    return QuantifySession(
        detect_params=CircleDetectParams(),
        skipped_centers=(),
        levels=LevelsParams(),
        seg=SegParams(),
    )


def process_image(
    image_path: Path,
    output_dir: Path,
    session: QuantifySession,
    *,
    apply_skips: bool,
) -> tuple[list[dict[str, object]], float]:
    """Detect, segment, export crops/overlay; return CSV rows and mean coverage."""
    gray = np.array(Image.open(image_path).convert("L"))
    cropped, _ = crop_metadata(gray)
    circles = detect_circles(cropped, params=session.detect_params)
    skips = session.skipped_centers if apply_skips else ()
    kept = filter_skipped_circles(circles, skips)
    analysis = inset_circles(kept, session.inner_offset_pct)

    leveled = apply_levels_params(cropped, session.levels)
    soft = segment_glc_params(leveled, analysis, session.seg)
    stats = coverage_stats(soft, analysis)
    overlay = render_segmentation_overlay(leveled, soft, analysis)

    output_dir.mkdir(parents=True, exist_ok=True)
    circles_dir = output_dir / "circles"
    circles_dir.mkdir(parents=True, exist_ok=True)

    overlay.save(output_dir / "overlay.jpg", quality=95)
    gray_crops = extract_circle_crops_from_gray(leveled, analysis)
    overlay_crops = extract_circle_overlay_crops(overlay, analysis)

    for idx, crop in enumerate(gray_crops, start=1):
        crop.image.save(circles_dir / f"{idx:05d}.jpg", quality=95)
    for idx, crop in enumerate(overlay_crops, start=1):
        crop.image.save(circles_dir / f"{idx:05d}_overlay.jpg", quality=95)

    rows: list[dict[str, object]] = []
    for idx, s in enumerate(stats, start=1):
        rows.append(
            {
                "image": str(image_path),
                "circle_id": idx,
                "x": s.x,
                "y": s.y,
                "r": s.r,
                "glc_area": f"{s.glc_area:.2f}",
                "circle_area": f"{s.circle_area:.2f}",
                "coverage_ratio": f"{s.coverage_ratio:.6f}",
                "status": "ok",
                "error": "",
            }
        )

    total_glc = sum(s.glc_area for s in stats)
    total_area = sum(s.circle_area for s in stats)
    total_ratio = (total_glc / total_area) if total_area > 0 else 0.0
    rows.append(
        {
            "image": str(image_path),
            "circle_id": "TOTAL",
            "x": "",
            "y": "",
            "r": "",
            "glc_area": f"{total_glc:.2f}",
            "circle_area": f"{total_area:.2f}",
            "coverage_ratio": f"{total_ratio:.6f}",
            "status": "ok",
            "error": "",
        }
    )
    return rows, mean_coverage(stats)


def write_report_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify GLC coverage in TEM grid circles"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image file(s) and/or folder(s) (quote paths that contain spaces)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root when input is a file (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_NAME,
        help=f"Report CSV filename (default: {DEFAULT_CSV_NAME})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the quantification CSV",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Run the interactive wizard for every image (default: first image only)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Skip the interactive wizard and use default settings",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = coalesce_image_paths(args.paths)
    image_paths = collect_images(input_paths)
    if not image_paths:
        raise FileNotFoundError("No images found in the given path(s)")

    folder_mode = len(input_paths) == 1 and input_paths[0].is_dir()
    results_root = input_paths[0] if folder_mode else args.output_dir

    shared_session: QuantifySession | None = None
    if not args.no_ui and not args.per_image:
        print(f"Tuning on: {image_paths[0]}")
        shared_session = run_quantify_wizard(image_paths[0])
        if shared_session is None:
            print("Quantify wizard cancelled; aborting.")
            return
        print(
            "Settings: "
            f"param1={shared_session.detect_params.param1:.0f}, "
            f"param2={shared_session.detect_params.param2:.0f}, "
            f"levels=({shared_session.levels.black:.0f}, "
            f"{shared_session.levels.gamma:.2f}, "
            f"{shared_session.levels.white:.0f}), "
            f"threshold={shared_session.seg.threshold:.0f}, "
            f"fuzziness={shared_session.seg.fuzziness:.0f}, "
            f"gap_fill={shared_session.seg.gap_fill}, "
            f"split={shared_session.seg.split}, "
            f"min_size={shared_session.seg.min_size}, "
            f"max_size={shared_session.seg.max_size}, "
            f"inner_offset={shared_session.inner_offset_pct:.1f}%"
        )
    elif args.no_ui:
        shared_session = default_session()
        print("Using default quantify settings (--no-ui)")

    all_rows: list[dict[str, object]] = []
    batch_glc = 0.0
    batch_area = 0.0
    tuned_image = image_paths[0].resolve()

    for image_path in image_paths:
        if args.per_image and not args.no_ui:
            print(f"\nTuning on: {image_path}")
            session = run_quantify_wizard(image_path)
            if session is None:
                print(f"  Skipped (cancelled): {image_path}")
                all_rows.append(
                    {
                        "image": str(image_path),
                        "circle_id": "TOTAL",
                        "x": "",
                        "y": "",
                        "r": "",
                        "glc_area": "",
                        "circle_area": "",
                        "coverage_ratio": "",
                        "status": "cancelled",
                        "error": "wizard cancelled",
                    }
                )
                continue
            apply_skips = True
        else:
            assert shared_session is not None
            session = shared_session
            apply_skips = image_path.resolve() == tuned_image

        image_output_dir = results_root / image_path.stem
        try:
            rows, _mean = process_image(
                image_path,
                image_output_dir,
                session,
                apply_skips=apply_skips,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch running
            print(f"\n{image_path}")
            print(f"  ERROR: {exc}")
            all_rows.append(
                {
                    "image": str(image_path),
                    "circle_id": "TOTAL",
                    "x": "",
                    "y": "",
                    "r": "",
                    "glc_area": "",
                    "circle_area": "",
                    "coverage_ratio": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        all_rows.extend(rows)
        # Accumulate from TOTAL row
        total_row = rows[-1]
        batch_glc += float(total_row["glc_area"])
        batch_area += float(total_row["circle_area"])
        n_circles = len(rows) - 1
        cov_pct = float(total_row["coverage_ratio"]) * 100.0
        print(f"\n{image_path}")
        print(f"  circles kept: {n_circles}")
        print(f"  coverage: {cov_pct:.1f}%")
        print(f"  wrote: {image_output_dir}")

    if len(image_paths) > 1:
        batch_ratio = (batch_glc / batch_area) if batch_area > 0 else 0.0
        all_rows.append(
            {
                "image": "BATCH",
                "circle_id": "TOTAL",
                "x": "",
                "y": "",
                "r": "",
                "glc_area": f"{batch_glc:.2f}",
                "circle_area": f"{batch_area:.2f}",
                "coverage_ratio": f"{batch_ratio:.6f}",
                "status": "ok",
                "error": "",
            }
        )
        print(f"\nBatch coverage: {batch_ratio * 100:.1f}%")

    if not args.no_csv:
        csv_path = results_root / args.csv
        write_report_csv(csv_path, all_rows)
        print(f"\nReport: {csv_path}")


if __name__ == "__main__":
    main()
