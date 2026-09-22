"""Circle detection, levels, and GLC segmentation for TEM grid images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.constants import GLC_OVERLAY_COLOR, GLC_OVERLAY_OPACITY

METADATA_FRACTION = 0.08
DETECT_MAX_DIM = 1024
SIZE_TOLERANCE = 0.20
SPACING_TOLERANCE = 0.12
GRID_MIN_NEIGHBORS = 2
INTERIOR_CONTRAST_MIN = 3.0
HOUGH_PARAM1 = 56
HOUGH_PARAM2 = 30
HOUGH_MIN_DIST_FACTOR = 1.8
MIN_RADIUS_FRAC = 0.04
MAX_RADIUS_FRAC = 0.12
# Defaults for single-circle (full-FOV) images, as fractions of min(h, w).
SINGLE_MIN_RADIUS_FRAC = 0.25
SINGLE_MAX_RADIUS_FRAC = 0.55
SINGLE_HOUGH_PARAM2 = 20
SINGLE_HOUGH_MIN_DIST_FACTOR = 2.5

DEFAULT_LEVELS_BLACK = 0.0
DEFAULT_LEVELS_GAMMA = 1.0
DEFAULT_LEVELS_WHITE = 255.0
DEFAULT_SEG_THRESHOLD = 128.0
DEFAULT_SEG_FUZZINESS = 0.0  # hard threshold; soft ramp unused in UI
DEFAULT_SEG_MIN_SIZE = 100
DEFAULT_SEG_MAX_SIZE = 0  # 0 = no upper limit
DEFAULT_SEG_GAP_FILL = 1  # closing radius (px); solidify islands without merging much
DEFAULT_SEG_SPLIT = 0  # opening radius (px); break thin bridges between islands
DEFAULT_INNER_OFFSET_PCT = 0.0


@dataclass(frozen=True)
class CircleDetectParams:
    """Tunable Hough circle detection parameters.

    Grid mode uses *min_radius_frac* / *max_radius_frac* (fraction of min side).
    Single-circle mode uses *min_radius_px* / *max_radius_px* (full-resolution pixels)
    and skips grid regularity filters.
    """

    param1: float = HOUGH_PARAM1
    param2: float = HOUGH_PARAM2
    min_dist_factor: float = HOUGH_MIN_DIST_FACTOR
    min_radius_frac: float = MIN_RADIUS_FRAC
    max_radius_frac: float = MAX_RADIUS_FRAC
    single_circle: bool = False
    min_radius_px: int = 0
    max_radius_px: int = 0


@dataclass(frozen=True)
class LevelsParams:
    """Photoshop-style input levels (black / gamma / white)."""

    black: float = DEFAULT_LEVELS_BLACK
    gamma: float = DEFAULT_LEVELS_GAMMA
    white: float = DEFAULT_LEVELS_WHITE


@dataclass(frozen=True)
class SegParams:
    """Soft darker-than-threshold segmentation for island / archipelago GLC.

    *gap_fill*: morphological closing radius (px) — fills holes/gaps inside islands
    *split*: morphological opening radius (px) — breaks thin bridges so islands stay separate
    *min_size* / *max_size*: keep connected components in this area range (px²);
      *max_size* 0 means no upper limit
    """

    threshold: float = DEFAULT_SEG_THRESHOLD
    fuzziness: float = DEFAULT_SEG_FUZZINESS
    min_size: int = DEFAULT_SEG_MIN_SIZE
    max_size: int = DEFAULT_SEG_MAX_SIZE
    gap_fill: int = DEFAULT_SEG_GAP_FILL
    split: int = DEFAULT_SEG_SPLIT


@dataclass(frozen=True)
class CircleCrop:
    """A circular crop with its position in the source image."""

    image: Image.Image
    x: int
    y: int
    r: int


@dataclass(frozen=True)
class CircleCoverage:
    """Coverage stats for one circle."""

    x: int
    y: int
    r: int
    glc_area: float
    circle_area: float
    coverage_ratio: float


def crop_metadata(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Crop the metadata bar; return cropped image and original height."""
    original_h = gray.shape[0]
    crop_h = int(original_h * (1.0 - METADATA_FRACTION))
    return gray[:crop_h, :], original_h


def _filter_by_average_size(
    circles: list[tuple[int, int, int]],
    tolerance: float = SIZE_TOLERANCE,
) -> list[tuple[int, int, int]]:
    """Keep circles whose radius is within *tolerance* of the median size."""
    if len(circles) <= 1:
        return circles

    median_r = float(np.median([r for _, _, r in circles]))
    if median_r <= 0:
        return circles

    lo, hi = median_r * (1.0 - tolerance), median_r * (1.0 + tolerance)
    filtered = [(x, y, r) for x, y, r in circles if lo <= r <= hi]
    return filtered if filtered else circles


def _filter_by_interior_brightness(
    gray: np.ndarray,
    circles: list[tuple[int, int, int]],
    min_contrast: float = INTERIOR_CONTRAST_MIN,
) -> list[tuple[int, int, int]]:
    """Keep circles whose interior is brighter than the surrounding ring."""
    h, w = gray.shape
    filtered: list[tuple[int, int, int]] = []

    for x, y, r in circles:
        if r < 3:
            continue
        inner = np.zeros((h, w), dtype=np.uint8)
        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner, (x, y), max(1, int(r * 0.65)), 255, thickness=-1)
        cv2.circle(ring, (x, y), int(r * 1.35), 255, thickness=-1)
        cv2.circle(ring, (x, y), int(r * 1.05), 0, thickness=-1)

        inner_vals = gray[inner > 0]
        ring_vals = gray[ring > 0]
        if inner_vals.size == 0 or ring_vals.size == 0:
            continue
        contrast = float(inner_vals.mean()) - float(ring_vals.mean())
        if contrast >= min_contrast:
            filtered.append((x, y, r))

    return filtered if filtered else circles


def _median_grid_pitch(circles: list[tuple[int, int, int]]) -> float | None:
    if len(circles) <= 1:
        return None
    centers = np.array([(x, y) for x, y, _ in circles], dtype=np.float32)
    nearest = []
    for i in range(len(centers)):
        deltas = centers - centers[i]
        dists = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
        dists[i] = np.inf
        nearest.append(float(dists.min()))
    return float(np.median(nearest))


def _filter_by_regular_spacing(
    circles: list[tuple[int, int, int]],
    tolerance: float = SPACING_TOLERANCE,
    min_neighbors: int = GRID_MIN_NEIGHBORS,
) -> list[tuple[int, int, int]]:
    """Keep circles that sit on the grid (enough neighbors at the median pitch)."""
    if len(circles) <= 2:
        return circles

    centers = np.array([(x, y) for x, y, _ in circles], dtype=np.float32)
    n = len(centers)
    pitch = _median_grid_pitch(circles)
    if pitch is None or pitch <= 0:
        return circles

    lo, hi = pitch * (1.0 - tolerance), pitch * (1.0 + tolerance)

    def keep_with(min_n: int) -> list[tuple[int, int, int]]:
        kept: list[tuple[int, int, int]] = []
        for i in range(n):
            deltas = centers - centers[i]
            dists = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
            dists[i] = np.inf
            neighbors = int(np.sum((dists >= lo) & (dists <= hi)))
            if neighbors >= min_n:
                kept.append(circles[i])
        return kept

    filtered = keep_with(min_neighbors)
    if len(filtered) < max(3, n // 4):
        filtered = keep_with(max(1, min_neighbors - 1))
    return filtered if filtered else circles


def _run_hough_circles(
    small: np.ndarray,
    min_r: int,
    max_r: int,
    params: CircleDetectParams,
) -> list[tuple[float, float, float]]:
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, int(min_r * params.min_dist_factor)),
        param1=params.param1,
        param2=params.param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in circles[0]]


def default_single_circle_radii(shape: tuple[int, int]) -> tuple[int, int]:
    """Sensible full-res min/max radius for a single large FOV circle."""
    side = min(shape)
    min_px = max(1, int(round(side * SINGLE_MIN_RADIUS_FRAC)))
    max_px = max(min_px + 1, int(round(side * SINGLE_MAX_RADIUS_FRAC)))
    return min_px, max_px


def detect_circles(
    gray: np.ndarray,
    params: CircleDetectParams | None = None,
) -> list[tuple[int, int, int]]:
    """Return list of (x, y, radius) in full-resolution coordinates."""
    if params is None:
        params = CircleDetectParams()

    h, w = gray.shape
    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    if scale < 1.0:
        small = cv2.resize(
            gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        small = gray
        scale = 1.0

    sh, sw = small.shape
    side = min(sh, sw)

    if params.single_circle:
        min_px, max_px = params.min_radius_px, params.max_radius_px
        if min_px <= 0 or max_px <= 0:
            min_px, max_px = default_single_circle_radii((h, w))
        if max_px <= min_px:
            max_px = min_px + 1
        min_r = max(1, int(round(min_px * scale)))
        max_r = max(min_r + 1, int(round(max_px * scale)))
        # Prefer at most one strong hit across the frame.
        single_params = CircleDetectParams(
            param1=params.param1,
            param2=params.param2,
            min_dist_factor=max(params.min_dist_factor, float(max(sh, sw)) / max(min_r, 1)),
            min_radius_frac=params.min_radius_frac,
            max_radius_frac=params.max_radius_frac,
            single_circle=True,
            min_radius_px=min_px,
            max_radius_px=max_px,
        )
        detected = _run_hough_circles(small, min_r, max_r, single_params)
        if not detected:
            detected_raw = _detect_circles_contours(
                small, scale, min_r_small=min_r, max_r_small=max_r
            )
        else:
            inv = 1.0 / scale
            detected_raw = [
                (int(round(x * inv)), int(round(y * inv)), int(round(r * inv)))
                for x, y, r in detected
            ]
        # Keep only circles within the requested full-res radius band; prefer largest.
        banded = [
            (x, y, r)
            for x, y, r in detected_raw
            if min_px <= r <= max_px
        ]
        if not banded:
            banded = detected_raw
        if len(banded) <= 1:
            return banded
        largest = max(banded, key=lambda c: c[2])
        return [largest]

    min_r = max(1, int(side * params.min_radius_frac))
    max_r = max(min_r + 1, int(side * params.max_radius_frac))

    detected = _run_hough_circles(small, min_r, max_r, params)

    if not detected:
        detected_raw = _detect_circles_contours(small, scale)
    else:
        inv = 1.0 / scale
        detected_raw = [
            (int(round(x * inv)), int(round(y * inv)), int(round(r * inv)))
            for x, y, r in detected
        ]

    sized = _filter_by_average_size(detected_raw)
    bright = _filter_by_interior_brightness(gray, sized)
    return _filter_by_regular_spacing(bright)


def _detect_circles_contours(
    small: np.ndarray,
    scale: float,
    min_r_small: int | None = None,
    max_r_small: int | None = None,
) -> list[tuple[int, int, int]]:
    """Fallback: Otsu threshold + circularity filter."""
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sh, sw = small.shape
    side = min(sh, sw)
    if min_r_small is None:
        min_area = np.pi * (side * 0.03) ** 2
    else:
        min_area = np.pi * (max(1, min_r_small) ** 2) * 0.5
    if max_r_small is None:
        max_area = np.pi * (side * 0.14) ** 2
    else:
        max_area = np.pi * (max_r_small**2) * 1.5
    inv = 1.0 / scale
    circles: list[tuple[int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < 0.8:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append((int(round(x * inv)), int(round(y * inv)), int(round(r * inv))))

    return circles


def build_circle_mask(
    shape: tuple[int, int], circles: list[tuple[int, int, int]]
) -> np.ndarray:
    """Build a binary mask (uint8 0/255) covering all detected circles."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(mask, (x, y), r, 255, thickness=-1)
    return mask


def matches_skip_center(
    x: int,
    y: int,
    r: int,
    skip_centers: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """True if (x, y) is close to a previously skipped circle center."""
    thresh = max(10.0, r * 0.6) ** 2
    for sx, sy in skip_centers:
        if (x - sx) ** 2 + (y - sy) ** 2 <= thresh:
            return True
    return False


def filter_skipped_circles(
    circles: list[tuple[int, int, int]],
    skip_centers: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
) -> list[tuple[int, int, int]]:
    """Drop circles whose centers match any skip entry."""
    if not skip_centers:
        return circles
    return [
        (x, y, r)
        for x, y, r in circles
        if not matches_skip_center(x, y, r, skip_centers)
    ]


def inset_circles(
    circles: list[tuple[int, int, int]],
    inner_offset_pct: float,
) -> list[tuple[int, int, int]]:
    """Shrink each circle radius by *inner_offset_pct* percent (inward offset)."""
    pct = float(np.clip(inner_offset_pct, 0.0, 99.0))
    if pct <= 0:
        return list(circles)
    factor = 1.0 - pct / 100.0
    return [(x, y, max(1, int(round(r * factor)))) for x, y, r in circles]


def apply_levels(
    gray: np.ndarray,
    black: float,
    gamma: float,
    white: float,
) -> np.ndarray:
    """Apply Photoshop-style input levels via a 256-entry LUT (fast)."""
    black_f = float(black)
    white_f = float(white)
    # Photoshop Levels: out = v ** (1/gamma). Gamma < 1 darkens midtones.
    gamma_f = max(float(gamma), 1e-6)
    span = max(white_f - black_f, 1e-6)

    xs = np.arange(256, dtype=np.float32)
    v = np.clip((xs - black_f) / span, 0.0, 1.0)
    lut = np.clip(255.0 * np.power(v, 1.0 / gamma_f), 0.0, 255.0).astype(np.uint8)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.LUT(gray, lut)


def apply_levels_params(gray: np.ndarray, levels: LevelsParams) -> np.ndarray:
    return apply_levels(gray, levels.black, levels.gamma, levels.white)


def segment_glc(
    levels_gray: np.ndarray,
    circles: list[tuple[int, int, int]],
    threshold: float,
    fuzziness: float,
    min_size: int = DEFAULT_SEG_MIN_SIZE,
    max_size: int = DEFAULT_SEG_MAX_SIZE,
    gap_fill: int = DEFAULT_SEG_GAP_FILL,
    split: int = DEFAULT_SEG_SPLIT,
) -> np.ndarray:
    """Soft darker-than-threshold membership (0–1), zero outside circles.

    Refines disconnected islands via gap-fill (close), split (open), then
    min/max connected-component size filtering. Work is limited to the
    bounding box of the circles for speed.
    """
    soft = np.zeros(levels_gray.shape, dtype=np.float32)
    if not circles:
        return soft

    h, w = levels_gray.shape
    x0 = max(0, min(x - r for x, y, r in circles))
    y0 = max(0, min(y - r for x, y, r in circles))
    x1 = min(w, max(x + r for x, y, r in circles) + 1)
    y1 = min(h, max(y + r for x, y, r in circles) + 1)
    if x1 <= x0 or y1 <= y0:
        return soft

    roi = levels_gray[y0:y1, x0:x1]
    roi_circles = [(x - x0, y - y0, r) for x, y, r in circles]
    circle_mask = build_circle_mask(roi.shape, roi_circles) > 0

    t = float(threshold)
    f = max(float(fuzziness), 0.0)
    xs = np.arange(256, dtype=np.float32)
    if f <= 0:
        membership_lut = (xs <= t).astype(np.float32)
    else:
        membership_lut = np.clip((t + f - xs) / (2.0 * f), 0.0, 1.0)
    membership = membership_lut[roi]

    soft_roi = np.zeros(roi.shape, dtype=np.float32)
    soft_roi[circle_mask] = membership[circle_mask]
    soft_roi = _refine_islands(
        soft_roi,
        circle_mask,
        min_size=min_size,
        max_size=max_size,
        gap_fill=gap_fill,
        split=split,
    )
    soft[y0:y1, x0:x1] = soft_roi
    return soft


def _odd_kernel(radius_px: int) -> np.ndarray | None:
    """Ellipse structuring element from a radius in pixels; None if radius < 1."""
    r = max(0, int(radius_px))
    if r < 1:
        return None
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _refine_islands(
    soft: np.ndarray,
    circle_mask: np.ndarray,
    min_size: int,
    max_size: int,
    gap_fill: int,
    split: int,
) -> np.ndarray:
    """Morphology + size filter for archipelago / island components."""
    binary = ((soft > 0.5) & circle_mask).astype(np.uint8)
    if not np.any(binary):
        return soft

    close_k = _odd_kernel(gap_fill)
    if close_k is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    open_k = _odd_kernel(split)
    if open_k is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)

    # Fill holes inside each island (external contours only).
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
        filled[~circle_mask] = 0
        binary = filled

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(soft)

    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(0, int(min_size))
    keep = areas >= min_area
    if int(max_size) > 0:
        keep &= areas <= int(max_size)
    keep_ids = np.where(keep)[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(soft)

    keep_mask = np.isin(labels, keep_ids)

    out = soft.copy()
    out[~keep_mask] = 0.0
    # Solidify morphology-filled pixels inside kept islands.
    filled = keep_mask & (out <= 0.0) & circle_mask
    out[filled] = 1.0
    out[~circle_mask] = 0.0
    return out


def segment_glc_params(
    levels_gray: np.ndarray,
    circles: list[tuple[int, int, int]],
    seg: SegParams,
) -> np.ndarray:
    return segment_glc(
        levels_gray,
        circles,
        seg.threshold,
        seg.fuzziness,
        min_size=seg.min_size,
        max_size=seg.max_size,
        gap_fill=seg.gap_fill,
        split=seg.split,
    )


def coverage_stats(
    soft_mask: np.ndarray,
    circles: list[tuple[int, int, int]],
) -> list[CircleCoverage]:
    """Per-circle GLC coverage from a soft membership mask."""
    h, w = soft_mask.shape
    results: list[CircleCoverage] = []
    for x, y, r in circles:
        disk = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disk, (x, y), r, 255, thickness=-1)
        inside = disk > 0
        circle_area = float(np.count_nonzero(inside))
        if circle_area <= 0:
            results.append(
                CircleCoverage(
                    x=x, y=y, r=r, glc_area=0.0, circle_area=0.0, coverage_ratio=0.0
                )
            )
            continue
        glc_area = float(soft_mask[inside].sum())
        results.append(
            CircleCoverage(
                x=x,
                y=y,
                r=r,
                glc_area=glc_area,
                circle_area=circle_area,
                coverage_ratio=glc_area / circle_area,
            )
        )
    return results


def mean_coverage(stats: list[CircleCoverage]) -> float:
    """Area-weighted mean coverage across circles."""
    total_glc = sum(s.glc_area for s in stats)
    total_area = sum(s.circle_area for s in stats)
    if total_area <= 0:
        return 0.0
    return total_glc / total_area


def render_segmentation_overlay(
    levels_gray: np.ndarray,
    soft_mask: np.ndarray,
    circles: list[tuple[int, int, int]] | None = None,
    color: tuple[int, int, int] = GLC_OVERLAY_COLOR,
    opacity: float = GLC_OVERLAY_OPACITY,
) -> Image.Image:
    """Composite pink GLC membership onto levels-adjusted grayscale; optional outlines."""
    rgb = cv2.cvtColor(levels_gray, cv2.COLOR_GRAY2RGB)
    alpha = np.clip(soft_mask.astype(np.float32) * opacity, 0.0, 1.0)
    hits = alpha > 1e-4
    if np.any(hits):
        pink = np.array(color, dtype=np.float32)
        region = rgb[hits].astype(np.float32)
        a = alpha[hits][..., None]
        rgb[hits] = np.clip(region * (1.0 - a) + pink * a, 0.0, 255.0).astype(np.uint8)

    if circles:
        thickness = max(1, int(round(min(levels_gray.shape) / 400)))
        for x, y, r in circles:
            cv2.circle(rgb, (x, y), r, (0, 220, 0), thickness)

    return Image.fromarray(rgb, mode="RGB")


def extract_circle_crops_from_gray(
    gray: np.ndarray,
    circles: list[tuple[int, int, int]],
) -> list[CircleCrop]:
    """Extract masked circular crops from a grayscale image (already levels-adjusted)."""
    h, w = gray.shape
    crops: list[CircleCrop] = []
    for x, y, r in circles:
        x0, x1 = max(0, x - r), min(w, x + r)
        y0, y1 = max(0, y - r), min(h, y + r)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = gray[y0:y1, x0:x1].copy()
        mask = np.zeros(patch.shape, dtype=np.uint8)
        cv2.circle(mask, (x - x0, y - y0), r, 255, thickness=-1)
        masked = np.zeros_like(patch)
        masked[mask > 0] = patch[mask > 0]
        crops.append(
            CircleCrop(
                image=Image.fromarray(masked, mode="L"),
                x=x,
                y=y,
                r=r,
            )
        )
    return crops


def extract_circle_overlay_crops(
    overlay_rgb: Image.Image,
    circles: list[tuple[int, int, int]],
) -> list[CircleCrop]:
    """Extract circular RGB overlay crops matching circle positions."""
    rgb = np.array(overlay_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    crops: list[CircleCrop] = []
    for x, y, r in circles:
        x0, x1 = max(0, x - r), min(w, x + r)
        y0, y1 = max(0, y - r), min(h, y + r)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = rgb[y0:y1, x0:x1].copy()
        mask = np.zeros(patch.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x - x0, y - y0), r, 255, thickness=-1)
        masked = np.zeros_like(patch)
        masked[mask > 0] = patch[mask > 0]
        crops.append(
            CircleCrop(
                image=Image.fromarray(masked, mode="RGB"),
                x=x,
                y=y,
                r=r,
            )
        )
    return crops


def intensity_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """Return a length-*bins* histogram of uint8 intensities."""
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    return hist.astype(np.float64)
