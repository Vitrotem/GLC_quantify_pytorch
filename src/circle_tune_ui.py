"""Interactive two-step wizard: circles + levels, then GLC segmentation."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.preprocessing import (
    DEFAULT_INNER_OFFSET_PCT,
    CircleDetectParams,
    LevelsParams,
    SegParams,
    SINGLE_HOUGH_MIN_DIST_FACTOR,
    SINGLE_HOUGH_PARAM2,
    apply_levels_params,
    build_circle_mask,
    crop_metadata,
    default_single_circle_radii,
    detect_circles,
    filter_skipped_circles,
    inset_circles,
    intensity_histogram,
    matches_skip_center,
    render_segmentation_overlay,
    segment_glc,
)

PREVIEW_MAX_DIM = 900
SEG_PREVIEW_MAX_DIM = 520
HIST_WIDTH = 256
HIST_HEIGHT = 80
DETECT_DEBOUNCE_MS = 80
LEVELS_DEBOUNCE_MS = 25
SEG_DEBOUNCE_MS = 35
KEEP_COLOR = (0, 220, 0)
SKIP_COLOR = (220, 40, 40)
INSET_COLOR = (0, 200, 255)


@dataclass(frozen=True)
class QuantifySession:
    """Settings chosen in the quantify wizard."""

    detect_params: CircleDetectParams
    skipped_centers: tuple[tuple[int, int], ...]
    levels: LevelsParams
    seg: SegParams
    inner_offset_pct: float = DEFAULT_INNER_OFFSET_PCT


def _draw_circles_on_rgb(
    rgb: np.ndarray,
    circles: list[tuple[int, int, int]],
    skipped_centers: list[tuple[int, int]],
    display_scale: float,
    inner_offset_pct: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    thickness = max(1, int(round(2 / display_scale))) if display_scale > 0 else 2
    kept = 0
    skipped = 0
    for x, y, r in circles:
        is_skipped = matches_skip_center(x, y, r, skipped_centers)
        color = SKIP_COLOR if is_skipped else KEEP_COLOR
        cv2.circle(rgb, (x, y), r, color, thickness)
        if is_skipped:
            skipped += 1
            arm = max(3, r // 5)
            cv2.line(rgb, (x - arm, y - arm), (x + arm, y + arm), SKIP_COLOR, thickness)
            cv2.line(rgb, (x - arm, y + arm), (x + arm, y - arm), SKIP_COLOR, thickness)
        else:
            kept += 1
            if inner_offset_pct > 0:
                inset_r = max(1, int(round(r * (1.0 - inner_offset_pct / 100.0))))
                if inset_r < r:
                    cv2.circle(rgb, (x, y), inset_r, INSET_COLOR, thickness)
    return rgb, kept, skipped


def _scale_circles(
    circles: list[tuple[int, int, int]], scale: float
) -> list[tuple[int, int, int]]:
    if scale >= 1.0:
        return list(circles)
    return [
        (
            int(round(x * scale)),
            int(round(y * scale)),
            max(1, int(round(r * scale))),
        )
        for x, y, r in circles
    ]


def _scale_points(
    points: list[tuple[int, int]], scale: float
) -> list[tuple[int, int]]:
    if scale >= 1.0:
        return list(points)
    return [(int(round(x * scale)), int(round(y * scale))) for x, y in points]


def _render_histogram(
    hist: np.ndarray,
    black: float,
    gamma: float,
    white: float,
    width: int = HIST_WIDTH,
    height: int = HIST_HEIGHT,
) -> Image.Image:
    """Draw a compact levels-style histogram with black/mid/white markers."""
    img = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    peak = float(hist.max()) if hist.size else 0.0
    if peak <= 0:
        peak = 1.0

    n = len(hist)
    for i, count in enumerate(hist):
        bar_h = int(round((float(count) / peak) * (height - 4)))
        if bar_h <= 0:
            continue
        x0 = int(i * width / n)
        x1 = max(x0 + 1, int((i + 1) * width / n))
        draw.rectangle([x0, height - bar_h, x1 - 1, height - 1], fill=(180, 180, 180))

    def x_for(value: float) -> int:
        return int(round(np.clip(value, 0.0, 255.0) / 255.0 * (width - 1)))

    # Midtone position approximates Photoshop's gamma slider placement.
    mid = black + (white - black) * (0.5 ** max(gamma, 1e-6))
    for value, color in (
        (black, (30, 30, 30)),
        (mid, (120, 120, 120)),
        (white, (230, 230, 230)),
    ):
        x = x_for(value)
        draw.polygon([(x, height - 1), (x - 5, height - 8), (x + 5, height - 8)], fill=color)

    return img


def run_quantify_wizard(image_path: Path) -> QuantifySession | None:
    """Two-step UI: circles+levels → segmentation. Returns session or None if cancelled."""
    gray = np.array(Image.open(image_path).convert("L"))
    cropped, _ = crop_metadata(gray)
    h, w = cropped.shape
    display_scale = min(1.0, PREVIEW_MAX_DIM / max(h, w))
    if display_scale < 1.0:
        preview_gray = cv2.resize(
            cropped,
            (int(w * display_scale), int(h * display_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview_gray = cropped
    seg_scale = min(1.0, SEG_PREVIEW_MAX_DIM / max(preview_gray.shape))
    if seg_scale < 1.0:
        ph, pw = preview_gray.shape
        seg_preview_gray = cv2.resize(
            preview_gray,
            (int(pw * seg_scale), int(ph * seg_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        seg_preview_gray = preview_gray
    # Combined scale from full-res → seg preview (for min_size / circle coords).
    full_to_seg_scale = display_scale * seg_scale
    base_hist = intensity_histogram(cropped)

    detect_defaults = CircleDetectParams()
    levels_defaults = LevelsParams()
    seg_defaults = SegParams()

    result: dict[str, QuantifySession | None] = {"value": None}
    skipped_centers: list[tuple[int, int]] = []
    current_circles: list[tuple[int, int, int]] = []
    step = {"n": 1}
    cached_circles: list[tuple[int, int, int]] = []
    cached_detect_key: list[tuple[object, ...] | None] = [None]

    root = tk.Tk()
    root.title(f"Quantify — {image_path.name}")
    root.resizable(True, True)

    main = ttk.Frame(root, padding=8)
    main.pack(fill=tk.BOTH, expand=True)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=0)
    main.rowconfigure(0, weight=1)

    preview_frame = ttk.Frame(main)
    preview_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 8))
    preview_label = ttk.Label(preview_frame)
    preview_label.pack()
    preview_label.configure(cursor="hand2")

    side = ttk.Frame(main)
    side.grid(row=0, column=1, sticky=tk.NS)

    step_var = tk.StringVar(value="Step 1 of 2 — Circles & Levels")
    ttk.Label(side, textvariable=step_var, font=("", 10, "bold")).pack(
        anchor=tk.W, pady=(0, 6)
    )

    status_var = tk.StringVar(value="Circles: 0 kept, 0 skipped")
    ttk.Label(side, textvariable=status_var).pack(anchor=tk.W, pady=(0, 2))
    hint_var = tk.StringVar(
        value="Click a circle to skip or include it\n(red = skipped)"
    )
    ttk.Label(side, textvariable=hint_var, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))

    step1_frame = ttk.Frame(side)
    step1_frame.pack(fill=tk.X)
    step2_frame = ttk.Frame(side)

    # --- Step 1 circle controls ---
    circle_controls = ttk.LabelFrame(step1_frame, text="Circle detection", padding=4)
    circle_controls.pack(fill=tk.X, pady=(0, 8))

    single_mode = {"on": False}
    default_single_min_px, default_single_max_px = default_single_circle_radii(cropped.shape)
    # Pixel slider upper bound: half-diagonal so one FOV circle is in range.
    px_slider_max = max(default_single_max_px * 2, int(min(h, w) * 0.7))

    param1_var = tk.DoubleVar(value=detect_defaults.param1)
    param2_var = tk.DoubleVar(value=detect_defaults.param2)
    min_dist_var = tk.DoubleVar(value=detect_defaults.min_dist_factor)
    min_r_pct_var = tk.DoubleVar(value=detect_defaults.min_radius_frac * 100.0)
    max_r_pct_var = tk.DoubleVar(value=detect_defaults.max_radius_frac * 100.0)
    min_r_px_var = tk.DoubleVar(value=float(default_single_min_px))
    max_r_px_var = tk.DoubleVar(value=float(default_single_max_px))
    inner_offset_var = tk.DoubleVar(value=DEFAULT_INNER_OFFSET_PCT)

    mode_btn = ttk.Button(circle_controls, text="Single circle mode")
    mode_btn.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 6))

    grid_radius_frame = ttk.Frame(circle_controls)
    single_radius_frame = ttk.Frame(circle_controls)
    shared_params_frame = ttk.Frame(circle_controls)
    shared_params_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW)
    grid_radius_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW)
    # single_radius_frame shown only in single mode

    # --- Step 1 levels controls ---
    levels_frame = ttk.LabelFrame(step1_frame, text="Levels", padding=4)
    levels_frame.pack(fill=tk.X)

    black_var = tk.DoubleVar(value=levels_defaults.black)
    gamma_var = tk.DoubleVar(value=levels_defaults.gamma)
    white_var = tk.DoubleVar(value=levels_defaults.white)

    hist_label = ttk.Label(levels_frame)
    hist_label.pack(pady=(0, 6))
    hist_photo_ref: list[ImageTk.PhotoImage | None] = [None]

    # --- Step 2 segmentation controls ---
    seg_controls = ttk.LabelFrame(step2_frame, text="Segmentation", padding=4)
    seg_controls.pack(fill=tk.X)
    threshold_var = tk.DoubleVar(value=seg_defaults.threshold)
    fuzziness_var = tk.DoubleVar(value=seg_defaults.fuzziness)
    min_size_var = tk.DoubleVar(value=float(seg_defaults.min_size))

    value_labels: dict[str, tk.StringVar] = {}
    detect_debounce_id: list[str | None] = [None]
    levels_debounce_id: list[str | None] = [None]
    seg_debounce_id: list[str | None] = [None]
    photo_ref: list[ImageTk.PhotoImage | None] = [None]
    leveled_cache_key: list[tuple[float, float, float] | None] = [None]
    leveled_cache_img: list[np.ndarray | None] = [None]

    def detect_cache_key(params: CircleDetectParams) -> tuple[object, ...]:
        return (
            params.param1,
            params.param2,
            params.min_dist_factor,
            params.min_radius_frac,
            params.max_radius_frac,
            params.single_circle,
            params.min_radius_px,
            params.max_radius_px,
        )

    def current_detect_params() -> CircleDetectParams:
        if single_mode["on"]:
            min_px = int(round(min_r_px_var.get()))
            max_px = int(round(max_r_px_var.get()))
            if max_px <= min_px:
                max_px = min_px + 1
            return CircleDetectParams(
                param1=float(param1_var.get()),
                param2=float(param2_var.get()),
                min_dist_factor=float(min_dist_var.get()),
                min_radius_frac=detect_defaults.min_radius_frac,
                max_radius_frac=detect_defaults.max_radius_frac,
                single_circle=True,
                min_radius_px=min_px,
                max_radius_px=max_px,
            )
        min_frac = min_r_pct_var.get() / 100.0
        max_frac = max_r_pct_var.get() / 100.0
        if max_frac <= min_frac:
            max_frac = min_frac + 0.01
        return CircleDetectParams(
            param1=float(param1_var.get()),
            param2=float(param2_var.get()),
            min_dist_factor=float(min_dist_var.get()),
            min_radius_frac=float(min_frac),
            max_radius_frac=float(max_frac),
            single_circle=False,
        )

    def current_inner_offset() -> float:
        return float(np.clip(inner_offset_var.get(), 0.0, 99.0))

    def current_levels() -> LevelsParams:
        black = float(black_var.get())
        white = float(white_var.get())
        if white <= black:
            white = black + 1.0
        return LevelsParams(
            black=black,
            gamma=max(float(gamma_var.get()), 0.01),
            white=white,
        )

    def current_seg() -> SegParams:
        return SegParams(
            threshold=float(threshold_var.get()),
            fuzziness=max(float(fuzziness_var.get()), 0.0),
            min_size=max(0, int(round(min_size_var.get()))),
        )

    def ensure_leveled_seg(levels: LevelsParams) -> np.ndarray:
        key = (levels.black, levels.gamma, levels.white)
        if leveled_cache_key[0] != key or leveled_cache_img[0] is None:
            leveled_cache_img[0] = apply_levels_params(seg_preview_gray, levels)
            leveled_cache_key[0] = key
        return leveled_cache_img[0]

    def ensure_circles(detect: CircleDetectParams) -> list[tuple[int, int, int]]:
        nonlocal current_circles, cached_circles
        key = detect_cache_key(detect)
        if cached_detect_key[0] != key:
            cached_circles = detect_circles(cropped, params=detect)
            cached_detect_key[0] = key
        current_circles = cached_circles
        return cached_circles

    def refresh_preview(*, redetect: bool = True) -> None:
        nonlocal current_circles
        detect = current_detect_params()
        levels = current_levels()
        if redetect or cached_detect_key[0] is None:
            circles = ensure_circles(detect)
        else:
            circles = cached_circles
            current_circles = circles

        if step["n"] == 1:
            offset = current_inner_offset()
            # Levels + draw only on preview-resolution image (much faster).
            leveled = apply_levels_params(preview_gray, levels)
            rgb = cv2.cvtColor(leveled, cv2.COLOR_GRAY2RGB)
            preview_circles = _scale_circles(circles, display_scale)
            preview_skips = _scale_points(skipped_centers, display_scale)
            rgb, kept, skipped = _draw_circles_on_rgb(
                rgb, preview_circles, preview_skips, 1.0, offset
            )
            preview = Image.fromarray(rgb, mode="RGB")
            status_var.set(f"Circles: {kept} kept, {skipped} skipped")

            hist_img = _render_histogram(base_hist, levels.black, levels.gamma, levels.white)
            hist_photo = ImageTk.PhotoImage(hist_img)
            hist_photo_ref[0] = hist_photo
            hist_label.configure(image=hist_photo)

            value_labels["param1"].set(f"{detect.param1:.0f}")
            value_labels["param2"].set(f"{detect.param2:.0f}")
            if detect.single_circle:
                value_labels["min_r_px"].set(f"{detect.min_radius_px}")
                value_labels["max_r_px"].set(f"{detect.max_radius_px}")
            else:
                value_labels["min_dist"].set(f"{detect.min_dist_factor:.2f}")
                value_labels["min_r"].set(f"{detect.min_radius_frac * 100:.1f}")
                value_labels["max_r"].set(f"{detect.max_radius_frac * 100:.1f}")
            value_labels["inner_offset"].set(f"{offset:.1f}")
            value_labels["black"].set(f"{levels.black:.0f}")
            value_labels["gamma"].set(f"{levels.gamma:.2f}")
            value_labels["white"].set(f"{levels.white:.0f}")
        else:
            # Faster interactive seg: smaller canvas, cached levels, scaled min_size.
            kept_circles = filter_skipped_circles(circles, skipped_centers)
            analysis_circles = inset_circles(kept_circles, current_inner_offset())
            leveled = ensure_leveled_seg(levels)
            seg = current_seg()
            preview_analysis = _scale_circles(analysis_circles, full_to_seg_scale)
            min_size_preview = max(
                0, int(round(seg.min_size * (full_to_seg_scale**2)))
            )
            soft = segment_glc(
                leveled,
                preview_analysis,
                seg.threshold,
                seg.fuzziness,
                min_size=min_size_preview,
            )
            overlay = render_segmentation_overlay(leveled, soft, preview_analysis)
            if seg_scale < 1.0:
                oh, ow = preview_gray.shape
                overlay = overlay.resize((ow, oh), Image.BILINEAR)
            preview = overlay
            mask = build_circle_mask(leveled.shape, preview_analysis) > 0
            area = int(np.count_nonzero(mask))
            cov = float(soft[mask].mean() * 100.0) if area > 0 else 0.0
            status_var.set(
                f"Circles: {len(kept_circles)} | Coverage: {cov:.1f}%"
            )
            value_labels["threshold"].set(f"{seg.threshold:.0f}")
            value_labels["fuzziness"].set(f"{seg.fuzziness:.0f}")
            value_labels["min_size"].set(f"{seg.min_size}")

        photo = ImageTk.PhotoImage(preview)
        photo_ref[0] = photo
        preview_label.configure(image=photo)

    def schedule_detect_refresh(*_args: object) -> None:
        if detect_debounce_id[0] is not None:
            root.after_cancel(detect_debounce_id[0])
        detect_debounce_id[0] = root.after(
            DETECT_DEBOUNCE_MS, lambda: refresh_preview(redetect=True)
        )

    def schedule_levels_refresh(*_args: object) -> None:
        if levels_debounce_id[0] is not None:
            root.after_cancel(levels_debounce_id[0])
        levels_debounce_id[0] = root.after(
            LEVELS_DEBOUNCE_MS, lambda: refresh_preview(redetect=False)
        )

    def schedule_seg_refresh(*_args: object) -> None:
        if seg_debounce_id[0] is not None:
            root.after_cancel(seg_debounce_id[0])
        seg_debounce_id[0] = root.after(
            SEG_DEBOUNCE_MS, lambda: refresh_preview(redetect=False)
        )

    def on_preview_click(event: tk.Event) -> None:
        if step["n"] != 1 or display_scale <= 0 or not current_circles:
            return
        full_x = event.x / display_scale
        full_y = event.y / display_scale

        best: tuple[int, int, int] | None = None
        best_dist = float("inf")
        for x, y, r in current_circles:
            dist = ((full_x - x) ** 2 + (full_y - y) ** 2) ** 0.5
            if dist <= r * 1.15 and dist < best_dist:
                best = (x, y, r)
                best_dist = dist

        if best is None:
            return

        bx, by, br = best
        matched_idx = None
        for i, (sx, sy) in enumerate(skipped_centers):
            if matches_skip_center(bx, by, br, [(sx, sy)]):
                matched_idx = i
                break
        if matched_idx is not None:
            skipped_centers.pop(matched_idx)
        else:
            skipped_centers.append((bx, by))
        refresh_preview(redetect=False)

    def add_slider(
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        key: str,
        resolution: float,
        on_change=None,
    ) -> None:
        if on_change is None:
            on_change = schedule_detect_refresh
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        scale = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            length=220,
            showvalue=False,
            command=lambda _v: on_change(),
        )
        scale.grid(row=row, column=1, sticky=tk.EW, padx=8, pady=2)
        value_var = tk.StringVar()
        value_labels[key] = value_var
        ttk.Label(parent, textvariable=value_var, width=6).grid(
            row=row, column=2, sticky=tk.E, pady=2
        )

    circle_controls.columnconfigure(1, weight=1)
    shared_params_frame.columnconfigure(1, weight=1)
    grid_radius_frame.columnconfigure(1, weight=1)
    single_radius_frame.columnconfigure(1, weight=1)

    add_slider(shared_params_frame, 0, "param1", param1_var, 10, 200, "param1", 1)
    add_slider(shared_params_frame, 1, "param2", param2_var, 5, 100, "param2", 1)
    add_slider(
        shared_params_frame,
        2,
        "inner offset %",
        inner_offset_var,
        0.0,
        40.0,
        "inner_offset",
        0.5,
        on_change=schedule_levels_refresh,
    )
    add_slider(
        grid_radius_frame, 0, "min dist factor", min_dist_var, 1.0, 4.0, "min_dist", 0.05
    )
    add_slider(grid_radius_frame, 1, "min radius %", min_r_pct_var, 1.0, 20.0, "min_r", 0.1)
    add_slider(grid_radius_frame, 2, "max radius %", max_r_pct_var, 2.0, 30.0, "max_r", 0.1)
    add_slider(
        single_radius_frame,
        0,
        "min radius px",
        min_r_px_var,
        10,
        px_slider_max,
        "min_r_px",
        1,
    )
    add_slider(
        single_radius_frame,
        1,
        "max radius px",
        max_r_px_var,
        20,
        px_slider_max,
        "max_r_px",
        1,
    )

    def apply_circle_mode(single: bool) -> None:
        single_mode["on"] = single
        skipped_centers.clear()
        cached_detect_key[0] = None
        if single:
            mode_btn.configure(text="Grid mode (many circles)")
            hint_var.set(
                "Single-circle mode: set min/max radius in pixels\n"
                "(click a false circle to skip it)"
            )
            grid_radius_frame.grid_remove()
            single_radius_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW)
            param2_var.set(SINGLE_HOUGH_PARAM2)
            min_dist_var.set(SINGLE_HOUGH_MIN_DIST_FACTOR)
            min_r_px_var.set(float(default_single_min_px))
            max_r_px_var.set(float(default_single_max_px))
        else:
            mode_btn.configure(text="Single circle mode")
            hint_var.set("Click a circle to skip or include it\n(red = skipped)")
            single_radius_frame.grid_remove()
            grid_radius_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW)
            param1_var.set(detect_defaults.param1)
            param2_var.set(detect_defaults.param2)
            min_dist_var.set(detect_defaults.min_dist_factor)
            min_r_pct_var.set(detect_defaults.min_radius_frac * 100.0)
            max_r_pct_var.set(detect_defaults.max_radius_frac * 100.0)
        schedule_detect_refresh()

    mode_btn.configure(command=lambda: apply_circle_mode(not single_mode["on"]))

    levels_sliders = ttk.Frame(levels_frame)
    levels_sliders.pack(fill=tk.X)
    levels_sliders.columnconfigure(1, weight=1)
    add_slider(
        levels_sliders, 0, "black", black_var, 0, 255, "black", 1,
        on_change=schedule_levels_refresh,
    )
    add_slider(
        levels_sliders, 1, "gamma", gamma_var, 0.01, 3.0, "gamma", 0.01,
        on_change=schedule_levels_refresh,
    )
    add_slider(
        levels_sliders, 2, "white", white_var, 0, 255, "white", 1,
        on_change=schedule_levels_refresh,
    )

    seg_controls.columnconfigure(1, weight=1)
    add_slider(
        seg_controls, 0, "threshold", threshold_var, 0, 255, "threshold", 1,
        on_change=schedule_seg_refresh,
    )
    add_slider(
        seg_controls, 1, "fuzziness", fuzziness_var, 0, 64, "fuzziness", 1,
        on_change=schedule_seg_refresh,
    )
    add_slider(
        seg_controls, 2, "min size px", min_size_var, 0, 5000, "min_size", 1,
        on_change=schedule_seg_refresh,
    )

    buttons = ttk.Frame(side)
    buttons.pack(fill=tk.X, pady=(16, 0))

    def show_step(n: int) -> None:
        step["n"] = n
        if n == 1:
            step_var.set("Step 1 of 2 — Circles & Levels")
            if single_mode["on"]:
                hint_var.set(
                    "Single-circle mode: set min/max radius in pixels\n"
                    "(click a false circle to skip it)"
                )
            else:
                hint_var.set("Click a circle to skip or include it\n(red = skipped)")
            step2_frame.pack_forget()
            step1_frame.pack(fill=tk.X)
            preview_label.configure(cursor="hand2")
            next_btn.configure(text="Next", command=on_next)
            back_btn.pack_forget()
        else:
            step_var.set("Step 2 of 2 — Segmentation")
            hint_var.set(
                "Pink = continuous GLC (darker than threshold)\n"
                "Adjust threshold, fuzziness, and min size"
            )
            step1_frame.pack_forget()
            step2_frame.pack(fill=tk.X)
            preview_label.configure(cursor="")
            next_btn.configure(text="Finish", command=on_finish)
            back_btn.pack(fill=tk.X, pady=(0, 4), before=next_btn)
        refresh_preview(redetect=True)

    def on_reset() -> None:
        if step["n"] == 1:
            if single_mode["on"]:
                param1_var.set(detect_defaults.param1)
                param2_var.set(SINGLE_HOUGH_PARAM2)
                min_dist_var.set(SINGLE_HOUGH_MIN_DIST_FACTOR)
                min_r_px_var.set(float(default_single_min_px))
                max_r_px_var.set(float(default_single_max_px))
            else:
                param1_var.set(detect_defaults.param1)
                param2_var.set(detect_defaults.param2)
                min_dist_var.set(detect_defaults.min_dist_factor)
                min_r_pct_var.set(detect_defaults.min_radius_frac * 100.0)
                max_r_pct_var.set(detect_defaults.max_radius_frac * 100.0)
            black_var.set(levels_defaults.black)
            gamma_var.set(levels_defaults.gamma)
            white_var.set(levels_defaults.white)
            inner_offset_var.set(DEFAULT_INNER_OFFSET_PCT)
            skipped_centers.clear()
            cached_detect_key[0] = None
            leveled_cache_key[0] = None
            leveled_cache_img[0] = None
            refresh_preview(redetect=True)
        else:
            threshold_var.set(seg_defaults.threshold)
            fuzziness_var.set(seg_defaults.fuzziness)
            min_size_var.set(float(seg_defaults.min_size))
            refresh_preview(redetect=False)

    def on_next() -> None:
        show_step(2)

    def on_back() -> None:
        show_step(1)

    def on_finish() -> None:
        result["value"] = QuantifySession(
            detect_params=current_detect_params(),
            skipped_centers=tuple(skipped_centers),
            levels=current_levels(),
            seg=current_seg(),
            inner_offset_pct=current_inner_offset(),
        )
        root.destroy()

    def on_cancel() -> None:
        result["value"] = None
        root.destroy()

    preview_label.bind("<Button-1>", on_preview_click)

    ttk.Button(buttons, text="Reset", command=on_reset).pack(fill=tk.X, pady=(0, 4))
    back_btn = ttk.Button(buttons, text="Back", command=on_back)
    next_btn = ttk.Button(buttons, text="Next", command=on_next)
    next_btn.pack(fill=tk.X, pady=(0, 4))
    ttk.Button(buttons, text="Cancel", command=on_cancel).pack(fill=tk.X)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    show_step(1)
    root.mainloop()
    return result["value"]
