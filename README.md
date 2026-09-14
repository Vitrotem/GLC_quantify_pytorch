# vitrotem_grid_ai

Quantify GLC (graphene-like carbon) coverage inside TEM grid circles using classical image processing — no machine learning.

## Setup

Requires [Poetry](https://python-poetry.org/) (Python 3.11+).

```bash
poetry install
```

## Convert folder images to JPG

```bash
to_jpg path/to/folder/
```

Recursively converts `.png`, `.tif`, `.tiff`, `.jpeg`, `.bmp`, `.webp`, `.gif` to max-quality `.jpg` (quality 100) in place and removes the originals. Existing `.jpg` files are left alone unless you pass `--overwrite`.

## Quantify

```bash
quantify path/to/image.jpg
quantify path/to/folder/
```

Opens an interactive two-step wizard on the **first** image:

1. **Circles + Levels** — tune Hough circle detection, skip false circles, adjust input levels (black / gamma / white). Click **Next**.
2. **Segmentation** — adjust **threshold** and **fuzziness**; pink overlay marks darker-than-threshold GLC within kept circles. Click **Finish**.

Those settings are then applied to every image in the folder. Use `--per-image` to run the wizard for each image instead. Use `--no-ui` to skip the wizard and use defaults.

### Outputs

Passing a folder writes into that folder:

- `quantification_report.csv` — per-circle and TOTAL coverage ratios
- `<image_stem>/overlay.jpg` — full-grid pink overlay + circle outlines
- `<image_stem>/circles/00001.jpg` — levels-adjusted circular crops
- `<image_stem>/circles/00001_overlay.jpg` — crops with pink GLC overlay

For a single file, results go under `--output-dir` (default `outputs/quantify`).

Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

## How it works

1. Detect grid circles (Hough + filters); optionally skip false positives
2. Apply Photoshop-style input levels to the grid
3. Soft-threshold darker pixels inside each circle (threshold + fuzziness)
4. Report coverage = sum of soft membership / circle area; export crops and overlays
