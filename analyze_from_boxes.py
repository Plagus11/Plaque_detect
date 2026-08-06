"""
analyze_from_boxes - run plaque analysis straight from a boxes CSV.

No saved crops needed: this re-reads each box region from the WSI at level 0,
recovers the plaque mask, and measures it with the shared engine
(plaque_features). Works with any of your box CSVs - it auto-detects the columns:

    x1,y1,x2,y2            (e.g. *_plaque_boxes_no_cut / _isolated / _variable)
    x,y,width,height       (QuPath export from convert_Qupath.py)

Outputs (next to the CSV, or OUT_DIR):
    <stem>_features.csv        one row per plaque (size, black core, intensity)
    <stem>_region_clusters.csv proximity-cluster summary
    <stem>_region_grid.csv     fixed-grid summary
"""

import os
import csv
import time

import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed

import plaque_features as pf

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_6_1_ABeta.ndpi"
BOXES_CSV = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes_isolated/Case_6_1_ABeta_plaque_boxes_isolated.csv"
OUT_DIR = None  # None -> alongside BOXES_CSV

LEVEL = 0
MICRONS_PER_PIXEL = 0.2305

# Measurement / grouping knobs (forwarded to the engine)
DARK_GRAY_THRESH = 70
MIN_BLACK_FRACTION = 0.02
MIN_BLACK_AREA_UM2 = 15.0
REGION_EPS_UM = 300.0
REGION_MIN_PLAQUES = 3
GRID_UM = 1000.0

# Read a little context around each box so the plaque is fully captured.
READ_PAD_PX = 8

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# -------------------------
# CSV parsing (flexible columns)
# -------------------------

def load_boxes(csv_path, wsi_name):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if {"x1", "y1", "x2", "y2"} <= cols:
            mode = "corners"
        elif {"x", "y", "width", "height"} <= cols:
            mode = "xywh"
        else:
            raise ValueError(f"CSV needs x1,y1,x2,y2 or x,y,width,height; got {cols}")

        boxes = []
        for row in reader:
            # optional per-row WSI filter
            row_wsi = row.get("wsi_name") or row.get("image") or ""
            row_wsi = os.path.splitext(os.path.basename(row_wsi))[0] if row_wsi else ""
            if row_wsi and wsi_name and row_wsi != wsi_name:
                continue
            if mode == "corners":
                x1 = int(float(row["x1"])); y1 = int(float(row["y1"]))
                x2 = int(float(row["x2"])); y2 = int(float(row["y2"]))
            else:
                x1 = int(float(row["x"])); y1 = int(float(row["y"]))
                x2 = x1 + int(float(row["width"])); y2 = y1 + int(float(row["height"]))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes


# -------------------------
# Per-box worker
# -------------------------

def _measure_box(slide, box, W, H):
    x1, y1, x2, y2 = box
    rx1 = max(0, x1 - READ_PAD_PX); ry1 = max(0, y1 - READ_PAD_PX)
    rx2 = min(W, x2 + READ_PAD_PX); ry2 = min(H, y2 + READ_PAD_PX)
    w, h = rx2 - rx1, ry2 - ry1
    if w <= 0 or h <= 0:
        return None
    region = slide.read_region((rx1, ry1), LEVEL, (w, h)).convert("RGB")
    bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    mask = pf.recover_mask_from_box(
        bgr, mpp=MICRONS_PER_PIXEL, dark_gray=DARK_GRAY_THRESH, normalize=False)
    if mask is None:
        return None
    feat = pf.measure_plaque(
        bgr, mask, origin_x=rx1, origin_y=ry1, mpp=MICRONS_PER_PIXEL,
        dark_gray=DARK_GRAY_THRESH, min_black_frac=MIN_BLACK_FRACTION,
        min_black_area_um2=MIN_BLACK_AREA_UM2)
    if feat is not None:
        feat["box_x1"] = x1; feat["box_y1"] = y1
        feat["box_x2"] = x2; feat["box_y2"] = y2
    return feat


# -------------------------
# Main
# -------------------------

def main():
    t0 = time.time()
    wsi_name = os.path.splitext(os.path.basename(WSI_PATH))[0]
    out_dir = OUT_DIR or os.path.dirname(os.path.abspath(BOXES_CSV))
    stem = os.path.splitext(os.path.basename(BOXES_CSV))[0]
    os.makedirs(out_dir, exist_ok=True)

    boxes = load_boxes(BOXES_CSV, wsi_name)
    print(f"Loaded {len(boxes)} boxes from {BOXES_CSV}")

    slide = openslide.OpenSlide(WSI_PATH)
    W, H = slide.level_dimensions[LEVEL]

    rows = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_measure_box, slide, b, W, H) for b in boxes]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(boxes)} boxes, {len(rows)} measured")
    print(f"{len(rows)} plaques measured (elapsed {time.time() - t0:.1f}s)")

    cluster_rows, grid_rows = pf.summarise_regions(
        rows, mpp=MICRONS_PER_PIXEL, eps_um=REGION_EPS_UM,
        min_plaques=REGION_MIN_PLAQUES, grid_um=GRID_UM)

    pf.write_csv(os.path.join(out_dir, f"{stem}_features.csv"), rows)
    pf.write_csv(os.path.join(out_dir, f"{stem}_region_clusters.csv"), cluster_rows)
    pf.write_csv(os.path.join(out_dir, f"{stem}_region_grid.csv"), grid_rows)

    cored = sum(r["has_black"] for r in rows)
    n_regions = sum(r["is_region"] for r in cluster_rows)
    print(f"Summary: {len(rows)} plaques, {cored} cored "
          f"({100 * cored / max(1, len(rows)):.1f}%), {len(cluster_rows)} clusters "
          f"({n_regions} >= {REGION_MIN_PLAQUES} plaques), {len(grid_rows)} grid bins")


if __name__ == "__main__":
    main()
