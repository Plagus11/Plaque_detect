"""
extract_and_analyze - detect plaques AND measure them in the SAME pass.

This runs alongside (in fact, instead of separately re-running) the extraction:
it reuses the exact detection + isolation from plaque_detect_isolated.py, and as
each plaque is found it measures it inline with the shared engine
(plaque_features) - no re-reading the slide, no dependence on saved PNGs.

All detection settings (WSI_PATH, BOX_SIZE_UM, thresholds, tissue mask, whether
to also SAVE_CROPS, etc.) are inherited from plaque_detect_isolated.py, so this
stays in sync with whatever you've tuned there.

Outputs (in plaque_detect_isolated.OUT_DIR):
    <wsi>_plaque_features.csv        one row per plaque (size, black core, intensity)
    <wsi>_plaque_boxes.csv           box coords (QuPath-compatible x1..y2)
    <wsi>_region_clusters.csv        proximity-cluster summary
    <wsi>_region_grid.csv            fixed-grid summary
    (+ isolated crops if SAVE_CROPS is True in plaque_detect_isolated.py)
"""

import os
import csv
import time

import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed

import plaque_detect_isolated as ext   # detection + all settings
import plaque_features as pf           # measurement engine

# -------------------------
# Analysis knobs (detection knobs come from plaque_detect_isolated)
# -------------------------

DARK_GRAY_THRESH = 70
MIN_BLACK_FRACTION = 0.02
MIN_BLACK_AREA_UM2 = 15.0
REGION_EPS_UM = 300.0
REGION_MIN_PLAQUES = 3
GRID_UM = 1000.0

MPP = ext.MICRONS_PER_PIXEL
_PAD = 2  # small pad around a plaque's contour bbox when measuring


# -------------------------
# Per-tile worker: detect + measure inline
# -------------------------

def _process_tile(slide, wsi_name, x0, y0, core_w, core_h, W, H, downsample,
                  tissue_mask, ds_mask_from_l0):
    x_exp = max(0, x0 - ext.MARGIN)
    y_exp = max(0, y0 - ext.MARGIN)
    x_exp_end = min(W, x0 + core_w + ext.MARGIN)
    y_exp_end = min(H, y0 + core_h + ext.MARGIN)
    exp_w = x_exp_end - x_exp
    exp_h = y_exp_end - y_exp
    if exp_w <= 0 or exp_h <= 0:
        return []
    if not ext.tile_has_tissue(tissue_mask, ds_mask_from_l0, x_exp, y_exp, exp_w, exp_h):
        return []

    region = slide.read_region((int(x_exp * downsample), int(y_exp * downsample)),
                               ext.LEVEL, (exp_w, exp_h)).convert("RGB")
    tile_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    comps = ext.detect_plaque_components(
        tile_bgr, sat_thresh=ext.SAT_THRESH, val_thresh=ext.VAL_THRESH,
        hue_range=ext.HUE_RANGE, min_area=ext.MIN_AREA,
        min_brown_fraction=ext.MIN_BROWN_FRACTION)

    Ht, Wt = tile_bgr.shape[:2]
    out = []
    for (cx, cy, cnt) in comps:
        gcx = x_exp + cx
        gcy = y_exp + cy
        # dedup: keep only plaques whose centre is in this tile's core
        if not (x0 <= gcx < x0 + core_w and y0 <= gcy < y0 + core_h):
            continue

        # measurement mask = this plaque's filled contour (true footprint,
        # no isolation dilation), cropped to its bbox for efficiency.
        bx, by, bw, bh = cv2.boundingRect(cnt)
        bx0 = max(0, bx - _PAD); by0 = max(0, by - _PAD)
        bx1 = min(Wt, bx + bw + _PAD); by1 = min(Ht, by + bh + _PAD)
        sub = tile_bgr[by0:by1, bx0:bx1]
        msk = np.zeros(sub.shape[:2], np.uint8)
        sc = cnt.copy()
        sc[:, :, 0] -= bx0
        sc[:, :, 1] -= by0
        cv2.drawContours(msk, [sc], -1, 255, thickness=cv2.FILLED)

        feat = pf.measure_plaque(
            sub, msk, origin_x=x_exp + bx0, origin_y=y_exp + by0, mpp=MPP,
            dark_gray=DARK_GRAY_THRESH, min_black_frac=MIN_BLACK_FRACTION,
            min_black_area_um2=MIN_BLACK_AREA_UM2)
        if feat is None:
            continue

        gx1 = max(0, gcx - ext.HALF_BOX_PX); gy1 = max(0, gcy - ext.HALF_BOX_PX)
        gx2 = min(W, gcx + ext.HALF_BOX_PX); gy2 = min(H, gcy + ext.HALF_BOX_PX)
        feat["box_x1"] = gx1; feat["box_y1"] = gy1
        feat["box_x2"] = gx2; feat["box_y2"] = gy2

        if ext.SAVE_CROPS:
            crop = ext.make_isolated_crop(tile_bgr, cx, cy, cnt)
            name = f"{wsi_name}_x{gcx - ext.HALF_BOX_PX}_y{gcy - ext.HALF_BOX_PX}_{ext.FULL_BOX_PX}px.png"
            cv2.imwrite(os.path.join(ext.CROP_DIR, name), crop)
            feat["crop_file"] = name

        out.append(feat)
    return out


# -------------------------
# Driver
# -------------------------

def main():
    t0 = time.time()
    slide = openslide.OpenSlide(ext.WSI_PATH)
    wsi_name = os.path.splitext(os.path.basename(ext.WSI_PATH))[0]
    W, H = slide.level_dimensions[ext.LEVEL]
    print(f"WSI: {wsi_name}, level {ext.LEVEL}, size: {W} x {H}")

    print("Building low-res tissue mask...")
    tissue_mask, mask_level, ds_mask_from_l0 = ext.build_tissue_mask(slide)
    downsample = slide.level_downsamples[ext.LEVEL]

    jobs = []
    for y0 in range(0, H, ext.TILE_SIZE):
        for x0 in range(0, W, ext.TILE_SIZE):
            core_w = min(ext.TILE_SIZE, W - x0)
            core_h = min(ext.TILE_SIZE, H - y0)
            if core_w > 0 and core_h > 0:
                jobs.append((x0, y0, core_w, core_h))
    print(f"Processing {len(jobs)} tiles with {ext.N_WORKERS} threads...")

    rows = []
    with ThreadPoolExecutor(max_workers=ext.N_WORKERS) as pool:
        futs = [pool.submit(_process_tile, slide, wsi_name, x0, y0, cw, ch,
                            W, H, downsample, tissue_mask, ds_mask_from_l0)
                for (x0, y0, cw, ch) in jobs]
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(futs)} tiles, {len(rows)} plaques so far")

    print(f"{len(rows)} plaques detected + measured (elapsed {time.time() - t0:.1f}s)")

    cluster_rows, grid_rows = pf.summarise_regions(
        rows, mpp=MPP, eps_um=REGION_EPS_UM,
        min_plaques=REGION_MIN_PLAQUES, grid_um=GRID_UM)

    os.makedirs(ext.OUT_DIR, exist_ok=True)
    pf.write_csv(os.path.join(ext.OUT_DIR, f"{wsi_name}_plaque_features.csv"), rows)
    pf.write_csv(os.path.join(ext.OUT_DIR, f"{wsi_name}_region_clusters.csv"), cluster_rows)
    pf.write_csv(os.path.join(ext.OUT_DIR, f"{wsi_name}_region_grid.csv"), grid_rows)

    # QuPath-compatible boxes CSV
    boxes_path = os.path.join(ext.OUT_DIR, f"{wsi_name}_plaque_boxes.csv")
    with open(boxes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2", "box_um"])
        for r in rows:
            w.writerow([wsi_name, ext.LEVEL, r["box_x1"], r["box_y1"],
                        r["box_x2"], r["box_y2"], ext.BOX_SIZE_UM])
    print(f"  wrote {len(rows)} rows -> {boxes_path}")

    cored = sum(r["has_black"] for r in rows)
    n_regions = sum(r["is_region"] for r in cluster_rows)
    print(f"Summary: {len(rows)} plaques, {cored} cored "
          f"({100 * cored / max(1, len(rows)):.1f}%), {len(cluster_rows)} clusters "
          f"({n_regions} >= {REGION_MIN_PLAQUES} plaques), {len(grid_rows)} grid bins, "
          f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
