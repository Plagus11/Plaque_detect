"""
Amyloid-beta plaque detection - VARIABLE box sizes, one plaque per box.

This is a variant of plaque_detect_b2.py. The detection (brown/DAB HSV gate,
normalisation, morphology) is unchanged; what changes is how boxes are made:

  b2.py            : every blob -> the SAME fixed FULL_BOX_PX square, centred on
                     the centroid, then OVERLAPPING boxes are merged. A fixed box
                     can straddle two nearby plaques, and the merge can pull
                     several plaques into one box.

  this script      : every qualifying connected component (one plaque, after the
                     morphological close that stitches a plaque's own fragments)
                     becomes ONE box = its own bounding rectangle + a little
                     padding. Box size therefore varies with the plaque, and each
                     box contains exactly one plaque. No overlap-merge step, so
                     two distinct plaques never collapse into a single box.

Cross-tile handling: a plaque is assigned to the tile whose CORE contains its
centroid (unchanged dedup rule). MARGIN is auto-sized from the largest allowed
box so a plaque sitting on a tile seam is still read whole in the expanded tile,
which means its bounding box is complete - no clipped or duplicated seam boxes.

Output CSV keeps the columns your convert_Qupath.py expects (wsi_name, level,
x1, y1, x2, y2) and adds width/height in px and um for convenience.
"""

import os
import csv
import time
import threading   # <-- add this
import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_1_1_ABeta.ndpi"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes_variable"
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL = 0
TILE_SIZE = 1536

MICRONS_PER_PIXEL = 0.2305

# --- Variable-box shaping -------------------------------------------------
# Padding added on every side of a plaque's tight bounding box.
PAD_UM = 8.0
# Clamp the final box side so specks aren't microscopic and giant merged
# blobs don't produce enormous crops. Set MAX_BOX_UM large to effectively
# disable the upper clamp.
MIN_BOX_UM = 20.0
MAX_BOX_UM = 300.0
# Square boxes (side = longest edge) are handy if a downstream classifier wants
# square crops. False = tight rectangle = minimal chance of catching a neighbour.
SQUARE_BOXES = False
# -------------------------------------------------------------------------

PAD_PX = int(round(PAD_UM / MICRONS_PER_PIXEL))
MIN_BOX_PX = int(round(MIN_BOX_UM / MICRONS_PER_PIXEL))
MAX_BOX_PX = int(round(MAX_BOX_UM / MICRONS_PER_PIXEL))

# Auto MARGIN: must exceed the largest half-box so a plaque centred at a tile
# core edge is still fully inside the expanded read.
MARGIN = max(256, MAX_BOX_PX // 2 + 32)

print(f"Variable boxes: pad {PAD_UM}um (~{PAD_PX}px), "
      f"clamp [{MIN_BOX_UM}, {MAX_BOX_UM}]um "
      f"(~[{MIN_BOX_PX}, {MAX_BOX_PX}]px), square={SQUARE_BOXES}, MARGIN={MARGIN}")

# Detector parameters (unchanged from b2)
SAT_THRESH = 40
VAL_THRESH = 40
HUE_RANGE = (4, 35)
MIN_AREA = 800
MIN_BROWN_FRACTION = 0.003

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

TISSUE_MASK_TARGET_PX = 2000
TISSUE_MIN_FRACTION_IN_TILE = 0.01

_tls = threading.local()

def _clahe():
    inst = getattr(_tls, "clahe", None)
    if inst is None:
        inst = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _tls.clahe = inst
    return inst

_KERN_3 = np.ones((3, 3), np.uint8)
_KERN_5_ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# -------------------------
# Box shaping helper
# -------------------------

def _shape_box(x, y, w, h, W, H):
    """Turn a tight bounding rect (x,y,w,h) into a padded, clamped box
    (x1,y1,x2,y2). Guarantees one plaque per box: no merging with neighbours."""
    cx = x + w / 2.0
    cy = y + h / 2.0

    if SQUARE_BOXES:
        side = max(w, h) + 2 * PAD_PX
        side = int(min(max(side, MIN_BOX_PX), MAX_BOX_PX))
        bw = bh = side
    else:
        bw = int(min(max(w + 2 * PAD_PX, MIN_BOX_PX), MAX_BOX_PX))
        bh = int(min(max(h + 2 * PAD_PX, MIN_BOX_PX), MAX_BOX_PX))

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = x1 + bw
    y2 = y1 + bh

    # keep inside the image, preserving size where possible
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > W:
        x1 -= (x2 - W); x2 = W
    if y2 > H:
        y1 -= (y2 - H); y2 = H
    x1 = max(0, x1); y1 = max(0, y1)
    return x1, y1, x2, y2


# -------------------------
# Normalisation (unchanged)
# -------------------------

def normalize_tile_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_raw = img_hsv[:, :, 1]
    v_raw = img_hsv[:, :, 2]
    bg_mask = (s_raw < 25) & (v_raw > 150)

    if bg_mask.any():
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        A = img_lab[:, :, 1]
        B = img_lab[:, :, 2]
        mean_A = float(A[bg_mask].mean())
        mean_B = float(B[bg_mask].mean())
        img_lab[:, :, 1] = np.clip(A.astype(np.int16) - int(mean_A - 128), 0, 255).astype(np.uint8)
        img_lab[:, :, 2] = np.clip(B.astype(np.int16) - int(mean_B - 128), 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    v_eq = _clahe().apply(img_hsv[:, :, 2])
    cv2.normalize(v_eq, v_eq, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_hsv[:, :, 2] = v_eq
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


# -------------------------
# Detector (variable boxes)
# -------------------------

def detect_brown_regions_variable_boxes(
    img_bgr,
    sat_thresh=40,
    val_thresh=40,
    hue_range=(4, 35),
    min_area=800,
    min_brown_fraction=0.003,
):
    """Return (brown_filled_mask, boxes). Each box is one plaque's own padded,
    clamped bounding rectangle - sizes vary, exactly one plaque per box."""
    H, W = img_bgr.shape[:2]

    img_bgr_norm = normalize_tile_bgr(img_bgr)
    img_hsv = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2HSV)

    h_min, h_max = hue_range
    lo = np.array([h_min, sat_thresh, val_thresh], dtype=np.uint8)
    hi = np.array([h_max, 255, 255], dtype=np.uint8)
    mask_brown = cv2.inRange(img_hsv, lo, hi)

    brown_clean = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, _KERN_3, iterations=1)
    brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, _KERN_3, iterations=1)
    brown_closed = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, _KERN_5_ELL, iterations=1)

    contours, _ = cv2.findContours(brown_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_filled = np.zeros_like(brown_closed)
    cv2.drawContours(brown_filled, contours, -1, 255, thickness=cv2.FILLED)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # Optional brown-density guard (very tight bboxes are dense; a thin
        # sliver could be sparse - drop those).
        patch = brown_filled[y:y + h, x:x + w]
        if patch.size == 0 or (cv2.countNonZero(patch) / float(patch.size)) < min_brown_fraction:
            continue
        boxes.append(_shape_box(x, y, w, h, W, H))

    return brown_filled, boxes


# -------------------------
# Whole-slide tissue mask (unchanged)
# -------------------------

def build_tissue_mask(slide, target_px=TISSUE_MASK_TARGET_PX):
    n_levels = slide.level_count
    chosen = n_levels - 1
    for l in range(n_levels):
        lw, _ = slide.level_dimensions[l]
        if lw <= target_px * 2:
            chosen = l
            break

    lw, lh = slide.level_dimensions[chosen]
    region = slide.read_region((0, 0), chosen, (lw, lh)).convert("RGB")
    rgb = np.array(region)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    tissue = ((s > 25) | (v < 200)).astype(np.uint8) * 255
    tissue = cv2.dilate(tissue, np.ones((5, 5), np.uint8), iterations=2)
    return tissue, chosen, slide.level_downsamples[chosen]


def tile_has_tissue(tissue_mask, ds_mask_from_l0, x_exp, y_exp, exp_w, exp_h,
                    min_fraction=TISSUE_MIN_FRACTION_IN_TILE):
    Hm, Wm = tissue_mask.shape[:2]
    mx1 = max(0, int(x_exp / ds_mask_from_l0))
    my1 = max(0, int(y_exp / ds_mask_from_l0))
    mx2 = min(Wm, int((x_exp + exp_w) / ds_mask_from_l0) + 1)
    my2 = min(Hm, int((y_exp + exp_h) / ds_mask_from_l0) + 1)
    if mx2 <= mx1 or my2 <= my1:
        return False
    sub = tissue_mask[my1:my2, mx1:mx2]
    if sub.size == 0:
        return False
    return (cv2.countNonZero(sub) / float(sub.size)) >= min_fraction


# -------------------------
# Per-tile worker
# -------------------------

def _process_one_tile(slide, x0, y0, core_w, core_h, W, H, downsample,
                      tissue_mask, ds_mask_from_l0):
    x_exp = max(0, x0 - MARGIN)
    y_exp = max(0, y0 - MARGIN)
    x_exp_end = min(W, x0 + core_w + MARGIN)
    y_exp_end = min(H, y0 + core_h + MARGIN)
    exp_w = x_exp_end - x_exp
    exp_h = y_exp_end - y_exp
    if exp_w <= 0 or exp_h <= 0:
        return []

    if not tile_has_tissue(tissue_mask, ds_mask_from_l0, x_exp, y_exp, exp_w, exp_h):
        return []

    x_exp_0 = int(x_exp * downsample)
    y_exp_0 = int(y_exp * downsample)
    region = slide.read_region((x_exp_0, y_exp_0), LEVEL, (exp_w, exp_h)).convert("RGB")
    tile_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    _, boxes_local = detect_brown_regions_variable_boxes(
        tile_bgr,
        sat_thresh=SAT_THRESH,
        val_thresh=VAL_THRESH,
        hue_range=HUE_RANGE,
        min_area=MIN_AREA,
        min_brown_fraction=MIN_BROWN_FRACTION,
    )

    out = []
    for (lx1, ly1, lx2, ly2) in boxes_local:
        cx_local = (lx1 + lx2) // 2
        cy_local = (ly1 + ly2) // 2
        gcx = x_exp + cx_local
        gcy = y_exp + cy_local
        # assign the plaque to the tile whose CORE holds its centre (dedup)
        if x0 <= gcx < x0 + core_w and y0 <= gcy < y0 + core_h:
            out.append((x_exp + lx1, y_exp + ly1,
                        x_exp + lx2, y_exp + ly2))
    return out


# -------------------------
# Driver
# -------------------------

def process_wsi_variable(wsi_path, out_dir):
    t0 = time.time()
    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    W, H = slide.level_dimensions[LEVEL]
    print(f"WSI: {wsi_name}, level {LEVEL}, size: {W} x {H}")

    print("Building low-res tissue mask...")
    tissue_mask, mask_level, ds_mask_from_l0 = build_tissue_mask(slide)
    print(f"  tissue mask at level {mask_level}, downsample {ds_mask_from_l0:.2f}, "
          f"tissue fraction = {cv2.countNonZero(tissue_mask) / float(tissue_mask.size):.3f}")

    downsample = slide.level_downsamples[LEVEL]

    jobs = []
    for y0 in range(0, H, TILE_SIZE):
        for x0 in range(0, W, TILE_SIZE):
            core_w = min(TILE_SIZE, W - x0)
            core_h = min(TILE_SIZE, H - y0)
            if core_w > 0 and core_h > 0:
                jobs.append((x0, y0, core_w, core_h))

    print(f"Processing {len(jobs)} tiles with {N_WORKERS} threads...")

    all_boxes = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [
            ex.submit(_process_one_tile, slide, x0, y0, cw, ch, W, H,
                      downsample, tissue_mask, ds_mask_from_l0)
            for (x0, y0, cw, ch) in jobs
        ]
        done = 0
        for fut in as_completed(futures):
            all_boxes.extend(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(futures)} tiles, {len(all_boxes)} boxes so far")

    # No overlap-merge: one plaque per box by construction.
    print(f"{len(all_boxes)} plaque boxes (variable size), "
          f"elapsed {time.time() - t0:.1f}s")

    csv_path = os.path.join(out_dir, f"{wsi_name}_plaque_boxes_variable.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2",
                         "width_px", "height_px", "width_um", "height_um"])
        for (x1, y1, x2, y2) in all_boxes:
            w = x2 - x1
            h = y2 - y1
            writer.writerow([wsi_name, LEVEL, x1, y1, x2, y2, w, h,
                             round(w * MICRONS_PER_PIXEL, 1),
                             round(h * MICRONS_PER_PIXEL, 1)])
    print(f"Saved boxes to {csv_path} (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    process_wsi_variable(WSI_PATH, OUT_DIR)