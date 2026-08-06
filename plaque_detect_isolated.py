"""
Amyloid-beta plaque detection - FIXED box size, ISOLATED single plaque per crop.

Variant of plaque_detect_b2.py. Detection (brown/DAB HSV gate, normalisation,
morphology) is unchanged. Differences:

  * Fixed box size again (BOX_SIZE_UM) - one tunable knob at the top.
  * NO overlap-merge, so one connected component (one plaque, after the
    morphological close that stitches a plaque's own fragments) = one box.
  * Each saved crop is ISOLATED: the box is cut from the slide, then every
    pixel that is NOT part of *this* plaque's mask is painted white. Background
    cells, stroma, and any neighbouring plaque that happens to fall inside the
    fixed box are removed - only the single target plaque remains.
  * Crops are always exactly FULL_BOX_PX x FULL_BOX_PX (white-padded at slide
    edges), so they drop straight into a classifier.

The plaque's own outline (filled contour), optionally dilated by MASK_DILATE_UM,
defines what is kept - so the whole plaque body is preserved, only its
surroundings are whitened.

Outputs:
  * <wsi>_plaque_boxes_isolated.csv  (wsi_name, level, x1, y1, x2, y2, box_um)
    - same coordinate columns your convert_Qupath.py expects.
  * a folder of isolated PNG crops, one per plaque.
"""

import os
import csv
import time
import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_6_1_ABeta.ndpi"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes_isolated"
CROP_DIR = os.path.join(OUT_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

LEVEL = 0
TILE_SIZE = 1536

MICRONS_PER_PIXEL = 0.2305

# --- Fixed box + isolation -------------------------------------------------
BOX_SIZE_UM = 125.0        # <-- the fixed box size; change this to vary it
MASK_DILATE_UM = 4.0       # grow each plaque mask a touch so the cut isn't razor-tight
WHITE = (255, 255, 255)    # background fill colour (BGR)
SAVE_CROPS = True          # write the isolated PNGs
MAX_CROPS = None           # e.g. 500 to cap output while tuning; None = all
# --------------------------------------------------------------------------

HALF_BOX_PX = int(round((BOX_SIZE_UM / 2.0) / MICRONS_PER_PIXEL))
FULL_BOX_PX = 2 * HALF_BOX_PX
MASK_DILATE_PX = max(0, int(round(MASK_DILATE_UM / MICRONS_PER_PIXEL)))

# Auto MARGIN: the fixed box around a core-centroid plaque must fit in the
# expanded tile read.
MARGIN = max(256, HALF_BOX_PX + MASK_DILATE_PX + 32)

print(f"Isolated crops: box {BOX_SIZE_UM}um (~{FULL_BOX_PX}px), "
      f"mask dilate {MASK_DILATE_UM}um (~{MASK_DILATE_PX}px), MARGIN={MARGIN}")

# Detector parameters (unchanged from b2)
SAT_THRESH = 40
VAL_THRESH = 40
HUE_RANGE = (4, 35)
MIN_AREA = 700
MIN_BROWN_FRACTION = 0.003

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

TISSUE_MASK_TARGET_PX = 2000
TISSUE_MIN_FRACTION_IN_TILE = 0.01

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_KERN_3 = np.ones((3, 3), np.uint8)
_KERN_5_ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERN_DIL = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
             (2 * MASK_DILATE_PX + 1, 2 * MASK_DILATE_PX + 1))
             if MASK_DILATE_PX > 0 else None)


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

    v_eq = _CLAHE.apply(img_hsv[:, :, 2])
    cv2.normalize(v_eq, v_eq, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_hsv[:, :, 2] = v_eq
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


# -------------------------
# Detector -> per-plaque components
# -------------------------

def detect_plaque_components(img_bgr, sat_thresh=40, val_thresh=40,
                             hue_range=(4, 35), min_area=800,
                             min_brown_fraction=0.003):
    """Return a list of (cx, cy, contour) - one per plaque (connected
    component) that passes the area and brown-density filters. Coordinates are
    in the given tile's pixel space."""
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

    comps = []
    for cnt in contours:
        m = cv2.moments(cnt)
        area = m['m00']
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # brown-density guard against thin slivers
        sub = brown_closed[y:y + h, x:x + w]
        if sub.size == 0 or (cv2.countNonZero(sub) / float(sub.size)) < min_brown_fraction:
            continue
        cx = int(m['m10'] / area)
        cy = int(m['m01'] / area)
        comps.append((cx, cy, cnt))
    return comps


def make_isolated_crop(tile_bgr, cx, cy, cnt):
    """Cut a FULL_BOX_PX square centred on (cx, cy) and white out everything
    that isn't this plaque's (dilated) mask. Returns a FULL_BOX_PX x FULL_BOX_PX
    BGR image, white-padded where the box runs past the tile/slide edge."""
    Ht, Wt = tile_bgr.shape[:2]
    lx1 = cx - HALF_BOX_PX
    ly1 = cy - HALF_BOX_PX

    # This plaque's mask, drawn directly in box coordinates.
    mask = np.zeros((FULL_BOX_PX, FULL_BOX_PX), np.uint8)
    shifted = cnt.copy()
    shifted[:, :, 0] -= lx1
    shifted[:, :, 1] -= ly1
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    if _KERN_DIL is not None:
        mask = cv2.dilate(mask, _KERN_DIL, iterations=1)

    canvas = np.full((FULL_BOX_PX, FULL_BOX_PX, 3), 255, np.uint8)

    # Copy the valid overlap of the box with the tile into the canvas.
    sx1 = max(0, lx1); sy1 = max(0, ly1)
    sx2 = min(Wt, lx1 + FULL_BOX_PX); sy2 = min(Ht, ly1 + FULL_BOX_PX)
    if sx2 > sx1 and sy2 > sy1:
        dx1 = sx1 - lx1; dy1 = sy1 - ly1
        canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = tile_bgr[sy1:sy2, sx1:sx2]

    # White out everything outside this plaque's mask.
    canvas[mask == 0] = WHITE
    return canvas


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
                      tissue_mask, ds_mask_from_l0, wsi_name):
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

    comps = detect_plaque_components(
        tile_bgr, sat_thresh=SAT_THRESH, val_thresh=VAL_THRESH,
        hue_range=HUE_RANGE, min_area=MIN_AREA,
        min_brown_fraction=MIN_BROWN_FRACTION)

    out = []
    for (cx, cy, cnt) in comps:
        gcx = x_exp + cx
        gcy = y_exp + cy
        # dedup: plaque belongs to the tile whose CORE holds its centre
        if not (x0 <= gcx < x0 + core_w and y0 <= gcy < y0 + core_h):
            continue

        gx1 = max(0, gcx - HALF_BOX_PX)
        gy1 = max(0, gcy - HALF_BOX_PX)
        gx2 = min(W, gcx + HALF_BOX_PX)
        gy2 = min(H, gcy + HALF_BOX_PX)

        crop_path = ""
        if SAVE_CROPS:
            crop = make_isolated_crop(tile_bgr, cx, cy, cnt)
            crop_name = f"{wsi_name}_x{gcx - HALF_BOX_PX}_y{gcy - HALF_BOX_PX}_{FULL_BOX_PX}px.png"
            crop_path = os.path.join(CROP_DIR, crop_name)
            cv2.imwrite(crop_path, crop)

        out.append((gx1, gy1, gx2, gy2, crop_path))
    return out


# -------------------------
# Driver
# -------------------------

def process_wsi_isolated(wsi_path, out_dir):
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
                      downsample, tissue_mask, ds_mask_from_l0, wsi_name)
            for (x0, y0, cw, ch) in jobs
        ]
        done = 0
        for fut in as_completed(futures):
            all_boxes.extend(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(futures)} tiles, {len(all_boxes)} plaques so far")
            if MAX_CROPS is not None and len(all_boxes) >= MAX_CROPS:
                pass  # tiles already submitted; we just stop writing below

    if MAX_CROPS is not None:
        all_boxes = all_boxes[:MAX_CROPS]

    print(f"{len(all_boxes)} isolated plaque boxes, elapsed {time.time() - t0:.1f}s")

    csv_path = os.path.join(out_dir, f"{wsi_name}_plaque_boxes_isolated.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2",
                         "box_um", "crop_path"])
        for (x1, y1, x2, y2, crop_path) in all_boxes:
            writer.writerow([wsi_name, LEVEL, x1, y1, x2, y2,
                             BOX_SIZE_UM, crop_path])
    print(f"Saved boxes to {csv_path}")
    if SAVE_CROPS:
        print(f"Saved isolated crops to {CROP_DIR}")
    print(f"Total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    process_wsi_isolated(WSI_PATH, OUT_DIR)
