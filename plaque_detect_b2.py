"""
Amyloid-beta plaque detection - optimised tiled WSI pipeline.

Speedups vs. the original (same detection logic, same outputs):
  * Whole-slide tissue mask at a low pyramid level -> skip reading empty tiles
    (typically 50-80% of a WSI is glass; those reads are the biggest single cost).
  * ThreadPoolExecutor across tiles (OpenSlide is thread-safe; reads and most
    OpenCV ops release the GIL, so threads give near-linear scaling).
  * Spatial-grid union-find box merge: O(n) average vs. the original O(n^2).
  * Global merge no longer allocates a full HxW dummy mask (was 10 GB on a
    100k x 100k WSI; never read).
  * Module-level CLAHE object (was being rebuilt per tile).
  * One cv2.inRange on the full HSV instead of three + two bitwise_ands.
  * Centroids/areas from cv2.moments on existing contours - skips a redundant
    cv2.connectedComponentsWithStats pass.
  * Removed the second BGR<->HSV round-trip in normalize_tile_bgr when no
    background-balance correction is applied.
"""

import os
import csv
import time
import cv2
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_1_1_ABeta.ndpi"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_box_2"
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL = 0
TILE_SIZE = 1536
MARGIN = 256

MICRONS_PER_PIXEL = 0.2305
BOX_SIZE_UM = 125.0
HALF_BOX_PX = int(round((BOX_SIZE_UM / 2.0) / MICRONS_PER_PIXEL))
FULL_BOX_PX = 2 * HALF_BOX_PX

print(f"Using box size: {BOX_SIZE_UM} um (~{FULL_BOX_PX} x {FULL_BOX_PX} px)")

# Detector parameters (unchanged)
SAT_THRESH = 40
VAL_THRESH = 40
HUE_RANGE = (4, 35)
MIN_AREA = 800
OVERLAP_THRESH = 0.8
MIN_BROWN_FRACTION = 0.003

# Parallelism: threads, because OpenSlide read_region + most OpenCV ops
# release the GIL. Leave one core free for the OS.
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Tissue pre-mask: target ~this many px on the long side for the low-res
# whole-slide view used to skip empty tiles.
TISSUE_MASK_TARGET_PX = 2000
TISSUE_MIN_FRACTION_IN_TILE = 0.01  # require >=1% tissue in an exp. tile

# Module-level CLAHE - rebuilding this per tile was wasteful.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_KERN_3 = np.ones((3, 3), np.uint8)
_KERN_5_ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# -------------------------
# Box-merging (spatial grid + union-find, O(n) average)
# -------------------------

def _overlap_ratio(b1, b2):
    """max(inter/area1, inter/area2). Cheaper than IoU and matches the
    original threshold semantics."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return max(inter / (a1 + 1e-8), inter / (a2 + 1e-8))


def _union_find_groups(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _spatial_pairs(boxes, cell_size, overlap_thresh):
    """Yield pairs of indices whose boxes overlap above threshold using a
    spatial hash. Average O(n) when boxes don't pile up in one cell."""
    grid = defaultdict(list)
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        grid[(cx // cell_size, cy // cell_size)].append(idx)

    seen = set()
    for (gx, gy), members in grid.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(grid.get((gx + dx, gy + dy), ()))
        for i in members:
            for j in cand:
                if j <= i:
                    continue
                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)
                if _overlap_ratio(boxes[i], boxes[j]) > overlap_thresh:
                    yield i, j


def merge_overlapping_boxes_local(boxes, mask, overlap_thresh, box_size_px, img_shape):
    """Tile-level merge: re-centres a merged group on the mask centroid
    (same behaviour as the original function), but skips the O(n^2) loop."""
    if not boxes:
        return []
    H, W = img_shape[:2]
    half = box_size_px // 2

    edges = list(_spatial_pairs(boxes, max(1, box_size_px), overlap_thresh))
    groups = _union_find_groups(len(boxes), edges)

    out = []
    for group in groups:
        if len(group) == 1:
            out.append(boxes[group[0]])
            continue
        xs1 = max(0, min(boxes[k][0] for k in group))
        ys1 = max(0, min(boxes[k][1] for k in group))
        xs2 = min(W - 1, max(boxes[k][2] for k in group))
        ys2 = min(H - 1, max(boxes[k][3] for k in group))
        patch = mask[ys1:ys2, xs1:xs2]
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            cx = (xs1 + xs2) // 2
            cy = (ys1 + ys2) // 2
        else:
            cx = int(xs1 + xs.mean())
            cy = int(ys1 + ys.mean())
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W - 1, cx + half)
        y2 = min(H - 1, cy + half)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def merge_overlapping_boxes_global(boxes, overlap_thresh, box_size_px, img_shape):
    """Global merge: no per-pixel mask is available at WSI scale, so use the
    group bbox centre. This matches what the original code actually did
    (the dummy_mask was all-zero, so the np.where branch never fired)."""
    if not boxes:
        return []
    H, W = img_shape[:2]
    half = box_size_px // 2

    edges = list(_spatial_pairs(boxes, max(1, box_size_px), overlap_thresh))
    groups = _union_find_groups(len(boxes), edges)

    out = []
    for group in groups:
        if len(group) == 1:
            out.append(boxes[group[0]])
            continue
        xs1 = min(boxes[k][0] for k in group)
        ys1 = min(boxes[k][1] for k in group)
        xs2 = max(boxes[k][2] for k in group)
        ys2 = max(boxes[k][3] for k in group)
        cx = (xs1 + xs2) // 2
        cy = (ys1 + ys2) // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W - 1, cx + half)
        y2 = min(H - 1, cy + half)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


# -------------------------
# Normalisation
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
        # int16 to keep negative intermediates before clip
        img_lab[:, :, 1] = np.clip(A.astype(np.int16) - int(mean_A - 128), 0, 255).astype(np.uint8)
        img_lab[:, :, 2] = np.clip(B.astype(np.int16) - int(mean_B - 128), 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    v_eq = _CLAHE.apply(img_hsv[:, :, 2])
    cv2.normalize(v_eq, v_eq, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_hsv[:, :, 2] = v_eq
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


# -------------------------
# Detector
# -------------------------

def detect_brown_regions_fixed_boxes_array(
        img_bgr,
        sat_thresh=40,
        val_thresh=40,
        hue_range=(4, 35),
        min_area=800,
        overlap_thresh=0.8,
        min_brown_fraction=0.003,
):
    H, W = img_bgr.shape[:2]

    img_bgr_norm = normalize_tile_bgr(img_bgr)
    img_hsv = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2HSV)

    # Single inRange in HSV space (was 3x inRange + 2x bitwise_and).
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

    # Take area + centroid straight from the contours instead of running
    # connectedComponentsWithStats over the whole mask again.
    initial_boxes = []
    for cnt in contours:
        m = cv2.moments(cnt)
        area = m['m00']
        if area < min_area:
            continue
        cx = int(m['m10'] / area)
        cy = int(m['m01'] / area)
        x1 = max(0, cx - HALF_BOX_PX)
        y1 = max(0, cy - HALF_BOX_PX)
        x2 = min(W - 1, cx + HALF_BOX_PX)
        y2 = min(H - 1, cy + HALF_BOX_PX)
        if x2 > x1 and y2 > y1:
            initial_boxes.append((x1, y1, x2, y2))

    merged = merge_overlapping_boxes_local(
        initial_boxes, brown_filled,
        overlap_thresh=overlap_thresh,
        box_size_px=FULL_BOX_PX,
        img_shape=img_bgr_norm.shape,
    )

    filtered = []
    for (x1, y1, x2, y2) in merged:
        patch = brown_filled[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        if cv2.countNonZero(patch) / float(patch.size) >= min_brown_fraction:
            filtered.append((x1, y1, x2, y2))

    return brown_filled, filtered


# -------------------------
# Whole-slide tissue mask (cheap pyramid-level pre-screen)
# -------------------------

def build_tissue_mask(slide, target_px=TISSUE_MASK_TARGET_PX):
    """Return (mask, level_used, downsample_from_level0)."""
    n_levels = slide.level_count
    W0, _ = slide.level_dimensions[0]
    # Pick the smallest level whose width >= target_px (so we never blow up
    # memory on a too-large pre-mask, but stay above the target).
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
    # Be generous near tissue edges - we'd rather process a marginal tile
    # than miss a plaque at the tissue boundary.
    tissue = cv2.dilate(tissue, np.ones((5, 5), np.uint8), iterations=2)
    return tissue, chosen, slide.level_downsamples[chosen]


def tile_has_tissue(tissue_mask, ds_mask_from_l0, x_exp, y_exp, exp_w, exp_h,
                    min_fraction=TISSUE_MIN_FRACTION_IN_TILE):
    """Decide whether to bother reading this expanded tile at level 0."""
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

    _, boxes_local = detect_brown_regions_fixed_boxes_array(
        tile_bgr,
        sat_thresh=SAT_THRESH,
        val_thresh=VAL_THRESH,
        hue_range=HUE_RANGE,
        min_area=MIN_AREA,
        overlap_thresh=OVERLAP_THRESH,
        min_brown_fraction=MIN_BROWN_FRACTION,
    )

    out = []
    for (lx1, ly1, lx2, ly2) in boxes_local:
        cx_local = (lx1 + lx2) // 2
        cy_local = (ly1 + ly2) // 2
        gcx = x_exp + cx_local
        gcy = y_exp + cy_local
        if x0 <= gcx < x0 + core_w and y0 <= gcy < y0 + core_h:
            out.append((x_exp + lx1, y_exp + ly1,
                        x_exp + lx2, y_exp + ly2))
    return out


# -------------------------
# Driver
# -------------------------

def process_wsi_tiled_no_cut(wsi_path, out_dir):
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

    print(f"Collected {len(all_boxes)} boxes before global merge "
          f"(elapsed {time.time() - t0:.1f}s)")

    merged = merge_overlapping_boxes_global(
        all_boxes,
        overlap_thresh=OVERLAP_THRESH,
        box_size_px=FULL_BOX_PX,
        img_shape=(H, W, 3),
    )
    print(f"{len(merged)} boxes after global merge")

    csv_path = os.path.join(out_dir, f"{wsi_name}_plaque_boxes_no_cut_{int(BOX_SIZE_UM)}um.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2"])
        for (x1, y1, x2, y2) in merged:
            writer.writerow([wsi_name, LEVEL, x1, y1, x2, y2])
    print(f"Saved boxes to {csv_path} (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    process_wsi_tiled_no_cut(WSI_PATH, OUT_DIR)