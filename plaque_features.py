"""
plaque_features - shared plaque measurement engine (source-agnostic).

One place that defines HOW a plaque is measured, so every entry point produces
identical numbers:

  * extract_and_analyze.py : measures each plaque inline during detection
    (mask = the detector's own filled contour; no re-reading).
  * analyze_from_boxes.py  : measures from a boxes CSV by re-reading each region
    from the WSI (mask = recover_mask_from_box on the raw patch).
  * plaque_analysis.py     : measures the saved isolated crops (mask = non-white).

The measurement itself is `measure_plaque(bgr, mask, origin_x, origin_y, ...)`:
given a BGR patch and a binary mask of ONE plaque (in patch coordinates), plus
the patch's top-left position on the slide, it returns size, black-core and
colour-intensity features in global level-0 coordinates.

Region grouping (`summarise_regions`) does proximity clusters (single-linkage,
grid + union-find, no scipy/sklearn) and fixed grid bins.

Only cv2 + numpy required.
"""

import os
import csv
import math

import cv2
import numpy as np


# -------------------------
# Defaults (override per call)
# -------------------------

DEFAULTS = dict(
    mpp=0.2305,
    hue=(4, 35),
    sat=40,
    val=40,
    dark_gray=70,           # <= this grey level counts as dark/"black" core
    min_black_frac=0.02,    # cored if dark >= 2% of plaque area ...
    min_black_area_um2=15.0,  # ... OR dark area >= this
)

_KERN_3 = np.ones((3, 3), np.uint8)
_KERN_5_ELL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# -------------------------
# Normalisation (same as the detection pipeline)
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
# Mask recovery for a raw WSI/box patch (used by analyze_from_boxes)
# -------------------------

def recover_mask_from_box(bgr, mpp=None, hue=None, sat=None, val=None,
                          dark_gray=None, normalize=False, min_area_px=1):
    """Recover the ONE plaque a box was drawn around, from a raw patch.
    Mirrors the detector (normalise -> brown gate -> morphology -> fill) and OR's
    in dark pixels so black cores are captured, then keeps the connected
    component nearest the patch centre. Returns a uint8 mask or None."""
    hue = hue if hue is not None else DEFAULTS["hue"]
    sat = sat if sat is not None else DEFAULTS["sat"]
    val = val if val is not None else DEFAULTS["val"]
    dark_gray = dark_gray if dark_gray is not None else DEFAULTS["dark_gray"]

    img = normalize_tile_bgr(bgr) if normalize else bgr
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lo = np.array([hue[0], sat, val], np.uint8)
    hi = np.array([hue[1], 255, 255], np.uint8)
    brown = cv2.inRange(hsv, lo, hi)
    dark = (gray < dark_gray).astype(np.uint8) * 255
    combined = cv2.bitwise_or(brown, dark)

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, _KERN_3, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _KERN_5_ELL, iterations=1)

    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(filled, 8)
    if n <= 1:
        return None
    H, W = bgr.shape[:2]
    pcx, pcy = W / 2.0, H / 2.0
    best, best_d = None, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        d = (cents[i][0] - pcx) ** 2 + (cents[i][1] - pcy) ** 2
        if best is None or d < best_d:
            best, best_d = i, d
    if best is None:
        return None
    return (labels == best).astype(np.uint8) * 255


# -------------------------
# Core measurement
# -------------------------

def measure_plaque(bgr, mask, origin_x=0, origin_y=0, mpp=None,
                   dark_gray=None, min_black_frac=None, min_black_area_um2=None):
    """Measure ONE plaque. `bgr` is a patch, `mask` a uint8 binary mask of the
    plaque in patch coords, `origin_x/y` the patch's top-left on the slide
    (level 0). Returns a feature dict (global coords) or None if the mask is
    empty. Intensity/black are measured on the ORIGINAL pixels inside the mask."""
    mpp = mpp if mpp is not None else DEFAULTS["mpp"]
    dark_gray = dark_gray if dark_gray is not None else DEFAULTS["dark_gray"]
    min_black_frac = min_black_frac if min_black_frac is not None else DEFAULTS["min_black_frac"]
    min_black_area_um2 = (min_black_area_um2 if min_black_area_um2 is not None
                          else DEFAULTS["min_black_area_um2"])
    px2um2 = mpp * mpp

    if mask is None:
        return None
    mbool = mask > 0
    area_px = int(mbool.sum())
    if area_px == 0:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # geometry
    mo = cv2.moments(mask, binaryImage=True)
    cx = mo["m10"] / mo["m00"]
    cy = mo["m01"] / mo["m00"]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    perim_px = cv2.arcLength(cnt, True)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    circularity = float(4 * math.pi * area_px / (perim_px ** 2)) if perim_px > 0 else 0.0
    extent = float(area_px / (bw * bh)) if bw * bh > 0 else 0.0
    aspect = float(max(bw, bh) / max(1, min(bw, bh)))
    area_um2 = area_px * px2um2
    diam_um = 2.0 * math.sqrt(area_px / math.pi) * mpp

    # colour / intensity inside the plaque
    g_in = gray[mbool].astype(np.float32)
    mean_gray = float(g_in.mean())
    mean_od = float((-np.log10((g_in + 1.0) / 256.0)).mean())
    b_in = bgr[:, :, 0][mbool].astype(np.float32)
    g2_in = bgr[:, :, 1][mbool].astype(np.float32)
    r_in = bgr[:, :, 2][mbool].astype(np.float32)
    h_in = hsv[:, :, 0][mbool].astype(np.float32)
    s_in = hsv[:, :, 1][mbool].astype(np.float32)
    v_in = hsv[:, :, 2][mbool].astype(np.float32)

    # black / dark core
    dark = (gray < dark_gray) & mbool
    black_px = int(dark.sum())
    black_area_um2 = black_px * px2um2
    black_frac = float(black_px / area_px)
    has_black = (black_frac >= min_black_frac) or (black_area_um2 >= min_black_area_um2)
    if black_px > 0:
        ys, xs = np.where(dark)
        bcx = float(xs.mean()); bcy = float(ys.mean())
        black_cx_g = origin_x + bcx
        black_cy_g = origin_y + bcy
        black_off = math.hypot(bcx - cx, bcy - cy) * mpp
    else:
        black_cx_g = black_cy_g = black_off = float("nan")

    return {
        "centre_x_px": origin_x + cx,
        "centre_y_px": origin_y + cy,
        "area_px": area_px,
        "area_um2": round(area_um2, 2),
        "diameter_um": round(diam_um, 2),
        "perimeter_um": round(perim_px * mpp, 2),
        "circularity": round(circularity, 4),
        "extent": round(extent, 4),
        "aspect_ratio": round(aspect, 3),
        "bbox_w_um": round(bw * mpp, 2),
        "bbox_h_um": round(bh * mpp, 2),
        "mean_gray": round(mean_gray, 2),
        "mean_optical_density": round(mean_od, 4),
        "mean_hue": round(float(h_in.mean()), 2),
        "mean_saturation": round(float(s_in.mean()), 2),
        "mean_value": round(float(v_in.mean()), 2),
        "mean_B": round(float(b_in.mean()), 2),
        "mean_G": round(float(g2_in.mean()), 2),
        "mean_R": round(float(r_in.mean()), 2),
        "has_black": int(has_black),
        "black_area_um2": round(black_area_um2, 2),
        "black_fraction": round(black_frac, 4),
        "black_centre_x_px": black_cx_g,
        "black_centre_y_px": black_cy_g,
        "black_offset_um": (round(black_off, 2) if not math.isnan(black_off) else ""),
    }


# -------------------------
# Region grouping
# -------------------------

def cluster_by_proximity(centres_px, eps_px):
    """Single-linkage clustering: points within eps_px join one region.
    Grid-hash + union-find. Returns an int array of region ids."""
    n = len(centres_px)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell = max(1.0, eps_px)
    grid = {}
    for i, (x, y) in enumerate(centres_px):
        grid.setdefault((int(x // cell), int(y // cell)), []).append(i)
    eps2 = eps_px * eps_px
    for i, (x, y) in enumerate(centres_px):
        gx, gy = int(x // cell), int(y // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j <= i:
                        continue
                    if (x - centres_px[j][0]) ** 2 + (y - centres_px[j][1]) ** 2 <= eps2:
                        ra, rb = find(i), find(j)
                        if ra != rb:
                            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap, ids = {}, np.empty(n, dtype=int)
    for i, r in enumerate(roots):
        ids[i] = remap.setdefault(r, len(remap))
    return ids


def _mean_nn_spacing_um(pts_px, mpp):
    k = len(pts_px)
    if k < 2:
        return float("nan")
    p = np.asarray(pts_px, dtype=np.float64)
    if k > 3000:
        idx = np.random.default_rng(0).choice(k, 3000, replace=False)
        p = p[idx]; k = len(p)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(1)).mean() * mpp)


def summarise_regions(rows, mpp=None, eps_um=300.0, min_plaques=3, grid_um=1000.0):
    """rows: list of per-plaque feature dicts (each needs centre_x_px/centre_y_px,
    area_um2, mean_optical_density, mean_gray, has_black). Mutates rows to add
    region_id. Returns (cluster_rows, grid_rows)."""
    mpp = mpp if mpp is not None else DEFAULTS["mpp"]
    px2um2 = mpp * mpp
    if not rows:
        return [], []
    centres = np.array([[r["centre_x_px"], r["centre_y_px"]] for r in rows], float)
    area = np.array([r["area_um2"] for r in rows], float)
    od = np.array([r["mean_optical_density"] for r in rows], float)
    gray = np.array([r["mean_gray"] for r in rows], float)
    cored = np.array([r["has_black"] for r in rows], float)

    region_id = cluster_by_proximity(centres, eps_um / mpp)
    for r, rid in zip(rows, region_id):
        r["region_id"] = int(rid)

    cluster_rows = []
    for rid in sorted(set(region_id.tolist())):
        sel = region_id == rid
        pts = centres[sel]
        n = int(sel.sum())
        cxy = pts.mean(0)
        if n >= 3:
            hull = cv2.convexHull(pts.astype(np.float32))
            hull_um2 = float(cv2.contourArea(hull)) * px2um2
        else:
            hull_um2 = 0.0
        density = (n / (hull_um2 / 1e6)) if hull_um2 > 0 else float("nan")
        cluster_rows.append({
            "region_id": int(rid),
            "n_plaques": n,
            "is_region": int(n >= min_plaques),
            "centroid_x_px": round(float(cxy[0]), 1),
            "centroid_y_px": round(float(cxy[1]), 1),
            "hull_area_mm2": round(hull_um2 / 1e6, 4),
            "density_per_mm2": (round(density, 2) if not math.isnan(density) else ""),
            "mean_area_um2": round(float(area[sel].mean()), 2),
            "median_area_um2": round(float(np.median(area[sel])), 2),
            "frac_cored": round(float(cored[sel].mean()), 3),
            "mean_optical_density": round(float(od[sel].mean()), 4),
            "mean_gray": round(float(gray[sel].mean()), 2),
            "mean_nn_spacing_um": (round(_mean_nn_spacing_um(pts, mpp), 2) if n >= 2 else ""),
        })

    grid_px = grid_um / mpp
    bin_area_mm2 = (grid_um / 1000.0) ** 2
    bins = {}
    for i in range(len(rows)):
        bins.setdefault((int(centres[i, 0] // grid_px), int(centres[i, 1] // grid_px)), []).append(i)
    grid_rows = []
    for (bxi, byi), idxs in sorted(bins.items()):
        idxs = np.array(idxs)
        grid_rows.append({
            "bin_x": bxi, "bin_y": byi,
            "x0_px": int(bxi * grid_px), "y0_px": int(byi * grid_px),
            "n_plaques": int(len(idxs)),
            "density_per_mm2": round(len(idxs) / bin_area_mm2, 2),
            "mean_area_um2": round(float(area[idxs].mean()), 2),
            "frac_cored": round(float(cored[idxs].mean()), 3),
            "mean_optical_density": round(float(od[idxs].mean()), 4),
        })
    return cluster_rows, grid_rows


# -------------------------
# CSV
# -------------------------

def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")
