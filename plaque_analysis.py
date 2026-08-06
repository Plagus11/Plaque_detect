"""
Plaque image analysis - individual (per-crop) and region-based (grouped).

Input: the isolated plaque crops produced by plaque_detect_isolated.py
(white background, one plaque per PNG, filename encodes the level-0 position:
    <wsi>_x<gx1>_y<gy1>_<FULL>px.png
so every plaque can be placed back on the slide).

For each crop it recovers the true stained plaque footprint (brown body OR dark
core, ignoring the white background and the thin dilation ring the crop adds)
and measures:

  size      : area (px, um^2), equivalent diameter (um), perimeter, circularity,
              bounding box, extent, aspect ratio
  black     : presence of a dark/black core, its area & fraction, the black
              CENTRE, the plaque CENTRE, and the offset between them (um)
  intensity : mean grey level, mean DAB optical density (OD = -log10(I/255);
              higher = darker/stronger stain), mean H/S/V, mean B/G/R
              - all measured inside the plaque only.

Then it groups plaques two ways:
  clusters  : plaques within REGION_EPS_UM of each other form a region
              (single-linkage; no sklearn needed). Per region: count, density,
              size stats, % cored, mean intensity, nearest-neighbour spacing.
  grid      : fixed GRID_UM square bins -> per-bin count / density / means
              (handy for density heatmaps).

Outputs (written next to the crops):
  plaque_features.csv   one row per plaque
  region_clusters.csv   one row per proximity cluster
  region_grid.csv       one row per occupied grid bin
  (optional) region_scatter.png, grid_density.png  if matplotlib is present

Only cv2 + numpy are required. matplotlib is used for the optional plots.
"""

import os
import re
import csv
import math
import glob

import cv2
import numpy as np

# -------------------------
# CONFIG
# -------------------------

CROP_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes_isolated/crops"
OUT_DIR = os.path.dirname(CROP_DIR.rstrip("/"))  # the *_isolated folder

MICRONS_PER_PIXEL = 0.2305

# Stain gates used to recover the plaque footprint from a crop.
HUE_RANGE = (4, 35)
SAT_THRESH = 40
VAL_THRESH = 40
# Dark / "black" core: pixels darker than this grey level (0-255).
DARK_GRAY_THRESH = 70
# A plaque is called "cored" if the dark region is at least this big.
MIN_BLACK_FRACTION = 0.02      # >= 2% of the plaque area, OR
MIN_BLACK_AREA_UM2 = 15.0      # >= this absolute dark area

# Region grouping
REGION_EPS_UM = 300.0          # plaques within this distance join one region
REGION_MIN_PLAQUES = 3         # clusters smaller than this flagged is_region=0
GRID_UM = 1000.0               # grid bin size for the grid summary

MAKE_PLOTS = True

MPP = MICRONS_PER_PIXEL
PX2UM2 = MPP * MPP             # px area -> um^2

_FNAME_RE = re.compile(r"_x(-?\d+)_y(-?\d+)_(\d+)px")


# -------------------------
# Per-crop analysis
# -------------------------

def _plaque_mask(bgr):
    """Recover the stained plaque footprint from an isolated crop: brown body
    OR dark core, excluding the white background. Returns (mask, gray, hsv) or
    (None, ...) if nothing is found."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    non_white = gray < 250
    lo = np.array([HUE_RANGE[0], SAT_THRESH, VAL_THRESH], np.uint8)
    hi = np.array([HUE_RANGE[1], 255, 255], np.uint8)
    brown = cv2.inRange(hsv, lo, hi) > 0
    dark = gray < DARK_GRAY_THRESH

    mask = ((brown | dark) & non_white).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(mask) == 0:
        return None, gray, hsv

    # keep the largest connected component (the plaque)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, gray, hsv
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255
    return mask, gray, hsv


def analyze_crop(path):
    """Return a dict of features for one crop, or None if unreadable/empty."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    m = _FNAME_RE.search(os.path.basename(path))
    gx1, gy1 = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    mask, gray, hsv = _plaque_mask(bgr)
    if mask is None:
        return None
    mbool = mask > 0
    area_px = int(mbool.sum())
    if area_px == 0:
        return None

    # --- geometry ---
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
    area_um2 = area_px * PX2UM2
    diam_um = 2.0 * math.sqrt(area_px / math.pi) * MPP

    # --- colour / intensity (inside the plaque only) ---
    g_in = gray[mbool].astype(np.float32)
    mean_gray = float(g_in.mean())
    mean_od = float((-np.log10((g_in + 1.0) / 256.0)).mean())
    b_in = bgr[:, :, 0][mbool].astype(np.float32)
    g2_in = bgr[:, :, 1][mbool].astype(np.float32)
    r_in = bgr[:, :, 2][mbool].astype(np.float32)
    h_in = hsv[:, :, 0][mbool].astype(np.float32)
    s_in = hsv[:, :, 1][mbool].astype(np.float32)
    v_in = hsv[:, :, 2][mbool].astype(np.float32)

    # --- black / dark core ---
    dark = (gray < DARK_GRAY_THRESH) & mbool
    black_px = int(dark.sum())
    black_area_um2 = black_px * PX2UM2
    black_frac = float(black_px / area_px)
    has_black = (black_frac >= MIN_BLACK_FRACTION) or (black_area_um2 >= MIN_BLACK_AREA_UM2)
    if black_px > 0:
        ys, xs = np.where(dark)
        bcx = float(xs.mean())
        bcy = float(ys.mean())
        black_cx_g = gx1 + bcx
        black_cy_g = gy1 + bcy
        black_offset_um = math.hypot(bcx - cx, bcy - cy) * MPP
    else:
        black_cx_g = black_cy_g = black_offset_um = float("nan")

    return {
        "crop_file": os.path.basename(path),
        "centre_x_px": gx1 + cx,
        "centre_y_px": gy1 + cy,
        "area_px": area_px,
        "area_um2": round(area_um2, 2),
        "diameter_um": round(diam_um, 2),
        "perimeter_um": round(perim_px * MPP, 2),
        "circularity": round(circularity, 4),
        "extent": round(extent, 4),
        "aspect_ratio": round(aspect, 3),
        "bbox_w_um": round(bw * MPP, 2),
        "bbox_h_um": round(bh * MPP, 2),
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
        "black_offset_um": (round(black_offset_um, 2)
                            if not math.isnan(black_offset_um) else ""),
    }


# -------------------------
# Region grouping
# -------------------------

def _cluster_by_proximity(centres_px, eps_px):
    """Single-linkage clustering: plaques within eps_px join one region.
    Grid-hash + union-find, no scipy/sklearn. Returns an array of region ids."""
    n = len(centres_px)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

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
                    ddx = x - centres_px[j][0]
                    ddy = y - centres_px[j][1]
                    if ddx * ddx + ddy * ddy <= eps2:
                        union(i, j)

    roots = [find(i) for i in range(n)]
    remap = {}
    ids = np.empty(n, dtype=int)
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = len(remap)
        ids[i] = remap[r]
    return ids


def _mean_nn_spacing_um(pts_px):
    """Mean nearest-neighbour distance (um) within a set of points."""
    k = len(pts_px)
    if k < 2:
        return float("nan")
    p = np.asarray(pts_px, dtype=np.float64)
    if k > 3000:  # cap the O(k^2) for very large clusters
        idx = np.random.default_rng(0).choice(k, 3000, replace=False)
        p = p[idx]
        k = len(p)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min(1)).mean() * MPP)


def summarise_regions(rows):
    """rows: list of per-plaque feature dicts. Returns (cluster_rows, grid_rows)."""
    if not rows:
        return [], []
    centres = np.array([[r["centre_x_px"], r["centre_y_px"]] for r in rows], float)
    area = np.array([r["area_um2"] for r in rows], float)
    od = np.array([r["mean_optical_density"] for r in rows], float)
    gray = np.array([r["mean_gray"] for r in rows], float)
    cored = np.array([r["has_black"] for r in rows], float)

    # --- proximity clusters ---
    eps_px = REGION_EPS_UM / MPP
    region_id = _cluster_by_proximity(centres, eps_px)
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
            hull_um2 = float(cv2.contourArea(hull)) * PX2UM2
        else:
            hull_um2 = 0.0
        density = (n / (hull_um2 / 1e6)) if hull_um2 > 0 else float("nan")
        cluster_rows.append({
            "region_id": int(rid),
            "n_plaques": n,
            "is_region": int(n >= REGION_MIN_PLAQUES),
            "centroid_x_px": round(float(cxy[0]), 1),
            "centroid_y_px": round(float(cxy[1]), 1),
            "hull_area_mm2": round(hull_um2 / 1e6, 4),
            "density_per_mm2": (round(density, 2) if not math.isnan(density) else ""),
            "mean_area_um2": round(float(area[sel].mean()), 2),
            "median_area_um2": round(float(np.median(area[sel])), 2),
            "frac_cored": round(float(cored[sel].mean()), 3),
            "mean_optical_density": round(float(od[sel].mean()), 4),
            "mean_gray": round(float(gray[sel].mean()), 2),
            "mean_nn_spacing_um": (round(_mean_nn_spacing_um(pts), 2)
                                   if n >= 2 else ""),
        })

    # --- fixed grid bins ---
    grid_px = GRID_UM / MPP
    bin_area_mm2 = (GRID_UM / 1000.0) ** 2
    bins = {}
    for i in range(len(rows)):
        bxi = int(centres[i, 0] // grid_px)
        byi = int(centres[i, 1] // grid_px)
        bins.setdefault((bxi, byi), []).append(i)
    grid_rows = []
    for (bxi, byi), idxs in sorted(bins.items()):
        idxs = np.array(idxs)
        grid_rows.append({
            "bin_x": bxi,
            "bin_y": byi,
            "x0_px": int(bxi * grid_px),
            "y0_px": int(byi * grid_px),
            "n_plaques": int(len(idxs)),
            "density_per_mm2": round(len(idxs) / bin_area_mm2, 2),
            "mean_area_um2": round(float(area[idxs].mean()), 2),
            "frac_cored": round(float(cored[idxs].mean()), 3),
            "mean_optical_density": round(float(od[idxs].mean()), 4),
        })
    return cluster_rows, grid_rows


# -------------------------
# IO + plots
# -------------------------

def _write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def _make_plots(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping plots: {e})")
        return

    xs = np.array([r["centre_x_px"] for r in rows]) * MPP / 1000.0  # mm
    ys = np.array([r["centre_y_px"] for r in rows]) * MPP / 1000.0
    area = np.array([r["area_um2"] for r in rows])
    rid = np.array([r.get("region_id", 0) for r in rows])

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xs, ys, c=rid % 20, cmap="tab20",
                    s=3 + 30 * (area / (area.max() + 1e-9)), alpha=0.8)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Plaques coloured by proximity region (size ~ area)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "region_scatter.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    nb = max(10, int((xs.max() - xs.min()) / (GRID_UM / 1000.0)) + 1)
    hb = ax.hist2d(xs, ys, bins=nb, cmap="magma")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("Plaque density"); fig.colorbar(hb[3], ax=ax, label="plaques / bin")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "grid_density.png"), dpi=130)
    plt.close(fig)
    print(f"  wrote plots -> {out_dir}/region_scatter.png, grid_density.png")


# -------------------------
# Main
# -------------------------

def main():
    crops = sorted(glob.glob(os.path.join(CROP_DIR, "*.png")))
    print(f"Analysing {len(crops)} crops in {CROP_DIR}")

    rows = []
    for i, p in enumerate(crops):
        feat = analyze_crop(p)
        if feat is not None:
            rows.append(feat)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(crops)} analysed")
    print(f"{len(rows)} plaques measured")

    cluster_rows, grid_rows = summarise_regions(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    _write_csv(os.path.join(OUT_DIR, "plaque_features.csv"), rows)
    _write_csv(os.path.join(OUT_DIR, "region_clusters.csv"), cluster_rows)
    _write_csv(os.path.join(OUT_DIR, "region_grid.csv"), grid_rows)

    n_regions = sum(r["is_region"] for r in cluster_rows)
    cored = sum(r["has_black"] for r in rows)
    print(f"Summary: {len(rows)} plaques, {cored} cored ({100*cored/max(1,len(rows)):.1f}%), "
          f"{len(cluster_rows)} clusters ({n_regions} >= {REGION_MIN_PLAQUES} plaques), "
          f"{len(grid_rows)} occupied {int(GRID_UM)}um grid bins")

    if MAKE_PLOTS and rows:
        _make_plots(rows, OUT_DIR)


if __name__ == "__main__":
    main()
