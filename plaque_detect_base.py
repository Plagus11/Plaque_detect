import os
import csv
import cv2
import numpy as np
import openslide

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_6_1_ABeta.ndpi"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes"
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL = 0

# Core tile size (grid you want to cover once)
TILE_SIZE = 1536  # px at LEVEL

# Margin around each tile so plaques near borders are fully visible
MARGIN = 256      # px

MICRONS_PER_PIXEL = 0.2305

# Box size in µm (FULL side length)
BOX_SIZE_UM = 125.0
HALF_BOX_PX = int(round((BOX_SIZE_UM / 2.0) / MICRONS_PER_PIXEL))
FULL_BOX_PX = 2 * HALF_BOX_PX

print(f"Using box size: {BOX_SIZE_UM} µm (~{FULL_BOX_PX} x {FULL_BOX_PX} px)")

# Detector parameters
SAT_THRESH = 40
VAL_THRESH = 40
HUE_RANGE = (4, 35)
MIN_AREA = 800              # min blob area (in pixels) to keep
OVERLAP_THRESH = 0.8
MIN_BROWN_FRACTION = 0.003

# -------------------------
# IOU / MERGE
# -------------------------

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(area1 + area2 - inter_area + 1e-8)
    overlap1 = inter_area / float(area1 + 1e-8)
    overlap2 = inter_area / float(area2 + 1e-8)
    return iou, overlap1, overlap2

def merge_overlapping_boxes(boxes, mask, overlap_thresh=0.8, box_size_px=None, img_shape=None):
    if not boxes:
        return []

    H, W = img_shape[:2]
    half_box = box_size_px // 2

    n = len(boxes)
    adj = [[] for _ in range(n)]

    # Build adjacency list
    for i in range(n):
        for j in range(i + 1, n):
            _, overlap1, overlap2 = compute_iou(boxes[i], boxes[j])
            if max(overlap1, overlap2) > overlap_thresh:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components in box graph
    visited = [False] * n
    groups = []

    for i in range(n):
        if not visited[i]:
            stack = [i]
            group = []
            visited[i] = True
            while stack:
                k = stack.pop()
                group.append(k)
                for nb in adj[k]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            groups.append(group)

    merged_boxes = []

    for group in groups:
        if len(group) == 1:
            merged_boxes.append(boxes[group[0]])
            continue

        xs1 = [boxes[idx][0] for idx in group]
        ys1 = [boxes[idx][1] for idx in group]
        xs2 = [boxes[idx][2] for idx in group]
        ys2 = [boxes[idx][3] for idx in group]

        ux1 = max(0, min(xs1))
        uy1 = max(0, min(ys1))
        ux2 = min(W - 1, max(xs2))
        uy2 = min(H - 1, max(ys2))

        patch = mask[uy1:uy2, ux1:ux2]
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            cx = (ux1 + ux2) // 2
            cy = (uy1 + uy2) // 2
        else:
            cx_patch = xs.mean()
            cy_patch = ys.mean()
            cx = int(ux1 + cx_patch)
            cy = int(uy1 + cy_patch)

        x1 = cx - half_box
        y1 = cy - half_box
        x2 = cx + half_box
        y2 = cy + half_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 > x1 and y2 > y1:
            merged_boxes.append((x1, y1, x2, y2))

    return merged_boxes

# -------------------------
# Normalisation
# -------------------------

def normalize_tile_bgr(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_raw, s_raw, v_raw = cv2.split(img_hsv)

    bg_mask = (s_raw < 25) & (v_raw > 150)

    if np.any(bg_mask):
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(img_lab)

        mean_A = np.mean(A[bg_mask])
        mean_B = np.mean(B[bg_mask])

        A = np.clip(A - (mean_A - 128), 0, 255).astype(np.uint8)
        B = np.clip(B - (mean_B - 128), 0, 255).astype(np.uint8)

        img_lab_balanced = cv2.merge([L, A, B])
        img_bgr = cv2.cvtColor(img_lab_balanced, cv2.COLOR_LAB2BGR)

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    v_norm = cv2.normalize(v_eq, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    img_hsv_norm = cv2.merge([h, s, v_norm])
    img_bgr_norm = cv2.cvtColor(img_hsv_norm, cv2.COLOR_HSV2BGR)

    return img_bgr_norm

# -------------------------
# Detector: solid mask + fixed-size boxes (BOX_SIZE_UM)
# -------------------------

def detect_brown_regions_fixed_boxes_array(
    img_bgr,
    sat_thresh=40,
    val_thresh=40,
    hue_range=(4, 35),
    min_area=800,
    overlap_thresh=0.8,
    min_brown_fraction=0.003
):
    H, W = img_bgr.shape[:2]

    # 1. Normalise
    img_bgr_norm = normalize_tile_bgr(img_bgr)

    # 2. Brown mask
    img_hsv = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    h_min, h_max = hue_range
    mask_h = cv2.inRange(h, h_min, h_max)
    mask_s = cv2.inRange(s, sat_thresh, 255)
    mask_v = cv2.inRange(v, val_thresh, 255)

    mask_brown = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))

    kernel = np.ones((3, 3), np.uint8)
    brown_clean = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel, iterations=1)
    brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Approximate outer boundary and fill interior (solid plaques)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    brown_closed = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(brown_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_filled = np.zeros_like(brown_closed)
    cv2.drawContours(brown_filled, contours, -1, 255, thickness=cv2.FILLED)

    # 4. Connected components on filled mask -> centroids -> fixed-size boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(brown_filled, connectivity=8)

    initial_boxes = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[i]
        cx = int(cx)
        cy = int(cy)

        x1 = cx - HALF_BOX_PX
        y1 = cy - HALF_BOX_PX
        x2 = cx + HALF_BOX_PX
        y2 = cy + HALF_BOX_PX

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 > x1 and y2 > y1:
            initial_boxes.append((x1, y1, x2, y2))

    # 5. Merge overlapping boxes
    merged_boxes = merge_overlapping_boxes(
        initial_boxes,
        brown_filled,
        overlap_thresh=overlap_thresh,
        box_size_px=FULL_BOX_PX,  # full side length in px
        img_shape=img_bgr_norm.shape
    )

    # 6. Filter by brown fraction in each box
    filtered_boxes = []
    for (x1, y1, x2, y2) in merged_boxes:
        patch = brown_filled[y1:y2, x1:x2]
        total_pixels = patch.size
        brown_pixels = np.count_nonzero(patch > 0)
        brown_fraction = brown_pixels / float(total_pixels + 1e-8)

        if brown_fraction >= min_brown_fraction:
            filtered_boxes.append((x1, y1, x2, y2))

    return brown_filled, filtered_boxes

# -------------------------
# WSI TILING WITH MARGIN (NO CUT PLAQUES)
# -------------------------

def process_wsi_tiled_no_cut(wsi_path, out_dir):
    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    W, H = slide.level_dimensions[LEVEL]
    print(f"WSI: {wsi_name}, level {LEVEL}, size: {W} x {H}")

    all_boxes = []

    for y0 in range(0, H, TILE_SIZE):
        for x0 in range(0, W, TILE_SIZE):

            core_w = min(TILE_SIZE, W - x0)
            core_h = min(TILE_SIZE, H - y0)
            if core_w <= 0 or core_h <= 0:
                continue

            # Expanded tile with margin
            x_exp = max(0, x0 - MARGIN)
            y_exp = max(0, y0 - MARGIN)
            x_exp_end = min(W, x0 + core_w + MARGIN)
            y_exp_end = min(H, y0 + core_h + MARGIN)

            exp_w = x_exp_end - x_exp
            exp_h = y_exp_end - y_exp
            if exp_w <= 0 or exp_h <= 0:
                continue

            downsample = slide.level_downsamples[LEVEL]
            x_exp_0 = int(x_exp * downsample)
            y_exp_0 = int(y_exp * downsample)

            region = slide.read_region((x_exp_0, y_exp_0), LEVEL, (exp_w, exp_h))
            region = region.convert("RGB")
            tile_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

            # Detect on expanded tile
            _, boxes_local = detect_brown_regions_fixed_boxes_array(
                tile_bgr,
                sat_thresh=SAT_THRESH,
                val_thresh=VAL_THRESH,
                hue_range=HUE_RANGE,
                min_area=MIN_AREA,
                overlap_thresh=OVERLAP_THRESH,
                min_brown_fraction=MIN_BROWN_FRACTION
            )

            # Keep only boxes whose centres lie inside the core tile (global coords)
            for (lx1, ly1, lx2, ly2) in boxes_local:
                cx_local = (lx1 + lx2) // 2
                cy_local = (ly1 + ly2) // 2

                gx1 = x_exp + lx1
                gy1 = y_exp + ly1
                gx2 = x_exp + lx2
                gy2 = y_exp + ly2
                gcx = x_exp + cx_local
                gcy = y_exp + cy_local

                if (gcx >= x0 and gcx < x0 + core_w and
                    gcy >= y0 and gcy < y0 + core_h):
                    all_boxes.append((gx1, gy1, gx2, gy2))

    print(f"Collected {len(all_boxes)} boxes before global merge")

    # Global merge to reduce overlaps further
    dummy_mask = np.zeros((H, W), dtype=np.uint8)

    globally_merged_boxes = merge_overlapping_boxes(
        all_boxes,
        dummy_mask,
        overlap_thresh=OVERLAP_THRESH,
        box_size_px=FULL_BOX_PX,
        img_shape=(H, W, 3)
    )

    print(f"{len(globally_merged_boxes)} boxes after global merge")

    # Save to CSV
    csv_path = os.path.join(out_dir, f"{wsi_name}_plaque_boxes_no_cut_{int(BOX_SIZE_UM)}um.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2"])
        for (x1, y1, x2, y2) in globally_merged_boxes:
            writer.writerow([wsi_name, LEVEL, x1, y1, x2, y2])

    print(f"Saved boxes to {csv_path}")

if __name__ == "__main__":
    process_wsi_tiled_no_cut(WSI_PATH, OUT_DIR)
#/Users/savin/PycharmProjects/PythonProject3/wsi_tile_data_1536_nobg/Case_1_1_ABeta_L0_x23040_y30720_1536.png

#/Users/savin/PycharmProjects/PythonProject3/wsi_tile_data_1536_nobg/Case_1_1_ABeta_L0_x29184_y44544_1536.png