import os
import csv
import cv2
import numpy as np
import openslide

from plaque_detect_base import merge_overlapping_boxes

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_1_1_ABeta.ndpi"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes"
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL = 0  # WSI level to process

# Core tile size (the grid you want to cover once)
TILE_SIZE = 1536  # px at LEVEL

# Margin around each tile so plaques near borders are fully visible
# Rule of thumb: >= 1–2 × expected plaque diameter in pixels
MARGIN = 256  # px

MICRONS_PER_PIXEL = 0.2305
BOX_SIZE_UM = 125.0

SAT_THRESH = 50
VAL_THRESH = 50
HUE_RANGE = (5, 35)
MIN_AREA = 50
OVERLAP_THRESH = 0.8
MIN_PLAQUE_FRACTION = 0.1
MIN_PLAQUE_PIXELS = 150
BLACK_VAL_THRESH = 40

# -------------------------
# MAIN
# -------------------------

def process_wsi_with_overlap(wsi_path, out_dir):
    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]

    W, H = slide.level_dimensions[LEVEL]
    print(f"WSI: {wsi_name}, level {LEVEL}, size: {W} x {H}")

    csv_path = os.path.join(out_dir, f"{wsi_name}_plaque_boxes_overlap.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "level", "x1", "y1", "x2", "y2"])

        # Loop over core tiles (no margin here)
        for y0 in range(0, H, TILE_SIZE):
            for x0 in range(0, W, TILE_SIZE):

                core_w = min(TILE_SIZE, W - x0)
                core_h = min(TILE_SIZE, H - y0)
                if core_w <= 0 or core_h <= 0:
                    continue

                # Expanded tile with margin, clipped to slide bounds
                x_exp = max(0, x0 - MARGIN)
                y_exp = max(0, y0 - MARGIN)
                x_exp_end = min(W, x0 + core_w + MARGIN)
                y_exp_end = min(H, y0 + core_h + MARGIN)

                exp_w = x_exp_end - x_exp
                exp_h = y_exp_end - y_exp
                if exp_w <= 0 or exp_h <= 0:
                    continue

                # Read expanded region from WSI
                downsample = slide.level_downsamples[LEVEL]
                x_exp_0 = int(x_exp * downsample)
                y_exp_0 = int(y_exp * downsample)

                region = slide.read_region((x_exp_0, y_exp_0), LEVEL, (exp_w, exp_h))
                region = region.convert("RGB")
                tile_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

                # Detect plaques on expanded tile
                _, boxes_local, _ = detect_brown_and_black_regions_fixed_boxes_array(
                    tile_bgr,
                    sat_thresh=SAT_THRESH,
                    val_thresh=VAL_THRESH,
                    hue_range=HUE_RANGE,
                    min_area=MIN_AREA,
                    microns_per_pixel=MICRONS_PER_PIXEL,
                    box_size_um=BOX_SIZE_UM,
                    overlap_thresh=OVERLAP_THRESH,
                    min_plaque_fraction=MIN_PLAQUE_FRACTION,
                    min_plaque_pixels=MIN_PLAQUE_PIXELS,
                    black_val_thresh=BLACK_VAL_THRESH
                )

                # Keep only boxes whose centers lie inside the core tile
                for (lx1, ly1, lx2, ly2) in boxes_local:
                    cx_local = (lx1 + lx2) // 2
                    cy_local = (ly1 + ly2) // 2

                    # Convert local (expanded) coords to global
                    gx1 = x_exp + lx1
                    gy1 = y_exp + ly1
                    gx2 = x_exp + lx2
                    gy2 = y_exp + ly2
                    gcx = x_exp + cx_local
                    gcy = y_exp + cy_local

                    # Check if center is inside the core tile (global)
                    if (gcx >= x0 and gcx < x0 + core_w and
                        gcy >= y0 and gcy < y0 + core_h):
                        writer.writerow([wsi_name, LEVEL, gx1, gy1, gx2, gy2])

    print(f"Saved global plaque boxes to {csv_path}")

def detect_brown_and_black_regions_fixed_boxes_array(
    img_bgr,
    sat_thresh=50,
    val_thresh=50,
    hue_range=(5, 35),
    min_area=90,
    microns_per_pixel=0.2305,
    box_size_um=100.0,
    overlap_thresh=0.7,
    min_plaque_fraction=0.005,
    min_plaque_pixels=80,
    black_val_thresh=40
):
    H, W = img_bgr.shape[:2]
    box_size_px = int(round(box_size_um / microns_per_pixel))
    half_box = box_size_px // 2

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    h_min, h_max = hue_range
    mask_h = cv2.inRange(h, h_min, h_max)
    mask_s = cv2.inRange(s, sat_thresh, 255)
    mask_v = cv2.inRange(v, val_thresh, 255)
    mask_brown = cv2.bitwise_and(mask_h, cv2.bitwise_and(mask_s, mask_v))

    mask_black = cv2.inRange(v, 0, black_val_thresh)

    mask_plaque = cv2.bitwise_or(mask_brown, mask_black)

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_plaque, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    initial_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x1 = cx - half_box
        y1 = cy - half_box
        x2 = cx + half_box
        y2 = cy + half_box

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 > x1 and y2 > y1:
            initial_boxes.append((x1, y1, x2, y2))

    merged_boxes = merge_overlapping_boxes(
        initial_boxes,
        mask_clean,
        overlap_thresh=overlap_thresh,
        box_size_px=box_size_px,
        img_shape=img_bgr.shape
    )

    filtered_boxes = []
    for (x1, y1, x2, y2) in merged_boxes:
        patch = mask_clean[y1:y2, x1:x2]
        total_pixels = patch.size
        plaque_pixels = np.count_nonzero(patch > 0)
        plaque_fraction = plaque_pixels / float(total_pixels + 1e-8)

        if plaque_fraction >= min_plaque_fraction and plaque_pixels >= min_plaque_pixels:
            filtered_boxes.append((x1, y1, x2, y2))

    overlay_bgr = img_bgr.copy()
    for (x1, y1, x2, y2) in filtered_boxes:
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return mask_clean, filtered_boxes, overlay_bgr

if __name__ == "__main__":
    process_wsi_with_overlap(WSI_PATH, OUT_DIR)