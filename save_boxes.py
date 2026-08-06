import os
import csv
import cv2
import numpy as np
import openslide

# -------------------------
# CONFIG
# -------------------------

WSI_PATH = "/Users/savin/Documents/Case_6_1_ABeta.ndpi"
CSV_PATH = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_box_2/Case_6_1_ABeta_plaque_box_2_qupath_3.csv"
OUT_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_box_crops_range"
os.makedirs(OUT_DIR, exist_ok=True)

LEVEL = 0  # crop at full resolution

# Range of boxes to save (1-based indices, inclusive)
START_INDEX = 20000   # e.g. "from box 50"
END_INDEX   = 20050   # e.g. "to box 53"

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 4

# -------------------------
# LOAD WSI
# -------------------------

slide = openslide.OpenSlide(WSI_PATH)
wsi_name = os.path.splitext(os.path.basename(WSI_PATH))[0]

downsample = slide.level_downsamples[LEVEL]
assert downsample == 1, "This script assumes LEVEL=0; adjust if needed."

# -------------------------
# READ BOXES FROM CSV
# -------------------------

boxes = []
with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image"] != os.path.basename(WSI_PATH):
            continue

        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        boxes.append((x1, y1, x2, y2))

total_boxes = len(boxes)
print(f"Loaded {total_boxes} boxes from CSV")

# Optionally sort boxes (e.g. by area, descending) – comment out if you want CSV order
# boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

# -------------------------
# SELECT RANGE
# -------------------------

# Convert 1-based indices to 0-based slice
start_idx = max(0, START_INDEX - 1)
end_idx = min(END_INDEX, total_boxes)  # end is inclusive in our definition

if start_idx >= end_idx:
    raise ValueError(f"Invalid range: START_INDEX={START_INDEX}, END_INDEX={END_INDEX}, total_boxes={total_boxes}")

boxes_range = boxes[start_idx:end_idx]
print(f"Saving boxes {START_INDEX} to {END_INDEX} (0-based indices {start_idx} to {end_idx-1})")
print(f"Number of crops to save: {len(boxes_range)}")

# -------------------------
# SAVE CROPS AROUND SELECTED BOXES
# -------------------------

for idx_offset, (x1, y1, x2, y2) in enumerate(boxes_range):
    idx = start_idx + idx_offset + 1  # back to 1-based index for naming

    w = x2 - x1
    h = y2 - y1

    # Read region from WSI at level 0
    region = slide.read_region((x1, y1), LEVEL, (w, h))
    region = region.convert("RGB")
    crop_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    # Draw the box on the crop (full crop is the box, but this makes it explicit)
    cv2.rectangle(crop_bgr, (0, 0), (w-1, h-1), BOX_COLOR, BOX_THICKNESS)

    out_name = f"{wsi_name}_box_{idx}_x{x1}_y{y1}_w{w}_h{h}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, crop_bgr)
    print(f"Saved crop for box {idx} to {out_path}")