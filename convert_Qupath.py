import os
import pandas as pd

BASE_DIR = "/Users/savin/PycharmProjects/PythonProject3/wsi_plaque_boxes_variable"
wsi_name = "Case_1_1_ABeta_plaque_boxes_variable.csv"

in_csv = os.path.join(BASE_DIR, f"{wsi_name}")
out_csv = os.path.join(BASE_DIR, f"{wsi_name}_plaque_box_variable_qupath_3.csv")

df = pd.read_csv(in_csv)

df_qupath = pd.DataFrame()
df_qupath["image"] = df["wsi_name"] + ".ndpi"  # adjust extension to match QuPath
df_qupath["x"] = df["x1"]
df_qupath["y"] = df["y1"]
df_qupath["width"] = df["x2"] - df["x1"]
df_qupath["height"] = df["y2"] - df["y1"]
df_qupath["class"] = "Plaque"

df_qupath.to_csv(out_csv, index=False)
print(f"Saved QuPath-friendly CSV to {out_csv}")