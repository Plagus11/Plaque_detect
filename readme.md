Amyloid-Beta Plaque Detection Pipeline
End-to-end workflow for detecting amyloid-beta plaques on whole-slide .ndpi images and reviewing the detections as annotations inside QuPath.
Pipeline at a glance
   plaque_detect_base.py   ->   save_boxes.py   ->   convert_to_qupath.py   ->   import_boxes.groovy

      (WSI  ->  boxes)         (boxes -> CSV)       (CSV -> QuPath CSV)       (CSV -> QuPath annotations)

Each step takes the output of the previous one, so you can rerun any single stage without redoing the whole slide.
Requirements
Python side (any recent Python 3.10+):

pip install opencv-python numpy openslide-python

On macOS you also need the OpenSlide native library so that openslide-python can find it:

brew install openslide

QuPath side: any recent QuPath (0.4+ tested). The Groovy import script uses only the standard qupath.lib.* APIs.
Step 1 - Run plaque detection
plaque_detect_base.py opens a .ndpi slide, tiles it at level 0, runs the HSV-based brown detector on each tile, and collects one fixed-size box per plaque candidate.

Edit these constants at the top of the file for the run you want:

WSI_PATH   = "/path/to/Case_6_1_ABeta.ndpi"   # input slide

OUT_DIR    = "/path/to/wsi_plaque_boxes"      # where the box CSV is written

LEVEL      = 0                                # detection resolution

TILE_SIZE  = 1536                             # core tile side (px at LEVEL)

MARGIN     = 256                              # halo around each tile

MICRONS_PER_PIXEL = 0.2305                    # from the slide metadata

BOX_SIZE_UM       = 125.0                     # full box side, in microns

Run it:

python plaque_detect_base.py

The script reports the box count before and after the global merge and writes a CSV named:

{OUT_DIR}/{wsi_name}_plaque_boxes_no_cut_{BOX_SIZE_UM}um.csv

with columns:

wsi_name, level, x1, y1, x2, y2

Coordinates are in level-0 pixels. Each row is one plaque bounding box.
Step 2 - Save boxes (save_boxes.py)
save_boxes.py is a light wrapper around the detection output. Its job in the pipeline is to take the raw box list from step 1 and persist it in whatever layout you're standardising on for downstream tools -- for example splitting one WSI-per-file, adding a run timestamp, sorting or renaming columns, or copying to a shared archive folder.

If your save_boxes.py expects arguments, run it as:

python save_boxes.py --in  /path/to/wsi_plaque_boxes/Case_6_1_ABeta_plaque_boxes_no_cut_125um.csv \

                     --out /path/to/wsi_plaque_box_2/

(Adjust the flag names to match your version. The important thing is that its input is the CSV produced by step 1 and its output is a per-slide box CSV that step 3 will read.)
Step 3 - Convert to QuPath format (convert_to_qupath.py)
QuPath's Groovy importer expects a different column layout than the detector writes. convert_to_qupath.py performs that translation.

Expected transformation:

From (detector)
To (QuPath)
wsi_name
image
x1
x
y1
y
x2 - x1
width
y2 - y1
height
(constant "plaque")
class


So the resulting file has the header:

image,x,y,width,height,class

One important detail: the image column must exactly match the image name that QuPath shows in its Project pane (usually the filename including extension, e.g. Case_6_1_ABeta.ndpi). The Groovy script filters rows by this string, so a mismatch means zero annotations get imported. If your detection CSV strips the extension, add it back in the conversion.

Run it:

python convert_to_qupath.py --in  /path/to/wsi_plaque_box_2/Case_6_1_ABeta_plaque_boxes.csv \

                            --out /path/to/wsi_plaque_box_2/Case_6_1_ABeta_plaque_box_2_qupath_3.csv
Step 4 - Import into QuPath (import_boxes.groovy)
Open the target slide in QuPath, then open the script editor (Automate -> Show script editor), paste in the Groovy script, and update the path near the top:

def csvPath = '/path/to/wsi_plaque_box_2/Case_6_1_ABeta_plaque_box_2_qupath_3.csv'

Then run it with Run -> Run (or Ctrl/Cmd+R).
What the script does, in order
Reads the CSV and grabs the header row so it can look up columns by name (image, x, y, width, height, class). Column order in the CSV doesn't matter as long as those names are present.
Gets the current image name from the active QuPath viewer via QPEx.getCurrentImageData() and server.getMetadata().getName().
Filters rows so that only rows whose image column matches the current slide get imported. This is what lets you keep detections for many slides in one CSV without needing to split it up per slide.
Builds one annotation per row: a rectangular ROI at (x, y, width, height) in level-0 pixel coordinates (same units the detector wrote), and assigns it the QuPath class named in the class column (creates the class if it doesn't already exist).
Adds all annotations to the hierarchy with QPEx.addObjects(annotations) and prints how many were imported.
After import
The plaques appear as rectangular annotations of the class you assigned. Save the QuPath project (or the individual image data) so the annotations persist across sessions:

File -> Save

To wipe them and re-import (e.g. after re-running detection with tweaked thresholds), select all annotations and delete:

Objects -> Delete... -> Delete all annotations

then re-run the Groovy script.
Reproducibility tips
Slide-name consistency: the image column in the QuPath CSV must match QuPath's image name character-for-character. If you rename slides in the filesystem after importing them into a QuPath project, the project keeps the old name -- check the Project pane rather than Finder / Explorer.
Pixel scale: the detector treats one CSV pixel as one level-0 pixel of the slide. QuPath does the same, so the boxes land in the right place without any scaling. If you ever run detection at LEVEL != 0, multiply coordinates by slide.level_downsamples[LEVEL] before writing them to the QuPath CSV.
Class colours: if you want plaques to render in a specific colour, create the class ahead of time in QuPath (Annotations -> Add/Remove... -> Add class) and set its colour there. The Groovy script only assigns the class; it doesn't set colours.
Troubleshooting
The Groovy script prints "Imported 0 annotations from CSV." -- the CSV's image column doesn't match the current slide name in QuPath. Print imageName from the script (it already does) and compare it byte-for-byte with the values in the CSV.

OpenSlide import error on macOS -- openslide-python needs the native OpenSlide library. Install it with brew install openslide and re-run the Python script.

Detection is slow -- the tissue pre-mask and threaded tile loop in the optimised detector are the biggest levers. If you're on the base detector, consider switching to the optimised version; on a typical brain section the tissue mask alone skips 50-80% of tile reads.

Boxes look shifted in QuPath -- almost always a coordinate-space mismatch: either the detector wrote coordinates at a level other than 0, or the conversion script computed width/height from something other than x2-x1/y2-y1. Sanity check: pick one row from the QuPath CSV and confirm (x, y, x+width, y+height) matches (x1, y1, x2, y2) in the detector CSV.

