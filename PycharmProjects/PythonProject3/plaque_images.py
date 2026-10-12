"""
plaque_images - image-based plaque classification, on a consistent scale.

Companion to plaque_classifier. That module classifies from the morphometric
table; this one classifies from the pixels. Both produce a pc.Dataset, so both
run through the same case-grouped CV, permutation test and learning curve, and
land in the same compare_models() table.

    import plaque_images as pi, plaque_classifier as pc

    man = pi.export_image_dataset("label_batch_multicase/..._features.csv",
                                  "labels_label_batch_multicase.csv",
                                  out_dir="image_dataset")
    X   = pi.image_features(man, kind="hog+colour")
    ds  = pi.make_image_dataset(man, X)
    print(pc.compare_models(ds).to_string(index=False))


THE SCALE PROBLEM - read this before using the labelling crops for training
--------------------------------------------------------------------------
The crops in label_batch_multicase were rendered with an ADAPTIVE field of view
(128-405 um) so the annotator could see oversized deposits whole. Every one was
then written at 320x320 px, which means microns-per-pixel differs by up to 3.2x
between images.

For a human that is fine - the burnt-in scale bar carries the size information.
For a model it is corrosive twice over: absolute size stops being recoverable
from the pixels, and the varying resample factor adds a nuisance dimension the
model will happily fit. Since plaque size is one of the main things separating
diffuse from cored deposits, this matters.

So this module re-cuts every crop at ONE fixed physical field (default 320 um,
which clips none of the 500 boxes) and pads at the tissue edge. Every exported
image then has identical um/px, and a plaque that is twice as wide really does
occupy twice as many pixels.


WHAT TO EXPECT AT 500 LABELS
----------------------------
Roughly 400 training images across 4-5 classes is very little for a CNN trained
from scratch - it will memorise. The ordering that usually holds at this size is:

    frozen pretrained embeddings  >  classical features (HOG/colour)  >  CNN from scratch

`hog+colour` needs only scikit-image and runs in seconds; `embed` needs torch and
is usually the strongest; `cnn` is provided because it is worth seeing overfit
rather than being told about it.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

try:
    import openslide
    HAVE_OPENSLIDE = True
except Exception:  # pragma: no cover
    HAVE_OPENSLIDE = False

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:  # pragma: no cover
    HAVE_PIL = False

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

import plaque_classifier as pc
import plaque_labeller as pl

# -------------------------
# CONFIG
# -------------------------

FIELD_UM = 320.0     # fixed physical field; 320 clips none of the current boxes
IMAGE_PX = 224       # output edge, matching what ImageNet backbones expect
from plaque_common import MICRONS_PER_PIXEL


# -------------------------
# Export
# -------------------------

def export_image_dataset(features_csv: str, labels_csv: str,
                         wsi_dir: str = pl.DEFAULT_WSI_DIR,
                         out_dir: str = "image_dataset",
                         field_um: float = FIELD_UM, image_px: int = IMAGE_PX,
                         mask_background: bool = False, resume: bool = True,
                         progress: bool = True) -> pd.DataFrame:
    """
    Re-cut every labelled plaque at a fixed physical field and write PNGs.

    Returns the manifest: plaque_id, label, wsi_name, path, plus the box size in
    microns so you can check afterwards that size survived the export.

    mask_background : blank everything outside the recovered plaque mask. Removes
        neighbouring deposits and background staining, which is either a helpful
        denoiser or the removal of real context - worth running both ways.
    resume : skip plaques whose PNG already exists. Reading 500 regions from
        multi-gigabyte NDPIs takes minutes and is easily interrupted; with this on
        a re-run picks up where it stopped.
    """
    if not (HAVE_OPENSLIDE and HAVE_PIL):
        raise ImportError("openslide-python and Pillow are required.")
    import cv2

    df = pc.load_labelled(features_csv, labels_csv)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    ctx_px = int(round(field_um / MICRONS_PER_PIXEL))
    rows, slides, clipped = [], {}, 0
    try:
        for i, (_, r) in enumerate(df.iterrows()):
            if progress and i % 100 == 0:
                print(f"  {i}/{len(df)}", flush=True)
            name = str(r["wsi_name"])
            if name not in slides:
                slides[name] = openslide.OpenSlide(pl.resolve_wsi_path(name, wsi_dir))
            slide = slides[name]
            W, H = slide.dimensions

            x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            if max(x2 - x1, y2 - y1) * MICRONS_PER_PIXEL > field_um:
                clipped += 1

            path = os.path.join(out_dir, "images", f"{r['plaque_id']}.png")
            if resume and os.path.exists(path):
                rows.append({"plaque_id": r["plaque_id"], "label": r["label"],
                             "wsi_name": name, "path": path,
                             "box_w_um": (x2 - x1) * MICRONS_PER_PIXEL,
                             "box_h_um": (y2 - y1) * MICRONS_PER_PIXEL})
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dx, dy = cx - ctx_px // 2, cy - ctx_px // 2

            # White canvas so edge-of-slide plaques stay on the same scale
            # instead of being stretched to fill the frame.
            canvas = Image.new("RGB", (ctx_px, ctx_px), (255, 255, 255))
            sx1, sy1 = max(0, dx), max(0, dy)
            sx2, sy2 = min(W, dx + ctx_px), min(H, dy + ctx_px)
            if sx2 > sx1 and sy2 > sy1:
                piece = slide.read_region((sx1, sy1), 0,
                                          (sx2 - sx1, sy2 - sy1)).convert("RGB")
                canvas.paste(piece, (sx1 - dx, sy1 - dy))

            if mask_background:
                arr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
                import plaque_features as pf
                m = pf.recover_mask_from_box(arr, mpp=MICRONS_PER_PIXEL,
                                             dark_gray=pl.DARK_GRAY_THRESH)
                if m is not None:
                    arr[m == 0] = 255
                    canvas = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

            img = canvas.resize((image_px, image_px), Image.LANCZOS)
            img.save(path, format="PNG", optimize=True)

            rows.append({"plaque_id": r["plaque_id"], "label": r["label"],
                         "wsi_name": name, "path": path,
                         "box_w_um": (x2 - x1) * MICRONS_PER_PIXEL,
                         "box_h_um": (y2 - y1) * MICRONS_PER_PIXEL})
    finally:
        for s in slides.values():
            s.close()

    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)
    print(f"Exported {len(man)} images at {field_um:.0f} um / {image_px} px "
          f"({field_um / image_px:.3f} um per pixel, identical for every image).")
    if clipped:
        warnings.warn(f"{clipped} box(es) exceed the {field_um:.0f} um field and are "
                      f"cropped. Raise field_um if that matters.")
    return man


# -------------------------
# Classical image features
# -------------------------

def image_features(manifest: pd.DataFrame, kind: str = "hog+colour",
                   progress: bool = True) -> pd.DataFrame:
    """
    Turn each image into a feature vector, no deep learning required.

    kind : "hog"        shape/edge structure, orientation histograms
           "colour"     per-channel RGB + HSV histograms; captures stain intensity
           "lbp"        local binary patterns; texture granularity
           "hog+colour" the usual best classical combination
           "all"        everything

    HOG is the interesting one here: it encodes the radial edge structure that
    distinguishes a cored deposit (sharp central gradient) from a diffuse one
    (soft, no dominant edge), which is exactly the call the morphometric features
    struggle with.
    """
    from skimage.feature import hog, local_binary_pattern
    from skimage.color import rgb2gray, rgb2hsv

    want = set(kind.split("+")) if kind != "all" else {"hog", "colour", "lbp"}
    feats = []
    for i, path in enumerate(manifest["path"]):
        if progress and i % 100 == 0:
            print(f"  {i}/{len(manifest)}", flush=True)
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        row = {}

        if "hog" in want:
            g = rgb2gray(img)
            h = hog(g, orientations=9, pixels_per_cell=(28, 28),
                    cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
            row.update({f"hog_{j}": v for j, v in enumerate(h)})

        if "colour" in want:
            hsv = rgb2hsv(img)
            for cname, chan in (("r", img[..., 0]), ("g", img[..., 1]),
                                ("b", img[..., 2]), ("h", hsv[..., 0]),
                                ("s", hsv[..., 1]), ("v", hsv[..., 2])):
                hist, _ = np.histogram(chan, bins=16, range=(0, 1), density=True)
                row.update({f"{cname}hist_{j}": v for j, v in enumerate(hist)})
                row[f"{cname}_mean"] = float(chan.mean())
                row[f"{cname}_std"] = float(chan.std())
            # Optical density: the stain measure, on the same definition the
            # morphometric pipeline uses.
            od = -np.log10(np.clip(img.mean(axis=2), 1e-3, 1.0))
            row["od_mean"], row["od_std"] = float(od.mean()), float(od.std())
            row["od_p90"] = float(np.percentile(od, 90))
            row["dark_frac"] = float((od > 0.5).mean())

        if "lbp" in want:
            g = (rgb2gray(img) * 255).astype(np.uint8)
            lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            row.update({f"lbp_{j}": v for j, v in enumerate(hist)})

        feats.append(row)

    X = pd.DataFrame(feats, index=manifest.index)
    print(f"{kind}: {X.shape[1]} features from {len(X)} images")
    return X


# -------------------------
# Pretrained embeddings (optional, needs torch)
# -------------------------

# The three backbones named in the Alzheimer's transfer-learning literature,
# plus resnet50. All are ImageNet-pretrained and available in torchvision.
#
# Embedding width differs a lot, which matters at n~400: alexnet's 4096 dims
# against 400 samples is a far harder estimation problem than googlenet's 1024
# or resnet18's 512, irrespective of feature quality. Expect the narrow ones to
# win here for that reason alone.
BACKBONES = {
    "alexnet":   {"weights": "AlexNet_Weights",   "head": "classifier", "dims": 4096},
    "googlenet": {"weights": "GoogLeNet_Weights", "head": "fc",         "dims": 1024},
    "resnet18":  {"weights": "ResNet18_Weights",  "head": "fc",         "dims": 512},
    "resnet50":  {"weights": "ResNet50_Weights",  "head": "fc",         "dims": 2048},
}


def embed_pretrained(manifest: pd.DataFrame, model_name: str = "resnet18",
                     batch_size: int = 32, progress: bool = True) -> pd.DataFrame:
    """
    Frozen ImageNet backbone as a fixed feature extractor.

    No fine-tuning: with ~400 training images, updating 11-60M weights overfits
    immediately, while the frozen features are already a good general-purpose
    description of texture and shape. This is the strongest image option at this
    label budget.

    model_name : any key of BACKBONES - alexnet, googlenet, resnet18, resnet50.

    A caution on the ImageNet normalisation below: these statistics come from
    natural photographs, and DAB/haematoxylin sections are nothing like them.
    It is still the right choice, because the frozen filters were trained under
    exactly this normalisation and changing it degrades them - but it is one
    reason pathology-pretrained encoders (UNI, CONCH, Virchow) generally beat
    ImageNet backbones on histology.
    """
    if not HAVE_TORCH:
        raise ImportError("torch is required. pip install torch torchvision")
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    if model_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {model_name!r}. "
                         f"Available: {sorted(BACKBONES)}")
    spec = BACKBONES[model_name]
    weights = getattr(models, spec["weights"]).DEFAULT
    net = getattr(models, model_name)(weights=weights)

    # Strip the 1000-class ImageNet head so the penultimate representation comes
    # out instead of class logits.
    if spec["head"] == "fc":
        net.fc = torch.nn.Identity()
    else:  # alexnet / vgg-style: drop only the final Linear, keep fc6+fc7
        net.classifier = torch.nn.Sequential(*list(net.classifier.children())[:-1])
    net.eval()

    tf = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    out = []
    with torch.no_grad():
        for start in range(0, len(manifest), batch_size):
            if progress and start % (batch_size * 5) == 0:
                print(f"  {start}/{len(manifest)}", flush=True)
            chunk = manifest["path"].iloc[start:start + batch_size]
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk])
            out.append(net(batch).numpy())

    X = pd.DataFrame(np.vstack(out), index=manifest.index)
    X.columns = [f"{model_name}_{i}" for i in range(X.shape[1])]
    print(f"{model_name}: {X.shape[1]} dims from {len(X)} images")
    return X


def compare_backbones(manifest: pd.DataFrame, names=("alexnet", "googlenet", "resnet18"),
                      models_to_try=("svm", "logreg"), n_splits: int = 4,
                      n_repeats: int = 2, cache_dir: str = "results/cache") -> pd.DataFrame:
    """
    Run several frozen backbones through the identical case-grouped CV.

    `names` can mix ImageNet backbones (BACKBONES) and pathology foundation
    models (FOUNDATION_BACKBONES) - each is routed to the right embedding
    function automatically. Embeddings are cached per backbone, so re-running
    to try a different classifier, or adding one more name to the list, costs
    nothing for the ones already done.

    Returns one row per (backbone, classifier), sorted by balanced accuracy, with
    the morphometric baseline appended for reference if it can be computed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    rows = []
    for name in names:
        cache = os.path.join(cache_dir, f"emb_{name}_{len(manifest)}.pkl")
        if os.path.exists(cache):
            X = pd.read_pickle(cache)
            print(f"{name}: reusing cached embeddings ({X.shape[1]} dims)")
        elif name in FOUNDATION_BACKBONES:
            X = embed_foundation(manifest, model_name=name)
            X.to_pickle(cache)
        else:
            X = embed_pretrained(manifest, model_name=name)
            X.to_pickle(cache)

        ds = make_image_dataset(manifest, X)
        comp = pc.compare_models(ds, kinds=list(models_to_try),
                                 n_splits=n_splits, n_repeats=n_repeats)
        for _, r in comp.iterrows():
            rows.append({"backbone": name, "dims": X.shape[1],
                         "classifier": r["model"],
                         "balanced_accuracy": r["balanced_accuracy"],
                         "lo95": r["lo95"], "hi95": r["hi95"],
                         "macro_f1": r["macro_f1"]})

    out = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    return out.reset_index(drop=True)


# -------------------------
# Pathology foundation models (optional, needs torch + timm;
# UNI/UNI2-h/Virchow2 additionally need a Hugging Face token with the
# model's license accepted)
# -------------------------

# Each entry describes how to build the backbone and how to turn its raw
# output into a single embedding vector. "repo" is the Hugging Face hub id.
# "gated" models require (1) accepting the license on the model's HF page
# while logged in, and (2) `huggingface-cli login` (or env var HF_TOKEN) in
# this environment - embed_foundation() raises a message saying exactly that
# if the download comes back as a permissions error, rather than a raw
# traceback.
#
# dims is the embedding width AFTER pooling (see "pool" below), which is what
# actually lands in the feature table.
FOUNDATION_BACKBONES = {
    # Owkin, DINOv2-based, trained on TCGA. Fully open - no HF login needed.
    "phikon":    {"backend": "transformers", "repo": "owkin/phikon",
                  "dims": 768, "gated": False},
    "phikon-v2": {"backend": "transformers", "repo": "owkin/phikon-v2",
                  "dims": 1024, "gated": False},

    # Bioptimus, ViT-giant, trained on a large multi-site WSI corpus. Gated:
    # https://huggingface.co/bioptimus/H-optimus-0 - and uses its own
    # (non-ImageNet) normalisation statistics.
    "h-optimus-0": {"backend": "timm", "repo": "hf-hub:bioptimus/H-optimus-0",
                    "dims": 1536, "gated": True,
                    "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": False},
                    "mean": (0.707223, 0.578729, 0.703617),
                    "std":  (0.211883, 0.230117, 0.177517)},

    # Mahmood Lab, ViT-L, >100k WSIs. Gated: accept the license at
    # https://huggingface.co/MahmoodLab/UNI, then huggingface-cli login.
    "uni": {"backend": "timm", "repo": "hf-hub:MahmoodLab/UNI",
            "dims": 1024, "gated": True,
            "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": True}},

    # Mahmood Lab, ViT-H with 8 register tokens, successor to UNI. Gated:
    # https://huggingface.co/MahmoodLab/UNI2-h
    "uni2-h": {"backend": "timm", "repo": "hf-hub:MahmoodLab/UNI2-h",
               "dims": 1536, "gated": True,
               "timm_kwargs": {
                   "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
                   "init_values": 1e-5, "embed_dim": 1536,
                   "mlp_ratio": 2.66667 * 2, "num_classes": 0,
                   "no_embed_class": True, "reg_tokens": 8,
                   "dynamic_img_size": True}},

    # Paige AI, ViT-H with 4 register tokens, trained on ~3M WSIs. Gated:
    # https://huggingface.co/paige-ai/Virchow2
    # Output embedding is [CLS ; mean(patch tokens)] per the model card, which
    # is why dims is 2*embed_dim rather than embed_dim.
    "virchow2": {"backend": "timm", "repo": "hf-hub:paige-ai/Virchow2",
                 "dims": 2560, "gated": True, "pool": "virchow",
                 "timm_kwargs": {}},

    # Lunit, ViT-S/16 trained with DINO on TCGA. Community timm port
    # (1aurent), confirmed non-gated and loads with plain pooled output -
    # no custom stem or token surgery needed, unlike the entries above.
    "lunit-dino": {"backend": "timm",
                   "repo": "hf-hub:1aurent/vit_small_patch16_224.lunit_dino",
                   "dims": 384, "gated": False, "timm_kwargs": {}},

    # Lunit, plain ResNet50 trained with SwAV on TCGA. Same family as the
    # ImageNet resnet50 already in BACKBONES, just different pretraining -
    # the cheapest apples-to-apples upgrade to try first.
    "lunit-swav": {"backend": "timm", "repo": "hf-hub:1aurent/resnet50.lunit_swav",
                   "dims": 2048, "gated": False, "timm_kwargs": {}},

    # Providence/Microsoft, ViT-giant tile encoder, 1.3B tiles. Gated:
    # https://huggingface.co/prov-gigapath/prov-gigapath . dims is the timm
    # default pooled output and hasn't been verified against real weights
    # (gated, no token in this environment) - confirm X.shape[1] once you
    # have access; embed_foundation doesn't rely on this number being right.
    "prov-gigapath": {"backend": "timm", "repo": "hf_hub:prov-gigapath/prov-gigapath",
                      "dims": 1536, "gated": True, "timm_kwargs": {}},
}

# Deliberately NOT wired in, to avoid silently-wrong embeddings rather than a
# loud failure:
#   - CTransPath: requires a custom "ConvStem" patch-embedding module (plain
#     timm.create_model fails with a state_dict key mismatch - confirmed by
#     trying it). The exact ConvStem source needs pulling from the original
#     repo and checking by hand before it's trustworthy; ask if you want it.
#   - kaiko-ai/midnight: loads fine via `transformers`, but published
#     descriptions of its embedding dim and CLS/patch-token pooling were
#     inconsistent across sources when checked, which is exactly the kind of
#     detail that corrupts an embedding silently if guessed wrong.
#   - CONCH: different shape of problem (vision-language, needs
#     `pip install git+https://github.com/Mahmoodlab/CONCH.git`, plus the
#     same HF-gated-license step as UNI). Worth doing as a separate pass
#     given the extra dependency, not a one-line registry entry.

# timm's custom SwiGLU MLP + SiLU activation, used by UNI2-h and Virchow2.
# Imported lazily inside _build_foundation_model so a missing/old timm only
# breaks the models that need it, not the whole module.
_SWIGLU_MODELS = {"uni2-h", "virchow2"}


def _build_foundation_model(model_name: str):
    """Construct and return (model, transform) for one FOUNDATION_BACKBONES entry."""
    spec = FOUNDATION_BACKBONES[model_name]
    import torch

    if spec["backend"] == "transformers":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(spec["repo"])
        model = AutoModel.from_pretrained(spec["repo"])
        model.eval()
        return model, processor

    # backend == "timm"
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for pathology foundation models. "
                          "pip install timm")

    kwargs = dict(spec.get("timm_kwargs", {}))
    kwargs.setdefault("pretrained", True)
    kwargs.setdefault("num_classes", 0)
    if model_name in _SWIGLU_MODELS:
        from timm.layers import SwiGLUPacked
        kwargs.setdefault("mlp_layer", SwiGLUPacked)
        kwargs.setdefault("act_layer", torch.nn.SiLU)

    try:
        model = timm.create_model(spec["repo"], **kwargs)
    except Exception as e:
        # Recognise an auth/gating failure by what the error actually says,
        # not by our own "gated" guess in the registry - that guess has
        # already been wrong once (h-optimus-0).
        msg = str(e).lower()
        if any(s in msg for s in ("401", "403", "gated", "restricted", "unauthorized")):
            raise PermissionError(
                f"{model_name}: access denied downloading {spec['repo']}. This "
                f"model is gated - log in at huggingface.co, accept the "
                f"license on the model's page, then run "
                f"`huggingface-cli login` (or set HF_TOKEN) and retry.") from e
        raise
    model.eval()

    if "mean" in spec:
        mean, std = spec["mean"], spec["std"]
    else:
        cfg = model.pretrained_cfg
        mean, std = cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std),
    ])
    return model, transform


def embed_foundation(manifest: pd.DataFrame, model_name: str = "phikon-v2",
                     batch_size: int = 16, device: str | None = None,
                     progress: bool = True) -> pd.DataFrame:
    """
    Frozen pathology foundation model as a fixed feature extractor.

    Same contract as embed_pretrained() - no fine-tuning, one row per image,
    columns named "{model_name}_{i}" - but the backbone was pretrained on
    histology instead of ImageNet photographs. model_name is any key of
    FOUNDATION_BACKBONES.

    These checkpoints are overwhelmingly H&E-trained, and your crops are
    DAB/brown-chromogen immunostain - a real domain shift that general
    benchmarks don't speak to. Treat the ranking from the literature as a
    prior to check with compare_backbones(), not a conclusion.

    device : "mps", "cuda", "cpu", or None to auto-pick the best available
        (Apple Silicon -> mps). Giant models (h-optimus-0, uni2-h, virchow2)
        are ~1-2.7GB of weights; if you are short on memory, pass a smaller
        batch_size rather than switching to cpu.
    """
    if not HAVE_TORCH:
        raise ImportError("torch is required. pip install torch torchvision")
    import torch

    if model_name not in FOUNDATION_BACKBONES:
        raise ValueError(f"Unknown foundation model {model_name!r}. "
                         f"Available: {sorted(FOUNDATION_BACKBONES)}")
    spec = FOUNDATION_BACKBONES[model_name]

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model, proc = _build_foundation_model(model_name)
    model = model.to(device)

    out = []
    with torch.no_grad():
        for start in range(0, len(manifest), batch_size):
            if progress and start % (batch_size * 5) == 0:
                print(f"  {start}/{len(manifest)}", flush=True)
            chunk = manifest["path"].iloc[start:start + batch_size]
            images = [Image.open(p).convert("RGB") for p in chunk]

            if spec["backend"] == "transformers":
                inputs = proc(images=images, return_tensors="pt").to(device)
                feats = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
            else:
                batch = torch.stack([proc(im) for im in images]).to(device)
                raw = model(batch)
                if spec.get("pool") == "virchow":
                    # Virchow2 ships 4 register tokens after the CLS token;
                    # the model card's embedding is [CLS ; mean(patch tokens)].
                    cls_tok = raw[:, 0]
                    patch_toks = raw[:, 5:]
                    feats = torch.cat([cls_tok, patch_toks.mean(dim=1)], dim=-1)
                else:
                    feats = raw

            out.append(feats.cpu().numpy())

    X = pd.DataFrame(np.vstack(out), index=manifest.index)
    X.columns = [f"{model_name}_{i}" for i in range(X.shape[1])]
    print(f"{model_name}: {X.shape[1]} dims from {len(X)} images")
    return X


# -------------------------
# Full-dataset embedding, no label required
# -------------------------
#
# embed_pretrained/embed_foundation above assume every image is already a
# PNG on disk (export_image_dataset's job). That is fine at 500 labelled
# crops; at ~242k detections it would mean reading every multi-gigabyte NDPI
# region twice (once to write the PNG, once to read it back) and leaving
# ~12GB of crops on disk nothing else needs. This instead reads each region
# straight from the slide and feeds it to the model without ever writing a
# crop file, and never holds more than one batch of pixels in memory at a
# time - RAM use is bounded by batch_size, not by how many plaques there are.
#
# A threaded reader stays ahead of the (single) model-inference thread so the
# GPU/ANE is never idle waiting on OpenSlide I/O, which plaque_detect.py's
# docstring already establishes as the dominant cost at this scale (OpenSlide
# releases the GIL on read_region). Embeddings are checkpointed to disk every
# few thousand plaques, keyed by plaque_id, so an interrupted run resumes
# instead of restarting.

def _read_crop(slide, row, ctx_px: int, image_px: int):
    """One plaque -> one resized RGB array, centred the same way export_image_dataset does."""
    x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    dx, dy = cx - ctx_px // 2, cy - ctx_px // 2
    W, H = slide.dimensions
    canvas = Image.new("RGB", (ctx_px, ctx_px), (255, 255, 255))
    sx1, sy1 = max(0, dx), max(0, dy)
    sx2, sy2 = min(W, dx + ctx_px), min(H, dy + ctx_px)
    if sx2 > sx1 and sy2 > sy1:
        piece = slide.read_region((sx1, sy1), 0, (sx2 - sx1, sy2 - sy1)).convert("RGB")
        canvas.paste(piece, (sx1 - dx, sy1 - dy))
    return np.asarray(canvas.resize((image_px, image_px), Image.LANCZOS))


def embed_full_dataset(detections: pd.DataFrame, model_name: str = "lunit-swav",
                       wsi_dir: str = None, field_um: float = FIELD_UM,
                       image_px: int = IMAGE_PX, batch_size: int = 64,
                       n_read_threads: int = 8, device: str | None = None,
                       checkpoint_path: str = "results/embeddings/full_dataset.pkl",
                       checkpoint_every: int = 2000, resume: bool = True,
                       progress: bool = True) -> pd.DataFrame:
    """
    Embed every row of a detection table (e.g. pc.load_detection_tables()'s
    ~242k plaques) straight from the slides, with bounded memory and
    checkpointed progress.

    model_name : any BACKBONES or FOUNDATION_BACKBONES key. Pick a light one
        for a full run - lunit-swav/resnet18/lunit-dino/phikon are all fast;
        the giant-tier foundation models will work but take much longer over
        this many images.
    n_read_threads : OpenSlide region reads are I/O-bound and release the
        GIL, so a handful of threads keeps the model fed without reading
        ahead faster than it can be consumed. Keep this well under your core
        count - it is reading, not the embedding forward pass, that this
        parallelises.
    checkpoint_path : embeddings so far, as {plaque_id: vector}. Re-running
        with the same path and resume=True skips plaques already embedded,
        so a long run can be stopped and continued.
    """
    if not (HAVE_OPENSLIDE and HAVE_TORCH):
        raise ImportError("openslide-python and torch are required.")
    import pickle
    import torch
    from concurrent.futures import ThreadPoolExecutor
    wsi_dir = wsi_dir or pl.DEFAULT_WSI_DIR

    done: dict[str, np.ndarray] = {}
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            done = pickle.load(f)
        print(f"resuming: {len(done):,} plaque(s) already embedded")

    todo = detections[~detections["plaque_id"].isin(done)].reset_index(drop=True)
    if todo.empty:
        print("nothing left to embed")
    else:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        if model_name in FOUNDATION_BACKBONES:
            spec = FOUNDATION_BACKBONES[model_name]
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            model, proc = _build_foundation_model(model_name)
            model = model.to(device)
            backend = spec["backend"]
            pool = spec.get("pool")
        else:
            import torchvision.models as tv_models
            import torchvision.transforms as T
            spec = BACKBONES[model_name]
            device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            net = getattr(tv_models, model_name)(
                weights=getattr(tv_models, spec["weights"]).DEFAULT)
            if spec["head"] == "fc":
                net.fc = torch.nn.Identity()
            else:
                net.classifier = torch.nn.Sequential(*list(net.classifier.children())[:-1])
            net.eval()
            model, backend, pool = net.to(device), "torchvision", None
            proc = T.Compose([T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        ctx_px = int(round(field_um / MICRONS_PER_PIXEL))
        n_before_run = len(done)
        last_checkpoint = n_before_run
        t0 = time.time()

        for wsi_name, grp in todo.groupby("wsi_name"):
            slide = openslide.OpenSlide(pl.resolve_wsi_path(wsi_name, wsi_dir))
            rows = list(grp.itertuples(index=False))
            try:
                with ThreadPoolExecutor(n_read_threads) as ex:
                    crops_iter = ex.map(lambda r: _read_crop(slide, r, ctx_px, image_px), rows)
                    batch_imgs, batch_ids = [], []

                    def flush_batch():
                        vecs = _embed_batch(batch_imgs, model, proc, backend, pool, device)
                        done.update(zip(batch_ids, vecs))
                        batch_imgs.clear()
                        batch_ids.clear()

                    for row, arr in zip(rows, crops_iter):
                        batch_imgs.append(arr)
                        batch_ids.append(row.plaque_id)
                        if len(batch_imgs) == batch_size:
                            flush_batch()
                            n_this_run = len(done) - n_before_run
                            if progress and n_this_run % (checkpoint_every) < batch_size:
                                rate = n_this_run / max(time.time() - t0, 1e-9)
                                remaining = len(todo) - n_this_run
                                print(f"  {len(done):,}/{len(detections):,} done overall, "
                                      f"{rate:.1f} plaques/s, "
                                      f"~{remaining / max(rate, 1e-9) / 60:.0f} min left",
                                      flush=True)
                            if len(done) - last_checkpoint >= checkpoint_every:
                                with open(checkpoint_path, "wb") as f:
                                    pickle.dump(done, f)
                                last_checkpoint = len(done)
                    if batch_imgs:
                        flush_batch()
            finally:
                slide.close()
            with open(checkpoint_path, "wb") as f:
                pickle.dump(done, f)
            last_checkpoint = len(done)
            print(f"{wsi_name}: done ({len(done):,} total)", flush=True)

    ids = detections["plaque_id"].tolist()
    X = pd.DataFrame([done[i] for i in ids if i in done],
                     index=[i for i in ids if i in done])
    X.columns = [f"{model_name}_{i}" for i in range(X.shape[1])]
    print(f"{model_name}: {X.shape[1]} dims from {len(X):,} of {len(detections):,} plaques")
    return X


def _embed_batch(images: list[np.ndarray], model, proc, backend: str, pool,
                 device: str) -> np.ndarray:
    """Run one batch of RGB arrays through model, return (n, dims) numpy."""
    import torch
    with torch.no_grad():
        if backend == "transformers":
            inputs = proc(images=[Image.fromarray(a) for a in images],
                          return_tensors="pt").to(device)
            feats = model(**inputs).last_hidden_state[:, 0, :]
        elif backend == "timm":
            batch = torch.stack([proc(Image.fromarray(a)) for a in images]).to(device)
            raw = model(batch)
            if pool == "virchow":
                feats = torch.cat([raw[:, 0], raw[:, 5:].mean(dim=1)], dim=-1)
            else:
                feats = raw
        else:  # torchvision
            batch = torch.stack([proc(Image.fromarray(a)) for a in images]).to(device)
            feats = model(batch)
    return feats.cpu().numpy()


# -------------------------
# Bridge into plaque_classifier
# -------------------------

def make_image_dataset(manifest: pd.DataFrame, X: pd.DataFrame,
                       min_per_class: int = 5) -> "pc.Dataset":
    """
    Wrap image features in a pc.Dataset so every tool in plaque_classifier works
    on them unchanged - case-grouped CV, permutation baseline, learning curve,
    compare_models, suggest_to_label.
    """
    df = X.copy()
    df["label"] = manifest["label"].values
    df["wsi_name"] = manifest["wsi_name"].values
    return pc.make_dataset(df, features=list(X.columns),
                           min_per_class=min_per_class)


def combine(tabular: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Concatenate morphometric and image features for a fusion model."""
    a = tabular.reset_index(drop=True)
    b = images.reset_index(drop=True)
    return pd.concat([a, b], axis=1)


# -------------------------
# Small CNN, trained from scratch (optional, needs torch)
# -------------------------

def train_small_cnn(manifest: pd.DataFrame, epochs: int = 25, batch_size: int = 32,
                    lr: float = 1e-3, seed: int = 0, n_splits: int = 4) -> dict:
    """
    A deliberately small CNN under the same case-grouped CV as everything else.

    Included so the overfitting is visible rather than assumed: at ~400 training
    images the training accuracy will climb well past the validation score. Read
    the gap, not the training curve.
    """
    if not HAVE_TORCH:
        raise ImportError("torch is required. pip install torch torchvision")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    import torchvision.transforms as T
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold

    torch.manual_seed(seed)
    classes = sorted(manifest["label"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}

    train_tf = T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20), T.ColorJitter(0.15, 0.15, 0.1, 0.03),
        T.ToTensor(), T.Normalize([0.8, 0.75, 0.8], [0.15, 0.18, 0.15]),
    ])
    eval_tf = T.Compose([
        T.ToTensor(), T.Normalize([0.8, 0.75, 0.8], [0.15, 0.18, 0.15]),
    ])

    class DS(TorchDataset):
        def __init__(self, frame, tf):
            self.frame, self.tf = frame.reset_index(drop=True), tf

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            r = self.frame.iloc[i]
            return self.tf(Image.open(r["path"]).convert("RGB")), cls_idx[r["label"]]

    def build():
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(64, len(classes)),
        )

    y = manifest["label"].map(cls_idx).values
    groups = manifest["wsi_name"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_scores, train_scores = [], []
    for fold, (tr, te) in enumerate(cv.split(manifest, y, groups)):
        net = build()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        counts = np.bincount(y[tr], minlength=len(classes)).astype(float)
        w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=w)

        dl = DataLoader(DS(manifest.iloc[tr], train_tf), batch_size=batch_size,
                        shuffle=True)
        for _ in range(epochs):
            net.train()
            for xb, yb in dl:
                opt.zero_grad()
                lossf(net(xb), yb).backward()
                opt.step()

        net.eval()
        def score(idx, tf):
            preds = []
            with torch.no_grad():
                for xb, _ in DataLoader(DS(manifest.iloc[idx], tf), batch_size=64):
                    preds.append(net(xb).argmax(1).numpy())
            return balanced_accuracy_score(y[idx], np.concatenate(preds))

        te_s, tr_s = score(te, eval_tf), score(tr, eval_tf)
        fold_scores.append(te_s)
        train_scores.append(tr_s)
        print(f"  fold {fold}: train BA {tr_s:.3f}  test BA {te_s:.3f}")

    gap = float(np.mean(train_scores) - np.mean(fold_scores))
    return {
        "balanced_accuracy": float(np.mean(fold_scores)),
        "sd": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "train_balanced_accuracy": float(np.mean(train_scores)),
        "overfit_gap": gap,
        "folds": fold_scores,
        "message": (f"CNN from scratch: test BA {np.mean(fold_scores):.3f}, "
                    f"train BA {np.mean(train_scores):.3f}, gap {gap:+.3f}"
                    + ("  - memorising, as expected at this n." if gap > 0.2 else "")),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Image-based plaque classification on a consistent scale.")
    ap.add_argument("--features",
                    default="label_batch_multicase/label_batch_multicase_features.csv")
    ap.add_argument("--labels", default="labels_label_batch_multicase.csv")
    ap.add_argument("--wsi-dir", default=pl.DEFAULT_WSI_DIR)
    ap.add_argument("--out", default="image_dataset")
    ap.add_argument("--field-um", type=float, default=FIELD_UM)
    ap.add_argument("--image-px", type=int, default=IMAGE_PX)
    ap.add_argument("--mask-background", action="store_true")
    ap.add_argument("--kind", default="hog+colour",
                    help="hog, colour, lbp, hog+colour, all, embed, an ImageNet "
                         "backbone name (alexnet/googlenet/resnet18/resnet50), or a "
                         "pathology foundation model name (phikon/phikon-v2/"
                         "h-optimus-0/uni/uni2-h/virchow2) - needs torch, "
                         "foundation models also need timm")
    ap.add_argument("--compare-backbones", action="store_true",
                    help="benchmark a set of backbones on the same folds; combine "
                         "with --backbones to choose which")
    ap.add_argument("--backbones", nargs="+",
                    default=["alexnet", "googlenet", "resnet18"],
                    help="names to pass to --compare-backbones, mixing ImageNet "
                         "and foundation model names freely")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse images already in --out")
    ap.add_argument("--cnn", action="store_true", help="also train a small CNN")
    ap.add_argument("--fuse", action="store_true",
                    help="also test image features combined with the morphometric ones")
    args = ap.parse_args()

    man_path = os.path.join(args.out, "manifest.csv")
    if args.skip_export and os.path.exists(man_path):
        man = pd.read_csv(man_path)
        print(f"Reusing {len(man)} images from {args.out}")
    else:
        man = export_image_dataset(args.features, args.labels, wsi_dir=args.wsi_dir,
                                   out_dir=args.out, field_um=args.field_um,
                                   image_px=args.image_px,
                                   mask_background=args.mask_background)

    if args.compare_backbones:
        man3 = man[man["label"].isin(["diffuse", "cored", "diffuse_core_partial"])]
        man3 = man3.reset_index(drop=True)
        print(f"\nComparing frozen backbones on {len(man3)} images, "
              f"{man3['wsi_name'].nunique()} cases\n")
        table = compare_backbones(man3, names=args.backbones)
        print("\n" + table.round(3).to_string(index=False))
        table.to_csv("results/backbone_comparison.csv", index=False)
        print("\nWrote results/backbone_comparison.csv")
        raise SystemExit(0)

    if args.kind in BACKBONES:
        X = embed_pretrained(man, model_name=args.kind)
    elif args.kind in FOUNDATION_BACKBONES:
        X = embed_foundation(man, model_name=args.kind)
    elif args.kind == "embed":
        X = embed_pretrained(man)
    else:
        X = image_features(man, args.kind)
    ds = make_image_dataset(man, X)
    print("\n" + ds.describe() + "\n")

    print(f"=== {args.kind} features, case-grouped CV ===")
    print(pc.compare_models(ds, n_splits=4, n_repeats=2).round(3).to_string(index=False))

    if args.fuse:
        tab = pc.load_labelled(args.features, args.labels)
        tab = tab.set_index("plaque_id").loc[man["plaque_id"]].reset_index()
        fused = combine(tab[pc.FEATURES], X)
        fds = make_image_dataset(man, fused)
        print(f"\n=== morphometry + {args.kind} fused ===")
        print(pc.compare_models(fds, n_splits=4, n_repeats=2).round(3).to_string(index=False))

    if args.cnn:
        print("\n=== small CNN from scratch ===")
        print(train_small_cnn(man)["message"])
