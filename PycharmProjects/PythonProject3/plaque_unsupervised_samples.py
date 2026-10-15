"""
plaque_unsupervised_samples - what did the best unsupervised clustering actually find?

plaque_unsupervised.py answers "do the labels track structure" with a single number
(ARI). This module answers the next question: structure in *what sense* - show the
actual groups KMeans found, cross-tabulated against the annotator's labels, and a
representative image sample per group so a human can look at what the model thought
was one type of plaque.

The best-scoring setting found so far (plaque_unsupervised's 6-method comparison,
results/unsupervised_methods_comparison_all.csv) is KMeans, k=3, on the morphometric
features of the full labelled batch - ARI 0.247. That is the default here, not a new
choice.

    python plaque_unsupervised_samples.py --n-samples 20

"Representative" = closest to that cluster's own centroid in the standardised feature
space used for clustering, not a random draw - the point is to show what KMeans
treated as the prototypical member of each group, not the full spread within it.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import plaque_classifier as pc
import plaque_unsupervised as pu

IMAGE_MANIFEST = "image_dataset/manifest.csv"


def best_kmeans_clusters(k: int = 3, seed: int = 0,
                         features_csv: str = pu.FEATURES_CSV) -> dict:
    """
    Reproduce the best-scoring clustering exactly as plaque_unsupervised's sweep
    found it: KMeans on the morphometric features of the full labelled batch.

    Returns the same shape as plaque_unsupervised.cluster_unlabelled(), plus
    centroid distances per plaque (needed for "representative" sampling).
    """
    cl = pu.cluster_unlabelled(features_csv, n_clusters=k, seed=seed, method="kmeans")
    centers = cl["model"].cluster_centers_
    dist_to_own_centroid = np.linalg.norm(cl["X"] - centers[cl["cluster"]], axis=1)
    cl["df"] = cl["df"].copy()
    cl["df"]["cluster"] = cl["cluster"]
    cl["df"]["dist_to_centroid"] = dist_to_own_centroid
    return cl


def correlation_with_labels(clustered: dict, labels_csv: str = pu.LABELS_CSV,
                            keep_classes: list | None = None) -> dict:
    """
    Full correlation picture: raw crosstab, row-normalised (what each cluster is
    made of) and column-normalised (where each label's plaques end up), plus the
    summary stats (ARI/AMI/purity) and a permutation null for the ARI.
    """
    cmp_ = pu.compare_to_labels(clustered, labels_csv, keep_classes)
    perm = pu.label_permutation_null(clustered, labels_csv, keep_classes)
    ct = cmp_["crosstab"]
    row_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    col_pct = (ct.div(ct.sum(axis=0), axis=1) * 100).round(1)
    return {"crosstab": ct, "row_pct": row_pct, "col_pct": col_pct,
           "ari": cmp_["ari"], "ami": cmp_["ami"], "purity": cmp_["purity"],
           "n": cmp_["n"], "permutation": perm}


def representative_samples(clustered: dict, n_samples: int = 20,
                           method: str = "centroid", seed: int = 0,
                           labels_csv: str = pu.LABELS_CSV,
                           image_manifest: str = IMAGE_MANIFEST) -> pd.DataFrame:
    """
    n_samples plaques per cluster, with their image path and true label attached.

    method : "centroid"  closest to the cluster's own centroid (the default -
                         what KMeans treated as most typical of this group)
             "random"    a plain random draw, for comparing against "centroid"
    """
    df = clustered["df"].copy()
    if "label" not in df.columns:
        lab = pd.read_csv(labels_csv)[["plaque_id", "label"]]
        lab["plaque_id"] = lab["plaque_id"].astype(str)
        df = df.merge(lab, on="plaque_id", how="left")
    man = pd.read_csv(image_manifest)[["plaque_id", "path"]]
    man["plaque_id"] = man["plaque_id"].astype(str)
    df = df.merge(man, on="plaque_id", how="left")

    rng = np.random.default_rng(seed)
    rows = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        have_image = sub["path"].notna()
        if have_image.sum() < len(sub):
            sub = sub[have_image]  # only plaques with a rendered crop can be shown
        if method == "centroid":
            sub = sub.sort_values("dist_to_centroid").head(n_samples)
        elif method == "random":
            sub = sub.iloc[rng.permutation(len(sub))].head(n_samples)
        else:
            raise ValueError(f"Unknown method={method!r}. Available: centroid, random")
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    return out[["cluster", "plaque_id", "label", "path", "dist_to_centroid"]]


def contact_sheet(samples: pd.DataFrame, cluster_id: int, cols: int = 5,
                  thumb_px: int = 160, out_path: str | None = None) -> plt.Figure:
    """One grid image per cluster: thumbnails with their true label as a caption."""
    sub = samples[samples["cluster"] == cluster_id].reset_index(drop=True)
    n = len(sub)
    rows = -(-n // cols)  # ceil
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.15))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        r = sub.iloc[i]
        try:
            img = Image.open(r["path"]).convert("RGB").resize((thumb_px, thumb_px))
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        ax.set_title(f"{r['label']}", fontsize=8.5, color="#1a1a1a", pad=2)
    fig.suptitle(f"Cluster {cluster_id} - {n} plaques closest to its centroid "
                f"(most representative members)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        fig.savefig(out_path, dpi=150, facecolor="white")
    return fig


def run(k: int = 3, n_samples: int = 20, method: str = "centroid", seed: int = 0,
       out_dir: str = "results/cluster_samples") -> dict:
    """Full pipeline: cluster, correlate with labels, sample, render contact sheets."""
    os.makedirs(out_dir, exist_ok=True)

    cl = best_kmeans_clusters(k=k, seed=seed)
    corr = correlation_with_labels(cl, keep_classes=pu.MAIN_CLASSES)
    samples = representative_samples(cl, n_samples=n_samples, method=method, seed=seed)
    samples.to_csv(os.path.join(out_dir, f"k{k}_representative_samples.csv"), index=False)

    print(f"KMeans k={k} on {len(cl['df']):,} labelled plaques, "
         f"{len(cl['features'])} morphometric features")
    print(f"ARI={corr['ari']:.3f}  AMI={corr['ami']:.3f}  purity={corr['purity']:.3f}  "
         f"(n={corr['n']} compared against {pu.MAIN_CLASSES})")
    print(corr["permutation"]["message"])
    print()
    print("Raw counts (cluster x label):")
    print(corr["crosstab"].to_string())
    print()
    print("Row-normalised - what each cluster is made of (%):")
    print(corr["row_pct"].to_string())
    print()
    print("Column-normalised - where each label's plaques end up (%):")
    print(corr["col_pct"].to_string())

    sheets = {}
    for c in sorted(samples["cluster"].unique()):
        path = os.path.join(out_dir, f"k{k}_cluster{c}_sample{n_samples}.png")
        contact_sheet(samples, cluster_id=c, out_path=path)
        sheets[c] = path
        print(f"\nwrote {path}")

    return {"clustered": cl, "correlation": corr, "samples": samples, "sheets": sheets}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Inspect the best unsupervised clustering: groups found, "
                   "correlation with labels, and a representative image sample per group.")
    ap.add_argument("--k", type=int, default=3,
                    help="number of KMeans clusters (default 3 - the best-scoring "
                         "setting found in the 6-method comparison)")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="images to sample per cluster (default 20) - increase this "
                         "for a larger contact sheet")
    ap.add_argument("--method", default="centroid", choices=["centroid", "random"],
                    help="centroid = closest to the cluster's own centroid (default, "
                         "most representative); random = plain random draw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/cluster_samples")
    args = ap.parse_args()

    run(k=args.k, n_samples=args.n_samples, method=args.method, seed=args.seed,
       out_dir=args.out_dir)
