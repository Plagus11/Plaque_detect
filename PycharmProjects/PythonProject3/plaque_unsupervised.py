"""
plaque_unsupervised - do the perceived categories correspond to real structure?

The brief was to look at plaques, decide what patterns seemed to be there, and
label accordingly. Supervised accuracy answers a narrower question than that
brief implies: it says how well the measurements predict the labels, which will
be non-zero for almost any consistently applied rule, including one built on a
distinction that does not exist in the tissue.

The direct test is unsupervised. Cluster the unlabelled plaques on morphology
alone, with no access to the labels, then ask whether the annotator's categories
line up with the structure that was already there.

  * They align       -> the perceived categories track something measurable.
  * They cut across  -> the labels may encode a real distinction the current
                        features cannot see, OR a distinction that is not in the
                        tissue. Unsupervised evidence alone cannot separate those
                        two, but it does tell you not to trust the supervised
                        number as confirmation.

    import plaque_unsupervised as pu
    res = pu.run(n_clusters=range(2, 9))
    print(res["summary"])

Reported statistics
-------------------
adjusted Rand index   agreement between two partitions, corrected for chance;
                      0 = as good as random, 1 = identical
adjusted mutual info  as above, information-theoretic; less biased for unequal
                      cluster sizes
purity                largest class share within each cluster, averaged
silhouette            how well-separated the clusters are in feature space,
                      independent of any labels

A high ARI with a low silhouette means the labels agree with clusters that are
themselves weak - suggestive, not conclusive.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import plaque_classifier as pc

FEATURES_CSV = "label_batch_multicase/label_batch_multicase_features.csv"
LABELS_CSV = "labels_label_batch_multicase.csv"
MAIN_CLASSES = ["diffuse", "cored", "diffuse_core_partial"]


def _prep(X: pd.DataFrame) -> np.ndarray:
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler()).fit_transform(X)


def cluster_unlabelled(features_csv: str | pd.DataFrame = FEATURES_CSV, n_clusters: int = 3,
                       features: list | None = None, seed: int = 0,
                       exclude_cases: list | None = None) -> dict:
    """
    Cluster every detected plaque on morphology, ignoring labels entirely.

    Fitted on the full detection table rather than the labelled subset, so the
    structure found is a property of the data and not of the sampling that chose
    what to annotate. features_csv can be a path (the usual case) or an
    already-loaded DataFrame - e.g. from pc.load_detection_tables(), so the
    ~242k-row full dataset does not need writing to disk first.
    """
    df = (features_csv if isinstance(features_csv, pd.DataFrame)
          else pc.load_features_only(features_csv))
    if exclude_cases:
        df = df[~df["wsi_name"].isin(exclude_cases)].reset_index(drop=True)
    feats = [f for f in (features or pc.FEATURES) if f in df.columns]
    X = _prep(df[feats])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    return {"df": df, "X": X, "features": feats, "model": km,
            "cluster": km.labels_, "n_clusters": n_clusters}


def compare_to_labels(clustered: dict, labels_csv: str = LABELS_CSV,
                      keep_classes: list | None = None) -> dict:
    """Cross-tabulate clusters against the annotator's labels and score agreement."""
    lab = pd.read_csv(labels_csv)
    lab["plaque_id"] = lab["plaque_id"].astype(str)

    df = clustered["df"].copy()
    df["cluster"] = clustered["cluster"]
    merged = df.merge(lab[["plaque_id", "label"]], on="plaque_id", how="inner")
    if keep_classes:
        merged = merged[merged["label"].isin(keep_classes)]
    merged = merged[merged["label"] != "unsure"]
    if merged.empty:
        raise ValueError("No labelled plaques left to compare.")

    y, c = merged["label"].values, merged["cluster"].values
    ct = pd.crosstab(merged["cluster"], merged["label"])
    purity = float((ct.max(axis=1).sum()) / ct.values.sum())

    return {"crosstab": ct, "merged": merged,
            "ari": float(adjusted_rand_score(y, c)),
            "ami": float(adjusted_mutual_info_score(y, c)),
            "purity": purity, "n": len(merged)}


def sweep(features_csv: str = FEATURES_CSV, labels_csv: str = LABELS_CSV,
          n_clusters=range(2, 9), keep_classes: list | None = None,
          exclude_cases: list | None = None, seed: int = 0,
          silhouette_sample: int = 5000) -> pd.DataFrame:
    """
    Agreement across a range of k, plus the silhouette of the clustering itself.

    Sweeping matters because a single k can flatter or bury agreement. If ARI
    peaks near the number of classes the annotator used, that is meaningful; if
    it is flat and near zero everywhere, the labels do not track the dominant
    structure in these features.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for k in n_clusters:
        cl = cluster_unlabelled(features_csv, k, seed=seed,
                                exclude_cases=exclude_cases)
        cmp_ = compare_to_labels(cl, labels_csv, keep_classes)
        X, lab = cl["X"], cl["cluster"]
        idx = (rng.choice(len(X), silhouette_sample, replace=False)
               if len(X) > silhouette_sample else np.arange(len(X)))
        sil = float(silhouette_score(X[idx], lab[idx]))
        rows.append({"k": k, "ari": cmp_["ari"], "ami": cmp_["ami"],
                     "purity": cmp_["purity"], "silhouette": sil,
                     "n_labelled": cmp_["n"]})
    return pd.DataFrame(rows)


def label_permutation_null(clustered: dict, labels_csv: str = LABELS_CSV,
                           keep_classes: list | None = None,
                           n_perm: int = 200, seed: int = 0) -> dict:
    """
    What ARI would a meaningless labelling achieve against these clusters?

    ARI is chance-corrected in expectation, but with unequal class sizes and a
    small labelled subset the realised spread is not negligible. This gives the
    null distribution actually applicable to this dataset.
    """
    cmp_ = compare_to_labels(clustered, labels_csv, keep_classes)
    merged = cmp_["merged"]
    y, c = merged["label"].values, merged["cluster"].values

    rng = np.random.default_rng(seed)
    null = np.array([adjusted_rand_score(rng.permutation(y), c)
                     for _ in range(n_perm)])
    p = float(((null >= cmp_["ari"]).sum() + 1) / (n_perm + 1))
    sd = float(null.std(ddof=1))
    z = (cmp_["ari"] - null.mean()) / sd if sd > 0 else np.nan
    return {"ari": cmp_["ari"], "null_mean": float(null.mean()), "null_sd": sd,
            "z": float(z), "p": p, "n_perm": n_perm,
            "message": (f"ARI {cmp_['ari']:.3f} vs shuffled-label null "
                        f"{null.mean():.3f} +/- {sd:.3f} (z = {z:.1f}, p = {p:.3f})")}


def run(features_csv: str = FEATURES_CSV, labels_csv: str = LABELS_CSV,
        n_clusters=range(2, 9), keep_classes: list | None = None,
        exclude_cases: list | None = None, seed: int = 0) -> dict:
    """Full comparison: sweep, best-k crosstab, and a permutation null."""
    keep_classes = keep_classes if keep_classes is not None else MAIN_CLASSES
    table = sweep(features_csv, labels_csv, n_clusters, keep_classes,
                  exclude_cases, seed)
    best_k = int(table.loc[table["ari"].idxmax(), "k"])
    cl = cluster_unlabelled(features_csv, best_k, seed=seed,
                            exclude_cases=exclude_cases)
    cmp_ = compare_to_labels(cl, labels_csv, keep_classes)
    perm = label_permutation_null(cl, labels_csv, keep_classes, seed=seed)

    ari = cmp_["ari"]
    if ari < 0.05:
        verdict = ("The labels do not track the dominant structure in these "
                   "features. Either the perceived categories rest on something "
                   "the 24 measurements do not capture, or they do not correspond "
                   "to a natural grouping in the tissue. The supervised score "
                   "should not be read as confirmation that the categories are real.")
    elif ari < 0.20:
        verdict = ("Weak alignment. Some correspondence between the perceived "
                   "categories and natural structure, but the clusters are largely "
                   "organised on something else.")
    else:
        verdict = ("The perceived categories align with structure that was already "
                   "present in the measurements, found without access to the labels. "
                   "That is independent support for the categories being real.")

    lines = [
        f"Clustered {len(cl['df']):,} unlabelled plaques on {len(cl['features'])} "
        f"morphometric features; compared against {cmp_['n']} labels.",
        "",
        table.round(3).to_string(index=False),
        "",
        f"best agreement at k={best_k}: ARI {ari:.3f}, AMI {cmp_['ami']:.3f}, "
        f"purity {cmp_['purity']:.3f}",
        perm["message"],
        "",
        "cluster x label:",
        cmp_["crosstab"].to_string(),
        "",
        verdict,
    ]
    return {"table": table, "best_k": best_k, "comparison": cmp_,
            "permutation": perm, "summary": "\n".join(lines)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Do the annotator's categories match unsupervised structure?")
    ap.add_argument("--features", default=FEATURES_CSV)
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--exclude-cases", default="Case_5_1_ABeta",
                    help="comma-separated cases to drop (QC failures)")
    ap.add_argument("--all-classes", action="store_true",
                    help="compare against all labels, not just the three main types")
    ap.add_argument("--out", default="results/unsupervised_agreement.csv")
    args = ap.parse_args()

    res = run(args.features, args.labels,
              range(args.kmin, args.kmax + 1),
              keep_classes=None if args.all_classes else MAIN_CLASSES,
              exclude_cases=[c for c in args.exclude_cases.split(",") if c])
    print(res["summary"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res["table"].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
