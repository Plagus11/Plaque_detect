"""
plaque_fusion - two-branch classification, combined at the decision level.

Branch A: the 24 morphometric features from plaque_features (shape, stain, core).
Branch B: image features from the scale-normalised crops (HOG / colour / LBP, or
          pretrained embeddings if torch is available).

Earlier attempts glued the two feature sets into one matrix. That failed badly -
0.570 morphometry alone dropped to 0.321 once 1876 image dimensions were
concatenated, because an RBF kernel weights every standardised dimension equally
and the informative 24 drowned. Late fusion avoids that: each branch is trained
in its own feature space, and only the two probability vectors are combined.

    import plaque_fusion as pf
    res = pf.run_fusion()
    print(res["summary"])


HOW THIS AVOIDS FOOLING ITSELF
------------------------------
1. Both branches are evaluated on the SAME case-grouped folds, generated once and
   passed to each. Without that the two probability vectors describe different
   held-out sets and combining them is meaningless.
2. Averaging fits nothing, so it cannot leak.
3. Stacking fits a meta-learner, which CAN leak: train it on the same predictions
   it is later scored on and it will look excellent. Here the meta-features for
   each outer training set come from an INNER case-grouped CV, so the meta-learner
   never sees a prediction made by a model that saw the outer test fold.
4. Improvements are tested with a paired bootstrap over plaques, not by comparing
   two point estimates. At n~500 a 0.03 difference is noise.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import plaque_classifier as pc
import plaque_images as pi

# -------------------------
# CONFIG
# -------------------------

FEATURES_CSV = "label_batch_multicase/label_batch_multicase_features.csv"
LABELS_CSV = "labels_label_batch_multicase.csv"
MANIFEST = "image_dataset/manifest.csv"
CACHE_DIR = "results/cache"

MORPH_MODEL = "svm"     # best tabular branch from compare_models
IMAGE_MODEL = "svm"


# -------------------------
# Shared folds
# -------------------------

def make_folds(y: pd.Series, groups: pd.Series, n_splits: int = 5,
               seed: int = 0) -> list:
    """
    One fold assignment, reused by every branch and every fusion rule.

    Grouped on wsi_name so no case appears in both sides. Falls back to ungrouped
    only if there are too few cases, and says so.
    """
    n_splits = max(2, min(n_splits, int(y.value_counts().min())))
    if groups.nunique() >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y, groups))
    warnings.warn(f"Only {groups.nunique()} case group(s) for {n_splits} folds - "
                  f"using ungrouped folds; scores will be optimistic.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


# -------------------------
# Branches
# -------------------------

def branch_oof(X: pd.DataFrame, y: pd.Series, folds: list, kind: str = "svm",
               seed: int = 0) -> tuple:
    """
    Out-of-fold probabilities for one branch.

    Returns (proba array [n x n_classes], class order). Every row is predicted by
    a model that never saw it.
    """
    classes = np.array(sorted(y.unique()))
    proba = np.full((len(y), len(classes)), np.nan)
    for tr, te in folds:
        model = pc.build_model(kind, seed=seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])
        # Re-order to the global class order; a fold can miss a rare class.
        cols = {c: i for i, c in enumerate(model.classes_)}
        for j, c in enumerate(classes):
            proba[te, j] = p[:, cols[c]] if c in cols else 0.0
    return proba, classes


def _inner_meta_features(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                         idx: np.ndarray, kind: str, n_splits: int,
                         seed: int) -> np.ndarray:
    """Inner-CV out-of-fold probabilities for a training subset (for stacking)."""
    classes = np.array(sorted(y.unique()))
    sub_y = y.iloc[idx].reset_index(drop=True)
    sub_g = groups.iloc[idx].reset_index(drop=True)
    sub_X = X.iloc[idx].reset_index(drop=True)
    out = np.zeros((len(idx), len(classes)))
    for tr, te in make_folds(sub_y, sub_g, n_splits=n_splits, seed=seed):
        model = pc.build_model(kind, seed=seed)
        model.fit(sub_X.iloc[tr], sub_y.iloc[tr])
        p = model.predict_proba(sub_X.iloc[te])
        cols = {c: i for i, c in enumerate(model.classes_)}
        for j, c in enumerate(classes):
            out[te, j] = p[:, cols[c]] if c in cols else 0.0
    return out


# -------------------------
# Fusion rules
# -------------------------

def fuse_average(pa: np.ndarray, pb: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Weighted soft vote. weight is the share given to branch A."""
    return weight * pa + (1.0 - weight) * pb


def fuse_stack(Xa: pd.DataFrame, Xb: pd.DataFrame, y: pd.Series, groups: pd.Series,
               folds: list, kind_a: str = "svm", kind_b: str = "svm",
               inner_splits: int = 3, seed: int = 0) -> np.ndarray:
    """
    Stacking with an inner CV, so the meta-learner is never trained on predictions
    from a model that saw the outer test fold.

    The meta-learner is a plain multinomial logistic regression on the 2*k
    probabilities. Anything more flexible overfits at this sample size.
    """
    classes = np.array(sorted(y.unique()))
    out = np.full((len(y), len(classes)), np.nan)

    for tr, te in folds:
        meta_tr = np.hstack([
            _inner_meta_features(Xa, y, groups, tr, kind_a, inner_splits, seed),
            _inner_meta_features(Xb, y, groups, tr, kind_b, inner_splits, seed),
        ])
        meta = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                  random_state=seed)
        meta.fit(meta_tr, y.iloc[tr])

        # Base models refit on the whole outer training set, applied to the test fold.
        parts = []
        for X, kind in ((Xa, kind_a), (Xb, kind_b)):
            m = pc.build_model(kind, seed=seed)
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[te])
            cols = {c: i for i, c in enumerate(m.classes_)}
            parts.append(np.column_stack(
                [p[:, cols[c]] if c in cols else np.zeros(len(te)) for c in classes]))
        meta_te = np.hstack(parts)

        p = meta.predict_proba(meta_te)
        cols = {c: i for i, c in enumerate(meta.classes_)}
        for j, c in enumerate(classes):
            out[te, j] = p[:, cols[c]] if c in cols else 0.0
    return out


# -------------------------
# Significance
# -------------------------

def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Is B better than A? Resample plaques (not folds) and recompute the difference.

    Paired, because both models predicted the same plaques - comparing independent
    confidence intervals would be far too conservative and would miss real, if
    small, differences.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            diffs[i] = 0.0
            continue
        diffs[i] = (balanced_accuracy_score(y_true[idx], pred_b[idx])
                    - balanced_accuracy_score(y_true[idx], pred_a[idx]))
    observed = (balanced_accuracy_score(y_true, pred_b)
                - balanced_accuracy_score(y_true, pred_a))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": float(observed), "lo95": float(lo), "hi95": float(hi),
            "p": float(min(1.0, p)),
            "verdict": ("improves" if lo > 0 else
                        "hurts" if hi < 0 else
                        "no detectable difference")}


# -------------------------
# Pipeline
# -------------------------

def load_branches(features_csv: str = FEATURES_CSV, labels_csv: str = LABELS_CSV,
                  manifest: str = MANIFEST, image_kind: str = "hog+colour",
                  merge: dict | None = None, drop: list | None = None,
                  cache: bool = True,
                  image_features_df: pd.DataFrame | None = None) -> dict:
    """
    Assemble both branches on identical, aligned rows.

    Alignment is on plaque_id - the morphometric table and the image manifest are
    written by different scripts and are not guaranteed to share a row order.

    image_features_df : a plaque_id-indexed embedding table (e.g. a pathology
        foundation model's output) to use as branch B instead of the classical
        HOG/colour features named by image_kind. image_kind is then only a
        label in the printed summary, not a recipe.
    """
    man = pd.read_csv(manifest)
    tab = pc.load_labelled(features_csv, labels_csv)

    common = [p for p in man["plaque_id"] if p in set(tab["plaque_id"])
             and (image_features_df is None or p in image_features_df.index)]
    man = man.set_index("plaque_id").loc[common].reset_index()
    tab = tab.set_index("plaque_id").loc[common].reset_index()
    assert (man["plaque_id"].values == tab["plaque_id"].values).all()

    if merge or drop:
        tab = pc.merge_classes(tab, merge or {}, drop=drop)
        keep = set(tab["plaque_id"])
        man = man[man["plaque_id"].isin(keep)].reset_index(drop=True)
        tab = tab.set_index("plaque_id").loc[man["plaque_id"]].reset_index()

    # 'unsure' is a labelling outcome, never a class.
    ok = tab["label"] != "unsure"
    tab, man = tab[ok].reset_index(drop=True), man[ok.values].reset_index(drop=True)

    counts = tab["label"].value_counts()
    rare = counts[counts < 5].index.tolist()
    if rare:
        warnings.warn(f"Dropping class(es) with <5 examples: {rare}")
        ok = ~tab["label"].isin(rare)
        tab, man = tab[ok].reset_index(drop=True), man[ok.values].reset_index(drop=True)

    if image_features_df is not None:
        Xb = image_features_df.loc[man["plaque_id"]].reset_index(drop=True)
        return {"Xa": tab[pc.FEATURES].reset_index(drop=True),
                "Xb": Xb,
                "y": tab["label"].reset_index(drop=True),
                "groups": tab["wsi_name"].reset_index(drop=True),
                "plaque_id": tab["plaque_id"].reset_index(drop=True)}

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"img_{image_kind.replace('+', '_')}.pkl")
    if cache and os.path.exists(cache_path):
        Xb = pd.read_pickle(cache_path)
        Xb = Xb.loc[Xb.index[:0].union(Xb.index)]  # no-op, keeps dtype
        Xb = Xb.reset_index(drop=True)
        if len(Xb) != len(man):
            Xb = None
    else:
        Xb = None
    if Xb is None:
        full_man = pd.read_csv(manifest)
        Xfull = pi.image_features(full_man, kind=image_kind, progress=False)
        Xfull.index = full_man["plaque_id"].values
        Xb = Xfull.loc[man["plaque_id"]].reset_index(drop=True)
        if cache:
            Xb.to_pickle(cache_path)

    return {"Xa": tab[pc.FEATURES].reset_index(drop=True),
            "Xb": Xb,
            "y": tab["label"].reset_index(drop=True),
            "groups": tab["wsi_name"].reset_index(drop=True),
            "plaque_id": tab["plaque_id"].reset_index(drop=True)}


def run_fusion(features_csv: str = FEATURES_CSV, labels_csv: str = LABELS_CSV,
               manifest: str = MANIFEST, image_kind: str = "hog+colour",
               morph_model: str = MORPH_MODEL, image_model: str = IMAGE_MODEL,
               n_splits: int = 5, seed: int = 0, merge: dict | None = None,
               drop: list | None = None, do_stack: bool = True,
               image_features_df: pd.DataFrame | None = None) -> dict:
    """
    Run both branches and every fusion rule on one shared set of folds.

    Returns per-strategy out-of-fold predictions, scores, per-class stats and a
    paired test of each fusion against the better single branch.

    image_features_df : see load_branches - pass a plaque_id-indexed foundation
        model embedding table to fuse against that instead of classical
        HOG/colour image features.
    """
    b = load_branches(features_csv, labels_csv, manifest, image_kind, merge, drop,
                      image_features_df=image_features_df)
    Xa, Xb, y, groups = b["Xa"], b["Xb"], b["y"], b["groups"]
    classes = np.array(sorted(y.unique()))
    folds = make_folds(y, groups, n_splits=n_splits, seed=seed)

    print(f"{len(y)} plaques, {len(classes)} classes, {groups.nunique()} cases, "
          f"{len(folds)} folds")
    print(f"branch A: morphometry, {Xa.shape[1]} features, model={morph_model}")
    print(f"branch B: {image_kind}, {Xb.shape[1]} features, model={image_model}\n")

    pa, _ = branch_oof(Xa, y, folds, morph_model, seed)
    pb, _ = branch_oof(Xb, y, folds, image_model, seed)

    strategies = {
        "morphology only": pa,
        "image only": pb,
        "average 50/50": fuse_average(pa, pb, 0.5),
        "weighted 75/25": fuse_average(pa, pb, 0.75),
        "weighted 90/10": fuse_average(pa, pb, 0.90),
    }
    if do_stack:
        strategies["stacked (inner CV)"] = fuse_stack(
            Xa, Xb, y, groups, folds, morph_model, image_model, seed=seed)

    yt = y.values
    rows, preds = [], {}
    for name, proba in strategies.items():
        pred = classes[np.nanargmax(proba, axis=1)]
        preds[name] = pred
        rows.append({
            "strategy": name,
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
            "accuracy": float((pred == yt).mean()),
        })
    scores = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)

    # Every strategy tested against the stronger single branch.
    base = ("morphology only"
            if scores.set_index("strategy").loc["morphology only", "balanced_accuracy"]
            >= scores.set_index("strategy").loc["image only", "balanced_accuracy"]
            else "image only")
    tests = []
    for name in strategies:
        if name == base:
            continue
        t = paired_bootstrap(yt, preds[base], preds[name], seed=seed)
        tests.append({"strategy": name, "vs": base, **t})
    tests = pd.DataFrame(tests)

    lines = [scores.round(3).to_string(index=False), "",
             f"Paired bootstrap against '{base}' (2000 resamples of the plaques):"]
    for _, r in tests.iterrows():
        lines.append(f"  {r['strategy']:<22} delta {r['delta']:+.3f} "
                     f"[{r['lo95']:+.3f}, {r['hi95']:+.3f}]  p={r['p']:.3f}  "
                     f"-> {r['verdict']}")
    if (tests["verdict"] == "improves").sum() == 0:
        lines.append("\nNo fusion rule beats the better single branch. With image "
                     "features at chance that is the expected result, not a bug.")

    return {"scores": scores, "tests": tests, "preds": preds, "y": yt,
             "classes": classes, "base": base, "folds": folds,
             "branches": b, "summary": "\n".join(lines)}


def plot_fusion(res: dict, ax=None) -> Figure:
    """Balanced accuracy per strategy, single branches marked."""
    with mpl.rc_context(pc._RC):
        owned = ax is None
        s = res["scores"].iloc[::-1]
        fig, ax = (plt.subplots(figsize=(7.6, 0.46 * len(s) + 1.8)) if owned
                   else (ax.figure, ax))
        colours = [pc.PALETTE[9] if "only" in m else pc.PALETTE[0]
                   for m in s["strategy"]]
        y = np.arange(len(s))
        ax.barh(y, s["balanced_accuracy"], color=colours, height=0.66, zorder=3)
        for i, v in enumerate(s["balanced_accuracy"]):
            ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
        ax.set_yticks(y, s["strategy"])
        chance = 1.0 / len(res["classes"])
        ax.axvline(chance, color=pc.PALETTE[2], ls="--", lw=1.3,
                   label=f"chance ({chance:.2f})")
        ax.set_xlim(0, max(0.9, s["balanced_accuracy"].max() * 1.18))
        ax.set_xlabel("Balanced accuracy (out-of-fold, case-grouped)")
        ax.set_title("Late fusion: morphology + image")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Two-branch late fusion.")
    ap.add_argument("--features", default=FEATURES_CSV)
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--image-kind", default="hog+colour")
    ap.add_argument("--morph-model", default=MORPH_MODEL)
    ap.add_argument("--image-model", default=IMAGE_MODEL)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--merge-core", action="store_true",
                    help="collapse core_halo/diffuse_core_partial into cored")
    ap.add_argument("--drop", default="", help="comma-separated classes to drop")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    res = run_fusion(
        args.features, args.labels, args.manifest, args.image_kind,
        args.morph_model, args.image_model, n_splits=args.splits,
        merge=pc.MERGE_CORE_LIKE if args.merge_core else None,
        drop=[c for c in args.drop.split(",") if c],
        do_stack=not args.no_stack)

    print(res["summary"])
    os.makedirs(args.outdir, exist_ok=True)
    res["scores"].to_csv(os.path.join(args.outdir, "fusion_scores.csv"), index=False)
    res["tests"].to_csv(os.path.join(args.outdir, "fusion_tests.csv"), index=False)
    fig = plot_fusion(res)
    fig.savefig(os.path.join(args.outdir, "fusion.png"), dpi=180, bbox_inches="tight")
    print(f"\nWrote {args.outdir}/fusion_scores.csv, fusion_tests.csv, fusion.png")
