"""
plaque_classifier - supervised plaque-type classification from the feature table.

Built for a small label budget (a few hundred plaques). At that size the model is
not the hard part; not fooling yourself is. So this module leads with the honest
evaluation machinery and treats the classifier itself as swappable:

  * case-grouped CV      - plaques on one slide share staining and scanning, so a
                           random split leaks. Splits are grouped on wsi_name.
  * repeated CV + CI     - one 5-fold score at n=300 is noise. Everything is
                           repeated and reported with a confidence interval.
  * permutation baseline - refits on shuffled labels to show what "chance" is for
                           YOUR class balance. If the real score sits inside the
                           null, you have nothing.
  * learning curve       - answers the only question that matters at this budget:
                           will another 200 labels help, or have you plateaued?
  * active learning      - suggest_to_label() ranks unlabelled plaques by model
                           uncertainty, so the next labels you buy are the useful
                           ones.

Typical loop:

    import plaque_classifier as pc

    df    = pc.load_labelled("features.csv", "labels.csv")
    ds    = pc.make_dataset(df)
    rep   = pc.evaluate(ds)                  # honest metrics before you believe anything
    print(rep["summary"])

    model = pc.fit_final(ds)
    pred  = pc.predict_unlabelled(model, pc.load_features_only("features.csv"))
    todo  = pc.suggest_to_label(model, unlabelled, n=100)   # next annotation batch

Requires pandas, numpy, scikit-learn, matplotlib. LightGBM is used if installed,
otherwise sklearn's HistGradientBoostingClassifier - same family of algorithm, no
extra dependency, near-identical behaviour at this sample size.


LABELS
------
Expected labels.csv: one row per labelled plaque, with a key that joins onto the
feature table plus a `label` column. Join key is `plaque_id` if present, else the
(wsi_name, x1, y1) triple that every detection CSV in this project carries.

Suggested classes for Abeta IHC (see CLASSES): diffuse, cored, coarse_grained,
CAA, artefact. Neuritic plaques are deliberately absent - they are defined by
dystrophic neurites and cannot be identified from Abeta staining alone.

Do NOT bootstrap labels from `has_black`. It fires on ~88% of plaques in
Case_6_1, which inverts the expected diffuse-dominant ratio; training on it would
just relearn a mis-set grey threshold.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

try:  # optional - better with correlated tabular features, but not required
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:  # pragma: no cover - depends on the local environment
    HAVE_LGBM = False

# -------------------------
# CONFIG
# -------------------------

# Kept in step with plaque_labeller.CLASSES - that module is the source of truth
# for what the annotator actually sees.
CLASSES = ["diffuse", "cored", "core_halo", "diffuse_core_partial",
           "nucleus_centre", "CAA", "artefact"]

# cored / core_halo / diffuse_core_partial sample one continuum rather than three
# discrete kinds. If the confusion matrix shows they are confused mainly with each
# other, merging recovers the statistical power spent on a distinction the features
# cannot support: make_dataset(merge_classes(df, MERGE_CORE_LIKE)).
MERGE_CORE_LIKE = {"core_halo": "cored", "diffuse_core_partial": "cored"}
MERGE_TO_BINARY = {"core_halo": "cored", "diffuse_core_partial": "diffuse",
                   "nucleus_centre": "diffuse"}

# Morphometric features from plaque_features.py. Identifiers, pixel duplicates and
# absolute positions are excluded: position is a shortcut the model would happily
# memorise, and it does not generalise to a new slide.
FEATURES = [
    "area_um2", "diameter_um", "perimeter_um", "circularity", "extent",
    "aspect_ratio", "box_w_um", "box_h_um",
    "mean_optical_density", "std_optical_density", "fill_fraction",
    "mean_gray", "std_gray", "mean_hue", "std_hue",
    "mean_saturation", "std_saturation", "mean_value",
    "mean_R", "mean_G", "mean_B",
    "black_fraction", "black_area_um2", "black_offset_um",
]

# Features derived from the DARK_GRAY_THRESH core detector. Legitimate predictors,
# but they encode the same threshold that looks miscalibrated - drop them with
# drop_black=True to check the model is not leaning entirely on it.
BLACK_FEATURES = ["black_fraction", "black_area_um2", "black_offset_um"]

GROUP_COL = "wsi_name"
LABEL_COL = "label"

PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951",
           "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0"]

_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c8ced6", "axes.grid": True, "grid.color": "#eef1f4",
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": "#6b7280", "ytick.color": "#6b7280",
    "legend.frameon": False, "figure.dpi": 110,
}


# -------------------------
# Data assembly
# -------------------------

@dataclass
class Dataset:
    """Feature matrix, labels, group vector and the metadata needed to report."""
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    features: list
    classes: list = field(default_factory=list)

    def __post_init__(self):
        if not self.classes:
            self.classes = sorted(self.y.unique().tolist())

    @property
    def n_groups(self) -> int:
        return int(self.groups.nunique())

    def describe(self) -> str:
        counts = self.y.value_counts()
        lines = [f"{len(self.X):,} labelled plaques, {len(self.features)} features, "
                 f"{self.n_groups} case group(s)"]
        for cls, n in counts.items():
            lines.append(f"  {cls:<16} {n:>6,}  ({100 * n / len(self.y):.1f}%)")
        smallest = counts.min()
        if smallest < 10:
            lines.append(f"  ! smallest class has {smallest} example(s) - too few to "
                         f"evaluate; consider merging it or labelling more.")
        if self.n_groups < 2:
            lines.append("  ! only one case: CV cannot estimate cross-case generalisation.")
        return "\n".join(lines)


def load_features_only(features_csv: str) -> pd.DataFrame:
    """Read a feature CSV and attach a stable `plaque_id` if one is not present."""
    df = pd.read_csv(features_csv)
    # plaque_id must never be coerced: it is a string key, and errors="coerce"
    # would silently turn the whole column into NaN and break every later join.
    keep_text = {"wsi_name", "crop_path", "plaque_id", LABEL_COL}
    for col in df.columns:
        if col not in keep_text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "plaque_id" not in df.columns:
        df["plaque_id"] = _make_id(df)
    df["plaque_id"] = df["plaque_id"].astype(str)
    return df


def load_detection_tables(boxes_dir: str = "wsi_plaque_boxes_variable",
                          pattern: str = "*_plaque_boxes_variable.csv",
                          exclude_cases: list | None = None) -> pd.DataFrame:
    """
    Concatenate every per-case detection CSV into one full-dataset table.

    This is the ~242k-plaque table the detector wrote, as opposed to
    load_features_only() pointed at label_batch_multicase's 500-row sample.
    Each CSV lacks a plaque_id column, so it is generated the same way
    load_features_only() would, after concatenation (not per-file - a
    plaque_id must stay unique across cases, which _make_id already
    guarantees by including wsi_name).

    exclude_cases : wsi_name values to drop, e.g. a case whose tissue mask is
        known to have failed QC.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(boxes_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {boxes_dir!r}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if exclude_cases:
        df = df[~df["wsi_name"].isin(exclude_cases)].reset_index(drop=True)
    keep_text = {"wsi_name", "crop_path", "plaque_id", LABEL_COL}
    for col in df.columns:
        if col not in keep_text:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["plaque_id"] = _make_id(df)
    return df


def _make_id(df: pd.DataFrame) -> pd.Series:
    """
    Deterministic id from slide + full level-0 box - stable across re-runs.

    The whole box is used, not just the top-left corner: in Case_6_1 four pairs of
    distinct plaques share an (x1, y1) while differing in (x2, y2), so a corner-only
    key silently collides and would mis-assign labels.
    """
    need = ["wsi_name", "x1", "y1", "x2", "y2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot build plaque_id, missing {missing}. "
                       f"Add a plaque_id column to both files instead.")
    ids = (df["wsi_name"].astype(str)
           + "_x" + df["x1"].astype(int).astype(str)
           + "_y" + df["y1"].astype(int).astype(str)
           + "_w" + (df["x2"].astype(int) - df["x1"].astype(int)).astype(str)
           + "_h" + (df["y2"].astype(int) - df["y1"].astype(int)).astype(str))
    dupes = int(ids.duplicated().sum())
    if dupes:
        warnings.warn(f"{dupes} plaque_id collision(s) even using the full box. "
                      f"Labels for those rows cannot be assigned unambiguously - add "
                      f"an explicit plaque_id column to your detection output.")
    return ids


def load_labelled(features_csv: str, labels_csv: str,
                  classes: list | None = None) -> pd.DataFrame:
    """
    Join labels onto features. Reports unmatched labels loudly rather than
    silently dropping them - a broken join is the most common way to end up
    training on 40 plaques and not noticing.
    """
    feats = load_features_only(features_csv)
    labs = pd.read_csv(labels_csv)
    if LABEL_COL not in labs.columns:
        raise KeyError(f"{labels_csv} has no `{LABEL_COL}` column.")

    if "plaque_id" not in labs.columns:
        labs["plaque_id"] = _make_id(labs)
    labs["plaque_id"] = labs["plaque_id"].astype(str)

    labs = labs[["plaque_id", LABEL_COL]].dropna(subset=[LABEL_COL])
    labs[LABEL_COL] = labs[LABEL_COL].astype(str).str.strip()

    dupes = labs["plaque_id"].duplicated().sum()
    if dupes:
        warnings.warn(f"{dupes} duplicate plaque_id(s) in labels; keeping the first.")
        labs = labs.drop_duplicates("plaque_id", keep="first")

    merged = feats.merge(labs, on="plaque_id", how="inner")
    unmatched = len(labs) - len(merged)
    if unmatched:
        warnings.warn(f"{unmatched} of {len(labs)} label(s) did not match any feature "
                      f"row. Check that both files come from the same detection run.")
    if merged.empty:
        looks_blind = labs["plaque_id"].astype(str).str.match(r"^[\w\-]+-\d{4}$").mean()
        hint = ("\n\nThese ids look like blind display ids (e.g. 'batch-0007'). "
                "Run plaque_labeller.unblind_labels(labels_csv, key_csv) first - "
                "the key was written next to the batch HTML."
                if looks_blind > 0.8 else "")
        raise ValueError("No labels matched the feature table - the join key is "
                         "wrong." + hint)

    known = set(classes or CLASSES)
    unknown = sorted(set(merged[LABEL_COL]) - known)
    if unknown:
        warnings.warn(f"Label value(s) not in CLASSES: {unknown}. They are kept; add "
                      f"them to CLASSES if intentional.")
    return merged


def merge_classes(df: pd.DataFrame, mapping: dict, drop: list | None = None
                  ) -> pd.DataFrame:
    """
    Collapse fine-grained labels into coarser ones without relabelling anything.

    The point of labelling at high granularity and merging afterwards is that it
    is reversible: you can always go from seven classes to three, never the other
    way. So label the distinctions you can see, then merge whichever ones the
    features turn out not to support.

        coarse = pc.merge_classes(df, pc.MERGE_CORE_LIKE)
        ds     = pc.make_dataset(coarse)

    drop : label values to remove entirely, e.g. ["artefact"] to model only real
        deposits, or ["unsure"] which is dropped automatically.
    """
    out = df.copy()
    drop = list(drop or [])
    if "unsure" not in drop:
        drop.append("unsure")
    out = out[~out[LABEL_COL].isin(drop)]
    out[LABEL_COL] = out[LABEL_COL].replace(mapping)
    if out.empty:
        raise ValueError("Every row was dropped - check `drop` and the label values.")
    return out.reset_index(drop=True)


def make_dataset(df: pd.DataFrame, features: list | None = None,
                 drop_black: bool = False, min_per_class: int = 5) -> Dataset:
    """
    Build X / y / groups. Classes with fewer than `min_per_class` examples are
    dropped, because they cannot appear in every CV fold and their reported
    per-class scores would be meaningless.
    """
    feats = list(features or FEATURES)
    if drop_black:
        feats = [f for f in feats if f not in BLACK_FEATURES]
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise KeyError("None of the requested feature columns are in the DataFrame.")
    if LABEL_COL not in df.columns:
        raise KeyError(f"No `{LABEL_COL}` column - use load_labelled() first.")

    work = df.dropna(subset=[LABEL_COL]).copy()
    # "unsure" is a labelling outcome, not a class. Training on it would teach the
    # model to reproduce the annotator's hesitation.
    n_unsure = int((work[LABEL_COL] == "unsure").sum())
    if n_unsure:
        warnings.warn(f"Dropping {n_unsure} plaque(s) labelled 'unsure'.")
        work = work[work[LABEL_COL] != "unsure"]
    counts = work[LABEL_COL].value_counts()
    too_few = counts[counts < min_per_class]
    if len(too_few):
        warnings.warn(f"Dropping class(es) with <{min_per_class} examples: "
                      f"{dict(too_few)}")
        work = work[~work[LABEL_COL].isin(too_few.index)]
    if work[LABEL_COL].nunique() < 2:
        raise ValueError("Fewer than two usable classes remain.")

    groups = (work[GROUP_COL] if GROUP_COL in work.columns
              else pd.Series(["all"] * len(work), index=work.index))
    return Dataset(X=work[feats].reset_index(drop=True),
                   y=work[LABEL_COL].reset_index(drop=True),
                   groups=groups.reset_index(drop=True),
                   features=feats)


# -------------------------
# Model
# -------------------------

# Every entry is a (constructor, needs_scaling) pair. Hyperparameters are
# deliberately conservative: these are sized for a few hundred rows, where the
# usual library defaults overfit badly and run slower for no benefit.
#
# Add your own with register_model(); everything downstream - evaluate,
# permutation_baseline, learning_curve, compare_models - works unchanged.
MODELS = {}


def register_model(name: str, factory, needs_scaling: bool = False,
                   description: str = "") -> None:
    """Add a classifier to the registry. `factory` takes a seed, returns an estimator."""
    MODELS[name] = {"factory": factory, "scale": needs_scaling,
                    "description": description}


def _register_defaults() -> None:
    if HAVE_LGBM:
        register_model(
            "lgbm",
            lambda seed: LGBMClassifier(
                n_estimators=120, learning_rate=0.08, num_leaves=8,
                min_child_samples=8, subsample=0.85, subsample_freq=1,
                colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced",
                random_state=seed, verbose=-1, n_jobs=1),
            description="LightGBM gradient boosting")

    register_model(
        "hgb",
        lambda seed: HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=8,
            min_samples_leaf=8, l2_regularization=1.0, early_stopping=False,
            class_weight="balanced", random_state=seed),
        description="sklearn histogram gradient boosting")

    register_model(
        "rf",
        lambda seed: RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1),
        description="random forest")

    register_model(
        "extra",
        lambda seed: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1),
        description="extremely randomised trees")

    # Linear/kernel/instance models need features on a common scale - the raw
    # columns span area_um2 in the thousands and circularity in [0, 1].
    register_model(
        "logreg",
        lambda seed: LogisticRegression(
            C=0.5, max_iter=2000, class_weight="balanced", random_state=seed),
        needs_scaling=True, description="L2 logistic regression")

    register_model(
        "svm",
        lambda seed: SVC(
            C=2.0, gamma="scale", kernel="rbf", class_weight="balanced",
            probability=True, random_state=seed),
        needs_scaling=True, description="RBF support vector machine")

    register_model(
        "knn",
        lambda seed: KNeighborsClassifier(n_neighbors=7, weights="distance"),
        needs_scaling=True, description="k-nearest neighbours")

    # early_stopping stays off: in sklearn 1.7 the internal validation score calls
    # np.isnan on the predicted labels, which raises TypeError for string classes
    # like "diffuse". Regularisation via alpha does the same job here, and at a few
    # hundred rows an MLP is a long shot against the trees anyway.
    register_model(
        "mlp",
        lambda seed: MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=600,
            early_stopping=False, random_state=seed),
        needs_scaling=True, description="small multilayer perceptron")

    # The number every other model must beat. A classifier that cannot beat
    # "always predict the majority class" has learned nothing.
    register_model(
        "dummy",
        lambda seed: DummyClassifier(strategy="prior", random_state=seed),
        description="majority-class baseline")


_register_defaults()


def available_models() -> pd.DataFrame:
    """List the registry, so you can see what --model accepts."""
    return pd.DataFrame([{"name": k, "scaled": v["scale"],
                          "description": v["description"]}
                         for k, v in MODELS.items()])


def build_model(kind: str = "auto", seed: int = 0, calibrate: bool = False) -> Pipeline:
    """
    Build any registered classifier as a ready-to-fit Pipeline.

    kind : a key of MODELS, or "auto" for the best available gradient booster.
    calibrate : wrap in sigmoid calibration. Only worth it above ~500 labels;
        below that the internal calibration split costs more than it buys.

    The returned Pipeline always imputes first - trees tolerate NaN, but
    permutation_importance and every scaled model do not.
    """
    if kind == "auto":
        kind = "lgbm" if HAVE_LGBM else "hgb"
    if kind not in MODELS:
        raise ValueError(f"Unknown model {kind!r}. Available: {sorted(MODELS)}")

    spec = MODELS[kind]
    clf = spec["factory"](seed)
    if calibrate:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)

    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec["scale"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def _splitter(ds: Dataset, n_splits: int, seed: int):
    """
    Case-grouped stratified folds, degrading loudly when the data cannot support
    them rather than silently producing a leaky split.
    """
    n_splits = max(2, min(n_splits, int(ds.y.value_counts().min())))
    if ds.n_groups >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed), ds.groups
    warnings.warn(
        f"Only {ds.n_groups} case group(s) for {n_splits} folds - falling back to "
        f"ungrouped StratifiedKFold. Scores will be optimistic because plaques from "
        f"the same slide appear in both train and test.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), None


# -------------------------
# Evaluation
# -------------------------

def evaluate(ds: Dataset, kind: str = "auto", n_splits: int = 5,
             n_repeats: int = 4, seed: int = 0) -> dict:
    """
    Repeated case-grouped CV. Returns per-fold scores, means with 95% intervals,
    an aggregated confusion matrix and out-of-fold predictions.
    """
    rows, oof_true, oof_pred, oof_proba, oof_idx = [], [], [], [], []

    for rep in range(n_repeats):
        cv, groups = _splitter(ds, n_splits, seed + rep)
        for fold, (tr, te) in enumerate(cv.split(ds.X, ds.y, groups)):
            model = build_model(kind, seed=seed + rep)
            model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
            pred = model.predict(ds.X.iloc[te])
            rows.append({
                "repeat": rep, "fold": fold, "n_test": len(te),
                "balanced_accuracy": balanced_accuracy_score(ds.y.iloc[te], pred),
                "macro_f1": f1_score(ds.y.iloc[te], pred, average="macro",
                                     zero_division=0),
                "mcc": matthews_corrcoef(ds.y.iloc[te], pred),
                "kappa": cohen_kappa_score(ds.y.iloc[te], pred),
            })
            if rep == 0:
                oof_true.extend(ds.y.iloc[te].tolist())
                oof_pred.extend(pred.tolist())
                oof_idx.extend(te.tolist())
                if hasattr(model, "predict_proba"):
                    oof_proba.append(model.predict_proba(ds.X.iloc[te]))

    folds = pd.DataFrame(rows)
    metrics = ["balanced_accuracy", "macro_f1", "mcc", "kappa"]
    summary = {}
    for m in metrics:
        v = folds[m].values
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 3 else (np.nan, np.nan)
        summary[m] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "lo95": float(lo), "hi95": float(hi)}

    labels = sorted(set(oof_true))
    cm = confusion_matrix(oof_true, oof_pred, labels=labels)
    chance = float(pd.Series(ds.y).value_counts(normalize=True).max())

    text = [ds.describe(), ""]
    for m in metrics:
        s = summary[m]
        text.append(f"{m:<20} {s['mean']:.3f} +/- {s['sd']:.3f} "
                    f"[95% {s['lo95']:.3f}-{s['hi95']:.3f}]")
    text.append(f"{'majority-class rate':<20} {chance:.3f}  "
                f"(balanced accuracy at chance = {1 / len(labels):.3f})")
    if summary["balanced_accuracy"]["mean"] < 1.0 / len(labels) + 0.05:
        text.append("! Balanced accuracy is at or near chance - the features do not "
                    "separate these classes.")

    return {
        "folds": folds, "summary": "\n".join(text), "metrics": summary,
        "confusion": cm, "labels": labels,
        "oof": pd.DataFrame({"index": oof_idx, "true": oof_true, "pred": oof_pred}),
        "oof_proba": np.vstack(oof_proba) if oof_proba else None,
        "report": classification_report(oof_true, oof_pred, zero_division=0),
        "n_groups": ds.n_groups,
    }


def per_class_stats(ds: Dataset, kind: str = "auto", n_splits: int = 5,
                    n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Per-class precision / recall / F1 / specificity / AUC, with intervals.

    A single classification_report is one draw from a noisy process; at ~100
    examples per class the fold-to-fold swing is large enough to change which
    classes look usable. So every metric here is recomputed on each fold of a
    repeated case-grouped CV and reported as mean with a 95% percentile interval.

    Columns:
      support        labelled examples of this class
      prevalence     share of the labelled set
      precision      of plaques predicted as this class, the fraction correct
      recall         of true plaques of this class, the fraction found
      f1             harmonic mean of the two
      specificity    of plaques NOT this class, the fraction correctly excluded
      auc            one-vs-rest ROC AUC; threshold-free, so it separates
                     "the model cannot rank this class" from "the decision
                     threshold is badly placed for it"
    """
    from sklearn.metrics import roc_auc_score

    classes = sorted(ds.y.unique())
    per_fold = []

    for rep in range(n_repeats):
        cv, groups = _splitter(ds, n_splits, seed + rep)
        for tr, te in cv.split(ds.X, ds.y, groups):
            model = build_model(kind, seed=seed + rep)
            model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
            pred = model.predict(ds.X.iloc[te])
            truth = ds.y.iloc[te].values
            proba = (model.predict_proba(ds.X.iloc[te])
                     if hasattr(model, "predict_proba") else None)
            model_classes = list(getattr(model, "classes_", classes))

            for cls in classes:
                t = (truth == cls)
                p = (pred == cls)
                tp, fp = int((t & p).sum()), int((~t & p).sum())
                fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
                if t.sum() == 0:      # class absent from this test fold
                    continue
                prec = tp / (tp + fp) if (tp + fp) else np.nan
                rec = tp / (tp + fn) if (tp + fn) else np.nan
                spec = tn / (tn + fp) if (tn + fp) else np.nan
                f1 = (2 * prec * rec / (prec + rec)
                      if prec and rec and (prec + rec) else 0.0)
                auc = np.nan
                if proba is not None and cls in model_classes and 0 < t.sum() < len(t):
                    try:
                        auc = roc_auc_score(t, proba[:, model_classes.index(cls)])
                    except ValueError:
                        pass
                per_fold.append({"cls": cls, "precision": prec, "recall": rec,
                                 "f1": f1, "specificity": spec, "auc": auc})

    folds = pd.DataFrame(per_fold)
    rows = []
    for cls in classes:
        sub = folds[folds["cls"] == cls]
        n = int((ds.y == cls).sum())
        row = {"class": cls, "support": n, "prevalence": n / len(ds.y)}
        for m in ("precision", "recall", "f1", "specificity", "auc"):
            v = sub[m].dropna().values
            row[m] = float(np.mean(v)) if len(v) else np.nan
            row[f"{m}_lo"] = float(np.percentile(v, 2.5)) if len(v) > 3 else np.nan
            row[f"{m}_hi"] = float(np.percentile(v, 97.5)) if len(v) > 3 else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("support", ascending=False)
    return out.reset_index(drop=True)


def format_class_stats(stats: pd.DataFrame) -> str:
    """Readable rendering of per_class_stats, metric with its interval."""
    W = 19
    lines = [f"{'class':<22}{'n':>5}  " + "".join(
        h.rjust(W) for h in ("precision", "recall", "F1", "AUC (1-vs-rest)"))]
    for _, r in stats.iterrows():
        def cell(m):
            if not np.isfinite(r[m]):
                return "-".rjust(W)
            return f"{r[m]:.2f} [{r[f'{m}_lo']:.2f}-{r[f'{m}_hi']:.2f}]".rjust(W)
        lines.append(f"{r['class']:<22}{int(r['support']):>5}  "
                     + cell("precision") + cell("recall") + cell("f1") + cell("auc"))
    return "\n".join(lines)


def plot_class_stats(stats: pd.DataFrame, ax=None) -> Figure:
    """Per-class recall and precision with intervals, ordered by support."""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(7.6, 0.62 * len(stats) + 2.0)) if owned
                   else (ax.figure, ax))
        y = np.arange(len(stats))[::-1]
        for off, metric, colour in ((0.19, "recall", PALETTE[0]),
                                    (-0.19, "precision", PALETTE[1])):
            v = stats[metric].values
            err = np.abs(np.vstack([v - stats[f"{metric}_lo"].values,
                                    stats[f"{metric}_hi"].values - v]))
            err = np.nan_to_num(err)
            ax.barh(y + off, v, height=0.34, color=colour, label=metric, zorder=3)
            ax.errorbar(v, y + off, xerr=err, fmt="none", ecolor="#374151",
                        elinewidth=1.0, capsize=2.5, zorder=4)
        ax.set_yticks(y, [f"{r['class']}\nn={r['support']}"
                          for _, r in stats.iterrows()], fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score (95% interval over repeated case-grouped CV)")
        ax.set_title("Per-class performance")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


def compare_models(ds: Dataset, kinds: list | None = None, n_splits: int = 5,
                   n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Run every listed model through the identical case-grouped CV and rank them.

    Always includes "dummy" so the table shows what beating nothing looks like.
    Sorted by balanced accuracy; read the CI, not the point estimate - at a few
    hundred labels the intervals usually overlap and "the best model" is noise.
    """
    kinds = list(kinds or [k for k in MODELS if k != "dummy"])
    if "dummy" not in kinds:
        kinds.append("dummy")

    rows = []
    for kind in kinds:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = evaluate(ds, kind=kind, n_splits=n_splits,
                               n_repeats=n_repeats, seed=seed)
            m = res["metrics"]
            rows.append({
                "model": kind,
                "description": MODELS[kind]["description"],
                "balanced_accuracy": m["balanced_accuracy"]["mean"],
                "sd": m["balanced_accuracy"]["sd"],
                "lo95": m["balanced_accuracy"]["lo95"],
                "hi95": m["balanced_accuracy"]["hi95"],
                "macro_f1": m["macro_f1"]["mean"],
                "mcc": m["mcc"]["mean"],
            })
        except Exception as exc:
            warnings.warn(f"Model {kind!r} failed: {type(exc).__name__}: {exc}")
            rows.append({"model": kind, "description": MODELS[kind]["description"],
                         "balanced_accuracy": np.nan, "sd": np.nan, "lo95": np.nan,
                         "hi95": np.nan, "macro_f1": np.nan, "mcc": np.nan})

    out = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    return out.reset_index(drop=True)


def plot_model_comparison(comp: pd.DataFrame, ax=None) -> Figure:
    """Model ranking with CV spread, dummy baseline marked."""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(7.4, 0.42 * len(comp) + 1.8)) if owned
                   else (ax.figure, ax))
        sub = comp.dropna(subset=["balanced_accuracy"]).iloc[::-1]
        pos = np.arange(len(sub))
        colours = [PALETTE[9] if m == "dummy" else PALETTE[0] for m in sub["model"]]
        err = np.abs(np.vstack([sub["balanced_accuracy"] - sub["lo95"],
                                sub["hi95"] - sub["balanced_accuracy"]]))
        ax.barh(pos, sub["balanced_accuracy"], color=colours, height=0.66, zorder=3)
        ax.errorbar(sub["balanced_accuracy"], pos, xerr=err, fmt="none",
                    ecolor="#374151", elinewidth=1.1, capsize=3, zorder=4)
        ax.set_yticks(pos, sub["model"])

        dummy = comp.loc[comp["model"] == "dummy", "balanced_accuracy"]
        if len(dummy) and np.isfinite(dummy.iloc[0]):
            ax.axvline(dummy.iloc[0], color=PALETTE[2], ls="--", lw=1.4, zorder=5,
                       label="majority-class baseline")
            ax.legend(loc="lower right")

        ax.set_xlabel("Balanced accuracy (case-grouped CV, 95% interval)")
        ax.set_title("Model comparison")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


def permutation_baseline(ds: Dataset, kind: str = "auto", n_permutations: int = 10,
                         n_splits: int = 5, seed: int = 0) -> dict:
    """
    Refit on shuffled labels to build the null distribution of balanced accuracy.

    This is the check that catches an impressive-looking score that is really just
    class imbalance. p is the fraction of shuffled runs that match or beat the
    real score.
    """
    real = evaluate(ds, kind, n_splits, n_repeats=1, seed=seed)
    observed = real["metrics"]["balanced_accuracy"]["mean"]

    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_permutations):
        shuffled = Dataset(X=ds.X, y=pd.Series(rng.permutation(ds.y.values)),
                           groups=ds.groups, features=ds.features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null.append(evaluate(shuffled, kind, n_splits, n_repeats=1,
                                 seed=seed + 100 + i)["metrics"]["balanced_accuracy"]["mean"])
    null = np.array(null)
    beaten = int((null >= observed).sum())
    p = float(beaten + 1) / (len(null) + 1)
    # A permutation p can never go below 1/(n+1). With 10 shuffles the floor is
    # 0.091, so "p > 0.05" would be guaranteed regardless of how good the model is.
    p_floor = 1.0 / (len(null) + 1)
    sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    z = (observed - null.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")

    if beaten == 0 and p_floor > 0.05:
        verdict = (f"  - no shuffle reached it, but {len(null)} permutations can only "
                   f"resolve p >= {p_floor:.3f}. Raise n_permutations to at least 20 "
                   f"for a p below 0.05.")
    elif p > 0.05:
        verdict = "  - indistinguishable from chance."
    else:
        verdict = "  - above chance."

    return {"observed": observed, "null": null, "p": p, "p_floor": p_floor,
            "z": z, "n_beaten": beaten,
            "message": (f"balanced accuracy {observed:.3f} vs shuffled-label null "
                        f"{null.mean():.3f} +/- {sd:.3f} (z = {z:.1f}), "
                        f"p = {p:.3f}{verdict}")}


def learning_curve(ds: Dataset, fractions=(0.3, 0.5, 0.7, 0.85, 1.0),
                   kind: str = "auto", n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Score against training-set size. The slope at the right-hand edge tells you
    whether the next batch of labels will buy anything.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        for rep in range(n_repeats):
            n = max(int(len(ds.X) * frac), ds.y.nunique() * 4)
            if n >= len(ds.X):
                idx = np.arange(len(ds.X))
            else:  # stratified subsample so rare classes survive
                idx = (pd.Series(range(len(ds.y))).groupby(ds.y.values, group_keys=False)
                       .apply(lambda s: s.sample(max(2, int(round(len(s) * frac))),
                                                 random_state=int(rng.integers(1e6))))
                       .values)
            sub = Dataset(X=ds.X.iloc[idx].reset_index(drop=True),
                          y=ds.y.iloc[idx].reset_index(drop=True),
                          groups=ds.groups.iloc[idx].reset_index(drop=True),
                          features=ds.features)
            if sub.y.nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = evaluate(sub, kind, n_splits=3, n_repeats=1, seed=seed + rep)
            rows.append({"fraction": frac, "n": len(sub.X), "repeat": rep,
                         "balanced_accuracy": res["metrics"]["balanced_accuracy"]["mean"]})
    return pd.DataFrame(rows)


def feature_importance(ds: Dataset, kind: str = "auto", n_repeats: int = 5,
                       seed: int = 0) -> pd.DataFrame:
    """
    Permutation importance on held-out folds. Preferred over tree gain, which is
    biased when features are correlated - and these features are very correlated
    (see plaque_plots.plot_correlation).
    """
    cv, groups = _splitter(ds, 5, seed)
    acc = np.zeros(len(ds.features))
    n_folds = 0
    for tr, te in cv.split(ds.X, ds.y, groups):
        model = build_model(kind, seed=seed)
        model.fit(ds.X.iloc[tr], ds.y.iloc[tr])
        r = permutation_importance(model, ds.X.iloc[te], ds.y.iloc[te],
                                   n_repeats=n_repeats, random_state=seed,
                                   scoring="balanced_accuracy")
        acc += r.importances_mean
        n_folds += 1
    return (pd.DataFrame({"feature": ds.features, "importance": acc / max(1, n_folds)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# -------------------------
# Fit and apply
# -------------------------

def fit_final(ds: Dataset, kind: str = "auto", seed: int = 0, calibrate: bool = False):
    """Fit on everything. Only do this after evaluate() has told you it is worth it."""
    model = build_model(kind, seed=seed, calibrate=calibrate)
    model.fit(ds.X, ds.y)
    # Pipeline.feature_names_in_ is a read-only property, so record the training
    # column order under our own name for predict_unlabelled/suggest_to_label.
    model.plaque_features_ = list(ds.features)
    return model


def _model_features(model, df: pd.DataFrame, features: list | None) -> list:
    """Recover the training feature order, preferring what the model recorded."""
    feats = (list(features) if features is not None
             else list(getattr(model, "plaque_features_", None)
                       or getattr(model, "feature_names_in_", FEATURES)))
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Input is missing feature column(s) the model was trained "
                       f"on: {missing}")
    return feats


def predict_unlabelled(model, df: pd.DataFrame, features: list | None = None,
                       abstain_below: float = 0.60) -> pd.DataFrame:
    """
    Apply to unlabelled plaques. Predictions whose top probability falls below
    `abstain_below` are marked `uncertain` rather than forced into a class - at a
    few hundred training labels, a confident-looking wrong answer is the failure
    mode to avoid.
    """
    feats = _model_features(model, df, features)
    proba = model.predict_proba(df[feats])
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    top = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]

    out = pd.DataFrame({
        "plaque_id": df["plaque_id"].values if "plaque_id" in df else np.arange(len(df)),
        "prediction": np.where(top >= abstain_below, pred, "uncertain"),
        "top_class": pred,
        "confidence": top,
        "entropy": -(proba * np.log(proba + 1e-12)).sum(axis=1),
    })
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]
    return out


def suggest_to_label(model, df: pd.DataFrame, n: int = 100,
                     strategy: str = "margin", features: list | None = None,
                     seed: int = 0) -> pd.DataFrame:
    """
    Rank unlabelled plaques for the next annotation batch. With a small budget,
    which plaques you label matters more than how many.

    strategy : "margin"  smallest gap between the top two classes (usually best)
               "entropy" most uncertain overall
               "random"  control - use it to check active learning is helping
    """
    feats = _model_features(model, df, features)
    proba = model.predict_proba(df[feats])

    if strategy == "random":
        score = np.random.default_rng(seed).random(len(df))
    elif strategy == "entropy":
        score = -(proba * np.log(proba + 1e-12)).sum(axis=1)
    elif strategy == "margin":
        part = np.sort(proba, axis=1)
        score = -(part[:, -1] - part[:, -2])  # smaller margin = more informative
    else:
        raise ValueError(f"Unknown strategy={strategy!r}")

    out = df.copy()
    out["_score"] = score
    cols = [c for c in ["plaque_id", "wsi_name", "x1", "y1", "area_um2",
                        "circularity", "mean_optical_density", "crop_path"]
            if c in out.columns]
    return (out.sort_values("_score", ascending=False).head(n)[cols + ["_score"]]
            .rename(columns={"_score": "uncertainty"}).reset_index(drop=True))


# -------------------------
# Plots
# -------------------------

def plot_confusion(report: dict, normalise: bool = True, ax=None) -> Figure:
    """Out-of-fold confusion matrix. Read the rows: where does each true class go?"""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(5.6, 4.8)) if owned else (ax.figure, ax))
        cm = report["confusion"].astype(float)
        labels = report["labels"]
        shown = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)) if normalise else cm

        im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() or 1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                txt = f"{shown[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="white" if shown[i, j] > shown.max() * 0.6 else "#374151")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Out-of-fold confusion" + (" (row-normalised)" if normalise else ""))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02).outline.set_visible(False)
        if owned:
            fig.tight_layout()
    return fig


def plot_permutation_test(perm: dict, ax=None) -> Figure:
    """Real score against the shuffled-label null."""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(6.2, 3.6)) if owned else (ax.figure, ax))
        ax.hist(perm["null"], bins=15, color=PALETTE[9], alpha=0.75,
                label="shuffled labels", zorder=3)
        ax.axvline(perm["observed"], color=PALETTE[2], lw=2.2, zorder=4,
                   label=f"observed {perm['observed']:.3f}")
        ax.set_xlabel("Balanced accuracy")
        ax.set_ylabel("Permutations")
        title = f"Permutation test (p = {perm['p']:.3f}"
        if perm.get("p_floor", 0) > 0.05:
            title += f", floor {perm['p_floor']:.3f}"
        ax.set_title(title + f", z = {perm.get('z', float('nan')):.1f})")
        ax.legend()
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


def plot_learning_curve(curve: pd.DataFrame, ax=None) -> Figure:
    """Balanced accuracy against training-set size, with spread across repeats."""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(6.2, 4.0)) if owned else (ax.figure, ax))
        g = curve.groupby("n")["balanced_accuracy"]
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        ax.fill_between(m.index, lo, hi, color=PALETTE[0], alpha=0.18, zorder=2)
        ax.plot(m.index, m.values, "o-", color=PALETTE[0], lw=1.8, ms=5, zorder=3)
        ax.set_xlabel("Labelled plaques used for training")
        ax.set_ylabel("Balanced accuracy (CV)")
        ax.set_title("Learning curve - is it worth labelling more?")
        ax.set_axisbelow(True)

        if len(m) >= 3:  # slope over the last two points, per 100 extra labels
            dn = m.index[-1] - m.index[-2]
            slope = (m.values[-1] - m.values[-2]) / dn * 100 if dn else 0.0
            verdict = ("still climbing - more labels should help" if slope > 0.02
                       else "flattening - more of the same labels will not help much")
            ax.annotate(f"{slope:+.3f} balanced accuracy per 100 labels\n{verdict}",
                        (0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=8, color="#6b7280",
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e3e6ea"))
        if owned:
            fig.tight_layout()
    return fig


def plot_importance(imp: pd.DataFrame, top: int = 20, ax=None) -> Figure:
    """Permutation importance, largest first."""
    with mpl.rc_context(_RC):
        owned = ax is None
        sub = imp.head(top).iloc[::-1]
        fig, ax = (plt.subplots(figsize=(6.4, 0.29 * len(sub) + 1.5)) if owned
                   else (ax.figure, ax))
        colours = [PALETTE[2] if v <= 0 else PALETTE[0] for v in sub["importance"]]
        ax.barh(range(len(sub)), sub["importance"], color=colours, height=0.7, zorder=3)
        ax.set_yticks(range(len(sub)), sub["feature"], fontsize=8)
        ax.axvline(0, color="#9498a0", lw=0.9)
        ax.set_xlabel("Drop in balanced accuracy when shuffled")
        ax.set_title("Permutation feature importance")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


def plot_confidence(pred: pd.DataFrame, abstain_below: float = 0.60, ax=None) -> Figure:
    """Confidence distribution over unlabelled plaques, with the abstain cut marked."""
    with mpl.rc_context(_RC):
        owned = ax is None
        fig, ax = (plt.subplots(figsize=(6.2, 3.6)) if owned else (ax.figure, ax))
        ax.hist(pred["confidence"], bins=40, color=PALETTE[0], alpha=0.8, zorder=3)
        ax.axvline(abstain_below, color=PALETTE[2], ls="--", lw=1.6, zorder=4)
        frac = 100.0 * (pred["confidence"] < abstain_below).mean()
        ax.annotate(f"{frac:.1f}% below the abstain threshold",
                    (abstain_below, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color="#7c4a03")
        ax.set_xlabel("Top-class probability")
        ax.set_ylabel("Plaques")
        ax.set_title("Prediction confidence on unlabelled data")
        ax.set_axisbelow(True)
        if owned:
            fig.tight_layout()
    return fig


def report_figure(ds: Dataset, report: dict, perm: dict | None = None,
                  curve: pd.DataFrame | None = None,
                  imp: pd.DataFrame | None = None) -> Figure:
    """Single-page summary of everything evaluate() produced."""
    with mpl.rc_context(_RC):
        fig = plt.figure(figsize=(14.5, 8.6))
        gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.32)
        plot_confusion(report, ax=fig.add_subplot(gs[0, 0]))
        if perm:
            plot_permutation_test(perm, ax=fig.add_subplot(gs[0, 1]))
        if curve is not None and len(curve):
            plot_learning_curve(curve, ax=fig.add_subplot(gs[0, 2]))
        if imp is not None and len(imp):
            plot_importance(imp, top=14, ax=fig.add_subplot(gs[1, 0:2]))

        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        ax.text(0, 1, report["summary"], va="top", ha="left", family="monospace",
                fontsize=7.6, color="#1a1d23")
        warn = ("CV is not case-grouped - scores are optimistic."
                if report["n_groups"] < 2 else "")
        if warn:
            ax.text(0, 0.02, warn, va="bottom", fontsize=8, color="#7c4a03",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fef6e7", ec="#f5d9a8"))
        fig.suptitle(f"Plaque classifier report - {len(ds.X):,} labels, "
                     f"{len(ds.classes)} classes", fontsize=12.5, weight="600")
    return fig


# -------------------------
# Synthetic labels, for testing the pipeline before real ones exist
# -------------------------

def synthetic_labels(df: pd.DataFrame, seed: int = 0, signal: float = 1.0) -> pd.Series:
    """
    Plausible fake labels derived from morphology, so the whole pipeline can be
    exercised end to end before any annotation exists.

    signal=1 gives learnable structure; signal=0 gives labels independent of the
    features, which MUST score at chance - that is the pipeline's own null test.
    NEVER use these for a real result.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if signal <= 0:
        return pd.Series(rng.choice(CLASSES[:4], n), index=df.index)

    area = df.get("area_um2", pd.Series(np.full(n, 100.0))).fillna(100.0).values
    circ = df.get("circularity", pd.Series(np.full(n, 0.5))).fillna(0.5).values
    ar = df.get("aspect_ratio", pd.Series(np.full(n, 1.3))).fillna(1.3).values
    od = df.get("mean_optical_density", pd.Series(np.full(n, 0.33))).fillna(0.33).values

    out = np.full(n, "diffuse", dtype=object)
    out[(ar > 2.2)] = "CAA"                                     # elongated, vessel-like
    out[(circ > 0.55) & (od > 0.34) & (area < 400)] = "cored"    # small, round, dark
    out[(circ > 0.62) & (od > 0.38)] = "core_halo"               # rounder and darker still
    out[(area > 600) & (circ < 0.45)] = "diffuse_core_partial"   # big and irregular
    out[(area > 250) & (circ < 0.30) & (od < 0.31)] = "nucleus_centre"
    out[(area < 25)] = "artefact"                                # too small to be real

    flip = rng.random(n) > signal                                # label noise
    out[flip] = rng.choice(CLASSES, flip.sum())
    return pd.Series(out, index=df.index)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Evaluate a plaque-type classifier. With --demo it invents "
                    "labels so you can see the whole pipeline run before annotating.")
    ap.add_argument("features",
                    default="wsi_plaque_boxes_variable/Case_6_1_ABeta_plaque_boxes_variable.csv",
                    nargs="?")
    ap.add_argument("--labels", default=None, help="labels.csv with plaque_id,label")
    ap.add_argument("--demo", action="store_true", help="use synthetic labels")
    ap.add_argument("--demo-n", type=int, default=400,
                    help="how many synthetic labels (default 400, a realistic budget)")
    ap.add_argument("--drop-black", action="store_true",
                    help="exclude the black-core features")
    ap.add_argument("--model", default="auto",
                    help="classifier to use; see --list-models")
    ap.add_argument("--compare", action="store_true",
                    help="run every registered model through the same CV and rank them")
    ap.add_argument("--list-models", action="store_true",
                    help="print the classifier registry and exit")
    ap.add_argument("--permutations", type=int, default=10,
                    help="shuffled-label refits for the null distribution")
    ap.add_argument("--outdir", default="plaque_classifier_out")
    args = ap.parse_args()

    if args.list_models:
        print(available_models().to_string(index=False))
        raise SystemExit(0)

    if args.demo:
        feats = load_features_only(args.features)
        sample = feats.sample(min(args.demo_n, len(feats)), random_state=0).copy()
        sample[LABEL_COL] = synthetic_labels(sample, signal=0.85)
        data = sample
        print(f"DEMO MODE - {len(data)} SYNTHETIC labels. Results are meaningless "
              f"except as a pipeline check.\n")
    elif args.labels:
        data = load_labelled(args.features, args.labels)
    else:
        ap.error("provide --labels, or --demo to run on synthetic labels")

    ds = make_dataset(data, drop_black=args.drop_black)
    print(ds.describe(), "\n")

    if args.compare:
        comp = compare_models(ds)
        print(comp.to_string(index=False), "\n")

    rep = evaluate(ds, kind=args.model)
    print(rep["summary"], "\n")
    print(rep["report"])

    perm = permutation_baseline(ds, kind=args.model, n_permutations=args.permutations)
    print(perm["message"], "\n")

    curve = learning_curve(ds, kind=args.model)
    imp = feature_importance(ds, kind=args.model)
    print(imp.head(10).to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    fig = report_figure(ds, rep, perm, curve, imp)
    path = os.path.join(args.outdir, "classifier_report.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nWrote {os.path.abspath(path)}")
