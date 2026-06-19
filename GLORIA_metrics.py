"""

This file is meant to be run next to GLORIA_app.py. It does not replace the GUI;
it adds metrics:
- explicit hyperparameter reporting
- repeated stratified CV with uncertainty
- class-level precision/recall/F1
- baseline comparisons
- SMOTE threshold sensitivity
- Century × Region association (Cramer's V)
- optional leave-one-site-out validation
- optional method-bias summaries

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, binomtest, kruskal

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from collections import defaultdict
from sklearn.inspection import permutation_importance

# =============================================================================

# =============================================================================
DATABASE_PATH = "Database_windows_only.xlsx"  
SHEET_NAME = "clean"       
TARGET = "Global region"         
OUTPUT_DIR = ""
FEATURES = None
N_ITER = 50
N_SPLITS = 5
N_REPEATS = 20
SMOTE_THRESHOLD = 10
RUN_SMOTE_SENSITIVITY = True
ASSOCIATION_COLS = ["Century", "Global region"]
GROUP_COL = None      
METHOD_COL = None     

RANDOM_STATE = 42

EXCLUDE_METADATA = [
    "ID", "Reference", "ID(Ref)", "Site", "City", "Country", "Region", "Global region",
    "Form", "Data Method?", "Colour", "Mg/Ca", "Mg", "Date - Early", "Date - Mean",
    "Date - Late", "Date", "(Na2O + MgO)/Sommes des autres",
]

RF_PARAM_DIST = {
    "clf__n_estimators": np.arange(300, 801, 100),
    "clf__max_depth": [None, 10, 20, 30, 40, 50],
    "clf__min_samples_split": [2, 3, 4, 5, 8, 10],
    "clf__min_samples_leaf": [1, 2, 3, 4],
    "clf__max_features": ["sqrt", "log2", None],
    "clf__bootstrap": [True, False],
}

DEFAULT_FEATURES = [
    "SiO2", "K2O", "CaO", "Na2O", "MgO", "Al2O3", "Fe2O3", "P2O5", "MnO",
    "Rb2O", "SrO", "TiO2", "ZnO", "ZrO2",
]

REPLACE_MAP = {
    "": 0, "REF": 0, "#VALUE!": 0, "-": 0, "<LOD": 0, "< LOD": 0,
    "BD": 0, "ND": 0, np.nan: 0,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def choose_features(df: pd.DataFrame, target: str, requested: Optional[list[str]] = None) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns are missing: {missing}")
        return requested
    preferred = [c for c in DEFAULT_FEATURES if c in df.columns]
    if preferred:
        return preferred
    excluded = set(EXCLUDE_METADATA + [target])
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]


def clean_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return (
        df[features]
        .replace(REPLACE_MAP)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def clean_y(y: pd.Series) -> pd.Series:
    y = y.astype(str).str.strip()
    return y[(y != "") & (y.str.lower() != "nan")]



def make_sampling_strategy_callable(min_class_for_smote: int | None):

    if min_class_for_smote is None:
        return None

    def strategy(y):
        y = pd.Series(y)
        counts = y.value_counts()
        majority = counts.max()

        sampling_strategy = {
            cls: majority
            for cls, n in counts.items()
            if n >= min_class_for_smote and n < majority
        }

        return sampling_strategy

    return strategy


def make_rf_pipeline(
    y: pd.Series,
    min_class_for_smote: int | None,
    random_state: int = RANDOM_STATE
):
    steps = []

    
    steps.append(("scaler", StandardScaler()))

    strategy = make_sampling_strategy_callable(min_class_for_smote)

    if strategy is not None:
        
        steps.append((
            "smote",
            SMOTE(
                sampling_strategy=strategy,
                k_neighbors=5,
                random_state=random_state
            )
        ))

    steps.append((
        "clf",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    ))

    return ImbPipeline(steps)


def make_baselines(random_state: int = RANDOM_STATE) -> dict:
    return {
        "dummy_most_frequent": SkPipeline([("scaler", StandardScaler()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "nearest_centroid": SkPipeline([("scaler", StandardScaler()), ("clf", NearestCentroid())]),
        "lda": SkPipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "multinomial_logistic": SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
        ]),
    }


def metric_summary(scores: list[float] | np.ndarray) -> dict:

    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n": 0}

    mean = float(np.mean(x))
    if len(x) == 1:
        return {"mean": mean, "sd": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}

    sd = float(np.std(x, ddof=1))
    half_width = float(1.96 * sd / np.sqrt(len(x)))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": int(len(x)),
    }



def make_json_safe(obj):
    """Convert numpy/pandas objects to standard Python objects before json.dump."""
    if isinstance(obj, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    if pd.isna(obj) and not isinstance(obj, str):
        return None
    return obj

def century_within_one(y_true: Iterable, y_pred: Iterable) -> float:
    true = pd.to_numeric(pd.Series(y_true).astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    pred = pd.to_numeric(pd.Series(y_pred).astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    ok = (true - pred).abs() <= 1
    return float(ok.mean())


def repeated_cv_predictions(estimator, X, y, n_splits=5, n_repeats=20, random_state=RANDOM_STATE):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    rows = []
    split_rows = []
    all_true, all_pred, all_fold = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        yt = y.iloc[test_idx].to_numpy()

        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(yt, pred),
            "balanced_accuracy": balanced_accuracy_score(yt, pred),
            "f1_macro": f1_score(yt, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(yt, pred, average="weighted", zero_division=0),
            "within_one_century": century_within_one(yt, pred)
            if pd.Series(y).str.contains(r"\d").any() else np.nan,
        })

        for idx in train_idx:
            split_rows.append({
                "fold": fold,
                "sample_index": X.index[idx],
                "split": "train",
                "label": y.iloc[idx],
            })

        for idx in test_idx:
            split_rows.append({
                "fold": fold,
                "sample_index": X.index[idx],
                "split": "validation",
                "label": y.iloc[idx],
            })

        all_true.extend(yt)
        all_pred.extend(pred)
        all_fold.extend([fold] * len(test_idx))

    pred_df = pd.DataFrame({
        "fold": all_fold,
        "true": all_true,
        "pred": all_pred
    })

    fold_metrics = pd.DataFrame(rows)
    split_df = pd.DataFrame(split_rows)

    return pred_df, fold_metrics, split_df


def export_rf_interpretability(
    fitted_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    target: str,
    outdir: Path,
    scoring: str = "f1_weighted",
    n_repeats: int = 30,
    random_state: int = RANDOM_STATE
):


    outdir.mkdir(parents=True, exist_ok=True)

    rf = fitted_pipeline.named_steps["clf"]
    scaler = fitted_pipeline.named_steps["scaler"]

    impurity_df = pd.DataFrame({
        "feature": features,
        "rf_impurity_importance": rf.feature_importances_
    })

    perm = permutation_importance(
        fitted_pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    permutation_df = pd.DataFrame({
        "feature": features,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std
    })

    split_counts = defaultdict(int)
    split_depths = defaultdict(list)
    split_thresholds_scaled = defaultdict(list)

    def walk_tree(tree, node_id=0, depth=0):
        feature_id = tree.feature[node_id]

        # -2 means leaf node
        if feature_id >= 0:
            feature_name = features[feature_id]

            split_counts[feature_name] += 1
            split_depths[feature_name].append(depth)
            split_thresholds_scaled[feature_name].append(tree.threshold[node_id])

            walk_tree(tree, tree.children_left[node_id], depth + 1)
            walk_tree(tree, tree.children_right[node_id], depth + 1)

    for estimator in rf.estimators_:
        walk_tree(estimator.tree_)

    split_df = pd.DataFrame({
        "feature": features,
        "split_count": [split_counts[f] for f in features],
        "mean_split_depth": [
            np.mean(split_depths[f]) if len(split_depths[f]) > 0 else np.nan
            for f in features
        ],
        "median_split_depth": [
            np.median(split_depths[f]) if len(split_depths[f]) > 0 else np.nan
            for f in features
        ],
    })

    threshold_rows = []

    for i, feature in enumerate(features):
        thresholds_scaled = split_thresholds_scaled[feature]

        if len(thresholds_scaled) == 0:
            continue

        mean_i = scaler.mean_[i]
        scale_i = scaler.scale_[i]

        thresholds_original = [
            t * scale_i + mean_i
            for t in thresholds_scaled
        ]

        threshold_rows.append({
            "feature": feature,
            "n_thresholds": len(thresholds_original),
            "threshold_min": float(np.min(thresholds_original)),
            "threshold_q25": float(np.percentile(thresholds_original, 25)),
            "threshold_median": float(np.median(thresholds_original)),
            "threshold_q75": float(np.percentile(thresholds_original, 75)),
            "threshold_max": float(np.max(thresholds_original)),
        })

    thresholds_df = pd.DataFrame(threshold_rows)

    summary_df = (
        impurity_df
        .merge(permutation_df, on="feature", how="left")
        .merge(split_df, on="feature", how="left")
    )

    summary_df = summary_df.sort_values(
        "permutation_importance_mean",
        ascending=False
    )

    thresholds_df = thresholds_df.sort_values(
        "n_thresholds",
        ascending=False
    )

    safe_target = target.replace(" ", "_").replace("/", "_")

    summary_df.to_csv(
        outdir / f"{safe_target}_variable_importance.csv",
        index=False
    )

    thresholds_df.to_csv(
        outdir / f"{safe_target}_split_thresholds.csv",
        index=False
    )

    return summary_df, thresholds_df

def evaluate_target(df: pd.DataFrame, target: str, features: list[str], outdir: Path,
                    n_iter: int = 50, n_splits: int = 5, n_repeats: int = 20,
                    smote_threshold: int | None = 10) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    valid = clean_y(df[target]).index
    d = df.loc[valid].copy()
    X = clean_X(d, features)
    y = d[target].astype(str).str.strip()

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    base_rf = make_rf_pipeline(y, smote_threshold)
    search = RandomizedSearchCV(
        base_rf,
        RF_PARAM_DIST,
        n_iter=n_iter,
        cv=inner_cv,
        scoring="f1_weighted",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    best_rf = search.best_estimator_
    importance_df, thresholds_df = export_rf_interpretability(
    fitted_pipeline=best_rf,
    X=X,
    y=y,
    features=features,
    target=target,
    outdir=outdir,
    scoring="f1_weighted",
    n_repeats=30,
    random_state=RANDOM_STATE
    )

    pred_df, fold_metrics, split_df = repeated_cv_predictions(best_rf, X, y, n_splits=n_splits, n_repeats=n_repeats)
    pred_df.to_csv(outdir / f"{target}_cv_predictions.csv", index=False)
    fold_metrics.to_csv(outdir / f"{target}_fold_metrics.csv", index=False)
    split_df.to_csv(outdir / f"{target}_cv_split_assignments.csv", index=False)

    report = pd.DataFrame(classification_report(pred_df["true"], pred_df["pred"], output_dict=True, zero_division=0)).T
    report.to_csv(outdir / f"{target}_class_report.csv")

    labels = sorted(y.unique())
    cm = confusion_matrix(pred_df["true"], pred_df["pred"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(outdir / f"{target}_confusion_counts.csv")
    cm_norm = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0)
    cm_norm.to_csv(outdir / f"{target}_confusion_recall_normalized.csv")

    baseline_rows = []
    metrics_to_summarise = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    if target.lower() == "century":
        metrics_to_summarise.append("within_one_century")

    for name, model in {"random_forest": best_rf, **make_baselines()}.items():
        p, m, _ = repeated_cv_predictions(model, X, y, n_splits=n_splits, n_repeats=n_repeats)
        row = {"model": name}
        for metric in metrics_to_summarise:
            row.update({f"{metric}_{k}": v for k, v in metric_summary(m[metric]).items()})
        baseline_rows.append(row)
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(outdir / f"{target}_baseline_comparison.csv", index=False)

    summary = {
        "target": target,
        "features": features,
        "n_samples": int(len(y)),
        "class_counts": y.value_counts().to_dict(),
        "cv": {"outer_n_splits": n_splits, "outer_n_repeats": n_repeats, "inner_n_splits": n_splits},
        "smote_threshold": smote_threshold,
        "randomized_search_n_iter": n_iter,
        "rf_search_space": {k: list(map(str, v)) for k, v in RF_PARAM_DIST.items()},
        "best_params": search.best_params_,
        "best_inner_cv_f1_weighted": float(search.best_score_),
        "performance": {
            metric: metric_summary(fold_metrics[metric])
            for metric in fold_metrics.columns
            if metric != "fold" and (metric != "within_one_century" or target.lower() == "century")
        },
    }
    with open(outdir / f"{target}_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=2, ensure_ascii=False)
    return summary


def smote_sensitivity(df: pd.DataFrame, target: str, features: list[str], outdir: Path,
                      thresholds=(None, 10, 20, 30, 50), n_splits=5, n_repeats=10, n_iter=20) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        label = "no_smote" if threshold is None else f"min_{threshold}"
        subout = outdir / f"smote_{label}"
        summary = evaluate_target(df, target, features, subout, n_iter=n_iter, n_splits=n_splits, n_repeats=n_repeats, smote_threshold=threshold)
        row = {"threshold": label, "best_params": json.dumps(make_json_safe(summary["best_params"]))}
        for metric, vals in summary["performance"].items():
            row[f"{metric}_mean"] = vals["mean"]
            row[f"{metric}_ci95_low"] = vals["ci95_low"]
            row[f"{metric}_ci95_high"] = vals["ci95_high"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(outdir / f"{target}_smote_threshold_sensitivity.csv", index=False)
    return result


def cramers_v_table(df: pd.DataFrame, col1="Century", col2="Global region") -> dict:
    tab = pd.crosstab(df[col1].astype(str), df[col2].astype(str))
    chi2, p, dof, expected = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    phi2 = chi2 / n
    # bias-corrected Cramer's V
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min(kcorr - 1, rcorr - 1) > 0 else np.nan
    return {"cramers_v": float(v), "chi2": float(chi2), "p_value": float(p), "dof": int(dof), "n": int(n), "table": tab}


def leave_one_group_out_validation(df: pd.DataFrame, target: str, features: list[str], group_col: str, outdir: Path,
                                   smote_threshold: int | None = 10) -> pd.DataFrame:
    valid = clean_y(df[target]).index
    d = df.loc[valid].dropna(subset=[group_col]).copy()
    X = clean_X(d, features)
    y = d[target].astype(str).str.strip()
    groups = d[group_col].astype(str)
    rows = []
    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        # skip folds where training has classes with fewer than 2 examples
        if y.iloc[train_idx].value_counts().min() < 2:
            continue
        model = make_rf_pipeline(y.iloc[train_idx], smote_threshold)
        model.set_params(clf__n_estimators=500, clf__max_features="sqrt", clf__class_weight="balanced")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        heldout = groups.iloc[test_idx].iloc[0]
        rows.append({
            "held_out_group": heldout,
            "n_test": int(len(test_idx)),
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[test_idx], pred),
            "f1_macro": f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y.iloc[test_idx], pred, average="weighted", zero_division=0),
        })
    result = pd.DataFrame(rows).sort_values("n_test", ascending=False)
    result.to_csv(outdir / f"{target}_leave_one_{group_col}_out.csv", index=False)
    return result


def method_bias_summary(df: pd.DataFrame, features: list[str], method_col: str, outdir: Path) -> pd.DataFrame:
    rows = []
    for feature in features:
        groups = []
        for method, vals in df.groupby(method_col)[feature]:
            v = pd.to_numeric(vals, errors="coerce").dropna()
            if len(v) >= 3:
                groups.append(v)
            rows.append({
                "feature": feature,
                "method": method,
                "n": int(len(v)),
                "median": float(v.median()) if len(v) else np.nan,
                "iqr": float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else np.nan,
            })
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            rows.append({"feature": feature, "method": "Kruskal-Wallis p", "n": np.nan, "median": float(p), "iqr": float(stat)})
    result = pd.DataFrame(rows)
    result.to_csv(outdir / f"method_bias_summary_by_{method_col}.csv", index=False)
    return result


def konigsfelden_binomial_test(n_samples: int, observed_wrong: int, expected_wrong_rate: float) -> dict:

    test = binomtest(observed_wrong, n_samples, expected_wrong_rate, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95)
    return {
        "n_samples": n_samples,
        "observed_wrong": observed_wrong,
        "observed_wrong_rate": observed_wrong / n_samples,
        "expected_wrong_rate": expected_wrong_rate,
        "p_value": test.pvalue,
        "ci95_low": ci.low,
        "ci95_high": ci.high,
    }


def run_analysis(database, sheet, target, features=None, out="gloria_review_outputs",
                 n_iter=50, n_splits=5, n_repeats=20, smote_threshold=10,
                 run_smote_sensitivity=False, association_cols=("Century", "Global region"),
                 group_col=None, method_col=None):
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    df = normalize_columns(pd.read_excel(database, sheet_name=sheet))
    feature_list = choose_features(df, target, features)

    summary = evaluate_target(df, target, feature_list, outdir, n_iter, n_splits, n_repeats, smote_threshold)
    print(json.dumps(make_json_safe(summary), indent=2, ensure_ascii=False))

    if run_smote_sensitivity:
        print(smote_sensitivity(df, target, feature_list, outdir))

    c1, c2 = association_cols
    if c1 in df.columns and c2 in df.columns:
        assoc = cramers_v_table(df, c1, c2)
        assoc["table"].to_csv(outdir / f"association_{c1}_x_{c2}.csv")
        assoc_json = {k: v for k, v in assoc.items() if k != "table"}
        with open(outdir / f"association_{c1}_x_{c2}.json", "w", encoding="utf-8") as f:
            json.dump(make_json_safe(assoc_json), f, indent=2, ensure_ascii=False)
        print("Association:", assoc_json)

    if group_col and group_col in df.columns:
        print(leave_one_group_out_validation(df, target, feature_list, group_col, outdir, smote_threshold).head())

    if method_col and method_col in df.columns:
        print(method_bias_summary(df, feature_list, method_col, outdir).head())


def main():
    if DATABASE_PATH and TARGET:
        return run_analysis(
            database=DATABASE_PATH,
            sheet=SHEET_NAME,
            target=TARGET,
            features=FEATURES,
            out=OUTPUT_DIR,
            n_iter=N_ITER,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            smote_threshold=SMOTE_THRESHOLD,
            run_smote_sensitivity=RUN_SMOTE_SENSITIVITY,
            association_cols=ASSOCIATION_COLS,
            group_col=GROUP_COL,
            method_col=METHOD_COL,
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--out", default="gloria_review_outputs")
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--smote-threshold", type=int, default=10)
    parser.add_argument("--run-smote-sensitivity", action="store_true")
    parser.add_argument("--association-cols", nargs=2, default=["Century", "Global region"])
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--method-col", default=None)
    args = parser.parse_args()

    return run_analysis(
        database=args.database,
        sheet=args.sheet,
        target=args.target,
        features=args.features,
        out=args.out,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        smote_threshold=args.smote_threshold,
        run_smote_sensitivity=args.run_smote_sensitivity,
        association_cols=args.association_cols,
        group_col=args.group_col,
        method_col=args.method_col,
    )


if __name__ == "__main__":
    main()
