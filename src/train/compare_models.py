import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_eval(X, y, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    results = {}

    # Logistic Regression + calibration
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, n_jobs=1)
    try:
        lr_cal = CalibratedClassifierCV(estimator=lr, cv=5, method="sigmoid")
    except TypeError:
        lr_cal = CalibratedClassifierCV(base_estimator=lr, cv=5, method="sigmoid")
    lr_cal.fit(X_train_s, y_train)
    lr_proba = lr_cal.predict_proba(X_test_s)[:, 1]

    # Tree-based baseline (no external deps)
    hgb = HistGradientBoostingClassifier(random_state=42)
    try:
        hgb_cal = CalibratedClassifierCV(estimator=hgb, cv=5, method="sigmoid")
    except TypeError:
        hgb_cal = CalibratedClassifierCV(base_estimator=hgb, cv=5, method="sigmoid")
    hgb_cal.fit(X_train, y_train)
    hgb_proba = hgb_cal.predict_proba(X_test)[:, 1]

    def metrics(y_true, p):
        auroc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else float("nan")
        aupr = average_precision_score(y_true, p)
        brier = brier_score_loss(y_true, p)
        return {"auroc": float(auroc), "aupr": float(aupr), "brier": float(brier), "n_test": int(len(y_true))}

    results["lr"] = metrics(y_test, lr_proba)
    results["tree"] = metrics(y_test, hgb_proba)

    # RandomForest (class-weighted) + calibration
    try:
        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        try:
            rf_cal = CalibratedClassifierCV(estimator=rf, cv=5, method="sigmoid")
        except TypeError:
            rf_cal = CalibratedClassifierCV(base_estimator=rf, cv=5, method="sigmoid")
        rf_cal.fit(X_train, y_train)
        rf_proba = rf_cal.predict_proba(X_test)[:, 1]
        results["rf"] = metrics(y_test, rf_proba)
    except Exception:
        rf_proba = None

    # ExtraTrees (class-weighted) + calibration
    try:
        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        try:
            et_cal = CalibratedClassifierCV(estimator=et, cv=5, method="sigmoid")
        except TypeError:
            et_cal = CalibratedClassifierCV(base_estimator=et, cv=5, method="sigmoid")
        et_cal.fit(X_train, y_train)
        et_proba = et_cal.predict_proba(X_test)[:, 1]
        results["et"] = metrics(y_test, et_proba)
    except Exception:
        et_proba = None

    # Calibration curves
    try:
        import matplotlib.pyplot as plt  # type: ignore

        for name, proba in [("lr", lr_proba), ("tree", hgb_proba)]:
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
            plt.figure(figsize=(4, 4))
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.plot(mean_pred, frac_pos, marker="o", lw=1)
            plt.title(f"Calibration: {model_name} {name}")
            plt.xlabel("Mean predicted")
            plt.ylabel("Fraction positive")
            plt.tight_layout()
            yield name, plt
    except Exception:
        # plotting not available
        return results, None

    yield results, None


def main() -> None:
    p = argparse.ArgumentParser(description="Compare models and plot calibration")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--target", type=str, default="label_long_wait")
    p.add_argument("--bootstrap_n", type=int, default=200)
    args = p.parse_args()

    fp = str(args.features)
    if fp.endswith(".parquet"):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features)

    y = df[args.target].astype(int).values
    feat_cols = [
        "num_containers","req_cpu_m","req_mem_mb","req_gpu",
        "has_node_selector","node_selector_keys","has_affinity",
        "na_required_terms","na_preferred_terms",
        "num_tolerations","toleration_keys_count","tolerations_effect_noSchedule",
        "image_pull_always","num_images_ecr","num_images_dockerhub","num_images_other",
        "job_parallelism","job_completions",
        "cluster_node_count","cluster_gpu_capacity",
        "recent_failed_scheduling_ns","recent_image_pull_err_ns","recent_backoff_ns",
    ]
    X = df[feat_cols].fillna(0.0).astype(float).values

    # Train and evaluate
    results = {}
    cal_plots = []
    # Use a simple structure to capture plots
    # Defer matplotlib import to plotting block to avoid hard dependency

    # Fit models and generate plots
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) == 2 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, n_jobs=1)
    try:
        lr_cal = CalibratedClassifierCV(estimator=lr, cv=5, method="sigmoid")
    except TypeError:
        lr_cal = CalibratedClassifierCV(base_estimator=lr, cv=5, method="sigmoid")
    lr_cal.fit(X_train_s, y_train)
    lr_proba = lr_cal.predict_proba(X_test_s)[:, 1]

    hgb = HistGradientBoostingClassifier(random_state=42)
    try:
        hgb_cal = CalibratedClassifierCV(estimator=hgb, cv=5, method="sigmoid")
    except TypeError:
        hgb_cal = CalibratedClassifierCV(base_estimator=hgb, cv=5, method="sigmoid")
    hgb_cal.fit(X_train, y_train)
    hgb_proba = hgb_cal.predict_proba(X_test)[:, 1]

    def metrics(y_true, p):
        auroc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else float("nan")
        aupr = average_precision_score(y_true, p)
        brier = brier_score_loss(y_true, p)
        return {"auroc": float(auroc), "aupr": float(aupr), "brier": float(brier), "n_test": int(len(y_true))}

    results["lr"] = metrics(y_test, lr_proba)
    results["tree"] = metrics(y_test, hgb_proba)

    # XGBoost (optional)
    xgb_proba = None
    xgb_proba_iso = None
    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
        try:
            xgb_cal = CalibratedClassifierCV(estimator=xgb, cv=5, method="sigmoid")
        except TypeError:
            xgb_cal = CalibratedClassifierCV(base_estimator=xgb, cv=5, method="sigmoid")
        xgb_cal.fit(X_train, y_train)
        xgb_proba = xgb_cal.predict_proba(X_test)[:, 1]
        results["xgb"] = metrics(y_test, xgb_proba)

        # Isotonic calibration variant
        try:
            try:
                xgb_cal_iso = CalibratedClassifierCV(estimator=xgb, cv=5, method="isotonic")
            except TypeError:
                xgb_cal_iso = CalibratedClassifierCV(base_estimator=xgb, cv=5, method="isotonic")
            xgb_cal_iso.fit(X_train, y_train)
            xgb_proba_iso = xgb_cal_iso.predict_proba(X_test)[:, 1]
            results["xgb_iso"] = metrics(y_test, xgb_proba_iso)
        except Exception:
            pass
    except Exception:
        pass

    # Bootstrapped CIs (on test predictions) for each model
    def bootstrap_ci(y_true, proba, n=200):
        rng = np.random.default_rng(42)
        aurocs = []
        auprs = []
        briers = []
        n_ = len(y_true)
        idx = np.arange(n_)
        for _ in range(n):
            samp = rng.integers(0, n_, size=n_)
            yt = y_true[samp]
            pt = proba[samp]
            try:
                aurocs.append(roc_auc_score(yt, pt))
            except Exception:
                aurocs.append(np.nan)
            auprs.append(average_precision_score(yt, pt))
            briers.append(brier_score_loss(yt, pt))
        def ci(a):
            a = np.array(a, dtype=float)
            return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
        return {"auroc_ci": ci(aurocs), "aupr_ci": ci(auprs), "brier_ci": ci(briers)}

    # Attach CIs
    y_test_np = np.asarray(y_test)
    results["lr_ci"] = bootstrap_ci(y_test_np, lr_proba, n=args.bootstrap_n)
    results["tree_ci"] = bootstrap_ci(y_test_np, hgb_proba, n=args.bootstrap_n)
    if xgb_proba is not None:
        results["xgb_ci"] = bootstrap_ci(y_test_np, xgb_proba, n=args.bootstrap_n)
    if xgb_proba_iso is not None:
        results["xgb_iso_ci"] = bootstrap_ci(y_test_np, xgb_proba_iso, n=args.bootstrap_n)
    if 'rf_proba' in locals() and rf_proba is not None:
        results["rf_ci"] = bootstrap_ci(y_test_np, rf_proba, n=args.bootstrap_n)
    if 'et_proba' in locals() and et_proba is not None:
        results["et_ci"] = bootstrap_ci(y_test_np, et_proba, n=args.bootstrap_n)

    # Save metrics
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"metrics_{args.target}.json").write_text(json.dumps(results, indent=2))
    # Optional plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        # Prepare a dict of probabilities to plot if available
        plot_items = {
            "LR": lr_proba,
            "Tree": hgb_proba,
        }
        if xgb_proba is not None:
            plot_items["XGB"] = xgb_proba
        if xgb_proba_iso is not None:
            plot_items["XGB-ISO"] = xgb_proba_iso
        if 'rf_proba' in locals() and rf_proba is not None:
            plot_items["RF"] = rf_proba
        if 'et_proba' in locals() and et_proba is not None:
            plot_items["ET"] = et_proba

        for label, proba in plot_items.items():
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
            plt.plot(mean_pred, frac_pos, marker="o", lw=1, label=label)

        plt.legend()
        plt.title(f"Calibration: {args.target}")
        plt.xlabel("Mean predicted")
        plt.ylabel("Fraction positive")
        plt.tight_layout()
        plt.savefig(args.outdir / f"calibration_{args.target}.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Optional SHAP on XGB if available
    try:
        import shap  # type: ignore
        from xgboost import XGBClassifier  # type: ignore
        # If we have xgb_proba, rebuild a simple XGB (uncalibrated) for SHAP on full data
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        explainer = shap.TreeExplainer(xgb)
        # Sample to speed up
        samp = np.random.default_rng(0).choice(len(X_test), size=min(200, len(X_test)), replace=False)
        shap_values = explainer.shap_values(X_test[samp])
        shap.summary_plot(shap_values, X_test[samp], feature_names=[
            "num_containers","req_cpu_m","req_mem_mb","req_gpu",
            "has_node_selector","node_selector_keys","has_affinity",
            "na_required_terms","na_preferred_terms",
            "num_tolerations","toleration_keys_count","tolerations_effect_noSchedule",
            "image_pull_always","num_images_ecr","num_images_dockerhub","num_images_other",
            "job_parallelism","job_completions",
            "cluster_node_count","cluster_gpu_capacity",
            "recent_failed_scheduling_ns","recent_image_pull_err_ns","recent_backoff_ns",
        ], show=False)
        import matplotlib.pyplot as plt  # type: ignore
        plt.tight_layout()
        plt.savefig(args.outdir / f"shap_summary_{args.target}.png", dpi=150)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()


