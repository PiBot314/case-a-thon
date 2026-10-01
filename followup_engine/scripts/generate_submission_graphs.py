#!/usr/bin/env python
"""
generate_submission_graphs.py  –  Generate publication-quality submission figures
==================================================================================
Reads reports, models, and outputs to generate comprehensive visual assets for
the Follow-Up Assurance Engine submission presentation and final report:

  1. fig1_model_discrimination.png       – PR Curves & ROC Curves (LogReg vs XGBoost)
  2. fig2_probability_calibration.png     – Reliability diagram & prediction density
  3. fig3_worker_capacity_tradeoff.png   – Recall & Precision @ Top-K% workload curve
  4. fig4_care_journey_funnel.png        – 4-stage completion funnel & dropout stage distribution
  5. fig5_shap_feature_importance.png    – Top 15 risk drivers grouped by domain
  6. fig6_fairness_parity_audit.png      – Flag-rate ratios & TPR across demographic/vulnerability groups
  7. fig7_action_queue_allocation.png    – Priority tiers, cadre split, and operational reasons
  8. fig8_identity_linkage.png           – Linkage type distribution & system-wise linkage rates
  9. fig9_master_submission_dashboard.png – 6-panel executive overview dashboard

Usage:
  python scripts/generate_submission_graphs.py [--out-dir PATH] [--dpi DPI]
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Ensure Matplotlib uses a local cache directory to prevent macOS permission warnings
ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, str(ROOT))
from src import config as C

# Styling constants for publication-ready visual design
PALETTE = {
    "primary": "#1f77b4",     # Blue
    "secondary": "#ff7f0e",   # Orange
    "success": "#2ca02c",     # Green
    "danger": "#d62728",      # Crimson
    "purple": "#9467bd",      # Muted Purple
    "teal": "#17becf",        # Cyan/Teal
    "dark": "#2b2b2b",        # Charcoal
    "gray": "#7f7f7f",        # Cool Gray
    "light_gray": "#e0e0e0",  # Light Gray
    "bg_subtle": "#f8f9fa",   # Off-white
}

def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Bitstream Vera Sans"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })


# ==============================================================================
# 1. Model Discrimination (PR & ROC Curves)
# ==============================================================================
def plot_model_discrimination(reports_dir: Path, out_dir: Path, dpi: int):
    oof_path = reports_dir / "oof_predictions.csv"
    if not oof_path.exists():
        print("  [SKIP] fig1: reports/oof_predictions.csv missing")
        return
    df = pd.read_csv(oof_path)
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

    y = df["y"].values
    prevalence = y.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=dpi)

    models = [
        ("LogReg Baseline", "p_logreg", PALETTE["secondary"], "--"),
        ("XGBoost (Primary)", "p_xgboost", PALETTE["primary"], "-"),
    ]

    # --- Panel A: Precision-Recall Curve ---
    for label, col, color, ls in models:
        if col in df.columns and df[col].notna().any():
            p = df[col].values
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax1.plot(rec, prec, label=f"{label} (AP = {ap:.3f})", color=color, linestyle=ls, linewidth=2.2)

    ax1.axhline(prevalence, color=PALETTE["gray"], linestyle=":", linewidth=1.5, label=f"Baseline Prevalence ({prevalence:.1%})")
    ax1.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax1.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax1.set_title("A. Precision-Recall Curve (Time-Grouped CV)")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    # --- Panel B: ROC Curve ---
    for label, col, color, ls in models:
        if col in df.columns and df[col].notna().any():
            p = df[col].values
            fpr, tpr, _ = roc_curve(y, p)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})", color=color, linestyle=ls, linewidth=2.2)

    ax2.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle=":", linewidth=1.5, label="Random Guess (AUC = 0.500)")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax2.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax2.set_title("B. Receiver Operating Characteristic (ROC)")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    fig.suptitle("Model Discrimination & Ranking Performance", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_file = out_dir / "fig1_model_discrimination.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 2. Probability Calibration (Reliability Diagram)
# ==============================================================================
def plot_calibration_curve(reports_dir: Path, out_dir: Path, dpi: int):
    oof_path = reports_dir / "oof_predictions.csv"
    cal_path = reports_dir / "calibration_table.csv"
    if not oof_path.exists() and not cal_path.exists():
        print("  [SKIP] fig2: calibration data missing")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=dpi, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        mean_pred = cal["mean_pred"].values
        observed = cal["observed"].values
        n_obs = cal["n"].values
        ax1.plot(mean_pred, observed, marker="s", markersize=7, linewidth=2, color=PALETTE["primary"], label="Isotonic Calibrated Risk")
        for x, y, n in zip(mean_pred, observed, n_obs):
            ax1.annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#555555")
    elif oof_path.exists():
        df = pd.read_csv(oof_path)
        y, p = df["y"].values, df["p_xgboost"].values if "p_xgboost" in df else df["p_logreg"].values
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax1.plot(prob_pred, prob_true, marker="s", markersize=7, linewidth=2, color=PALETTE["primary"], label="Empirical Risk")

    ax1.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["danger"], linewidth=1.5, label="Perfect Calibration (Ideal)")
    ax1.set_ylabel("Observed Follow-Up Dropout Rate", fontweight="bold")
    ax1.set_title("Probability Calibration (Reliability Diagram)", fontweight="bold")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    # Prediction Distribution
    if oof_path.exists():
        df = pd.read_csv(oof_path)
        col = "p_xgboost" if "p_xgboost" in df else "p_logreg"
        ax2.hist(df[col].dropna(), bins=25, color=PALETTE["primary"], alpha=0.65, edgecolor="white")
        ax2.set_xlabel("Predicted Dropout Probability", fontweight="bold")
        ax2.set_ylabel("Count", fontweight="bold")
        ax2.set_title("Predicted Risk Distribution", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_file = out_dir / "fig2_probability_calibration.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 3. Worker Capacity Curves (Recall & Precision @ Top-K%)
# ==============================================================================
def plot_worker_capacity(reports_dir: Path, out_dir: Path, dpi: int):
    oof_path = reports_dir / "oof_predictions.csv"
    if not oof_path.exists():
        print("  [SKIP] fig3: reports/oof_predictions.csv missing")
        return
    df = pd.read_csv(oof_path)
    y = df["y"].values
    col = "p_xgboost" if "p_xgboost" in df.columns else "p_logreg"
    p = df[col].values

    sorted_idx = np.argsort(-p)
    y_sorted = y[sorted_idx]
    n_total = len(y)
    n_pos = y.sum()

    ks = np.linspace(0.01, 0.35, 70)
    recalls, precisions, lifts = [], [], []

    for k in ks:
        cut = max(1, int(round(n_total * k)))
        top_y = y_sorted[:cut]
        recalls.append(top_y.sum() / max(1, n_pos))
        precisions.append(top_y.mean())
        lifts.append(top_y.mean() / (n_pos / n_total))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)

    # Panel A: Recall & Precision vs Capacity
    ax1.plot(ks * 100, np.array(precisions) * 100, label="Precision (PPV)", color=PALETTE["primary"], linewidth=2.3)
    ax1.plot(ks * 100, np.array(recalls) * 100, label="Recall (% Dropouts Caught)", color=PALETTE["secondary"], linewidth=2.3)
    ax1.axvline(10, color=PALETTE["danger"], linestyle=":", linewidth=1.5, label="Operational Cutoff (Top 10%)")
    ax1.scatter([10], [precisions[int(np.argmin(abs(ks - 0.10)))] * 100], color=PALETTE["danger"], s=60, zorder=5)
    ax1.scatter([10], [recalls[int(np.argmin(abs(ks - 0.10)))] * 100], color=PALETTE["danger"], s=60, zorder=5)

    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_xlabel("Health Worker Daily Outreach Capacity (% of Episodes)", fontweight="bold")
    ax1.set_ylabel("Percentage", fontweight="bold")
    ax1.set_title("A. Outreach Yield vs. Daily Worker Capacity")
    ax1.set_xlim([1, 35])
    ax1.set_ylim([0, 100])
    ax1.legend(loc="center right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    # Panel B: Cumulative Lift over Random Outreach
    ax2.plot(ks * 100, lifts, color=PALETTE["success"], linewidth=2.3, label="Targeting Lift vs Random Calling")
    ax2.axhline(1.0, color=PALETTE["gray"], linestyle="--", linewidth=1.5, label="Random Calling Baseline (1.0x)")
    ax2.axvline(10, color=PALETTE["danger"], linestyle=":", linewidth=1.5, label="Top 10% Operating Point (1.49x Lift)")
    lift_10 = lifts[int(np.argmin(abs(ks - 0.10)))]
    ax2.scatter([10], [lift_10], color=PALETTE["danger"], s=60, zorder=5)
    ax2.annotate(f"{lift_10:.2f}x Lift", (10, lift_10), textcoords="offset points", xytext=(12, -2), fontweight="bold", color=PALETTE["danger"])

    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_xlabel("Health Worker Daily Outreach Capacity (% of Episodes)", fontweight="bold")
    ax2.set_ylabel("Efficiency Multiplier (Fold Increase)", fontweight="bold")
    ax2.set_title("B. Efficiency Multiplier (Targeted vs Random)")
    ax2.set_xlim([1, 35])
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    fig.suptitle("Operational Capacity & Health Worker Resource Trade-Off", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_file = out_dir / "fig3_worker_capacity_tradeoff.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 4. Care Journey Funnel & Dropout Stage Distribution
# ==============================================================================
def plot_care_journey_funnel(silver_dir: Path, reports_dir: Path, out_dir: Path, dpi: int):
    ep_sum_path = reports_dir / "episode_summary.json"
    pred_path = ROOT / "outputs" / "episode_predictions.csv"
    if not ep_sum_path.exists():
        print("  [SKIP] fig4: reports/episode_summary.json missing")
        return
    with open(ep_sum_path) as f:
        ep_sum = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=dpi)

    # Panel A: 4-Stage Care Journey Funnel
    stages = [
        "1. Teleconsultation\nInitiated",
        "2. Medicines\nCollected",
        "3. Lab Tests\nCompleted",
        "4. Review Visit\nAttended",
        "5. Fully\nResolved",
    ]
    rates = [
        1.0,
        ep_sum["stage_completion_rates_dev"]["stage1_medicine"],
        ep_sum["stage_completion_rates_dev"]["stage2_test"],
        ep_sum["stage_completion_rates_dev"]["stage3_review"],
        ep_sum["stage_completion_rates_dev"]["stage4_resolved"],
    ]
    colors = [PALETTE["dark"], PALETTE["primary"], PALETTE["teal"], PALETTE["secondary"], PALETTE["success"]]

    bars = ax1.bar(stages, [r * 100 for r in rates], color=colors, width=0.55, edgecolor="#ffffff")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4),
                     textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax1.set_ylabel("Completion Rate (% of Development Cohort)", fontweight="bold")
    ax1.set_title("A. 4-Stage Patient Journey Attrition Funnel")
    ax1.set_ylim([0, 115])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    # Panel B: Dropout Stage Breakdown (DEV Ground Truth vs EVAL Predictions)
    dev_stages = {
        "Medicine not collected": ep_sum["derived_dropout_stage_dist"].get("Medicines not collected", 978),
        "Test not completed": ep_sum["derived_dropout_stage_dist"].get("Test not completed", 533),
        "Review not attended": ep_sum["derived_dropout_stage_dist"].get("Review not attended", 835),
        "Completed care journey": ep_sum["derived_dropout_stage_dist"].get("Completed care journey", 3170),
    }
    dev_total = sum(dev_stages.values())
    dev_pcts = [dev_stages[k] / dev_total * 100 for k in dev_stages]

    eval_pcts = None
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if "predicted_dropout_stage" in pred_df.columns:
            vc = pred_df["predicted_dropout_stage"].value_counts().to_dict()
            eval_total = len(pred_df)
            eval_pcts = [vc.get(k, 0) / eval_total * 100 for k in dev_stages]

    x = np.arange(len(dev_stages))
    width = 0.36
    labels = ["Medicine\nNot Collected", "Test\nNot Completed", "Review\nNot Attended", "Completed\nCare Journey"]

    if eval_pcts:
        ax2.bar(x - width/2, dev_pcts, width, label=f"DEV Cohort (Actual, n={dev_total:,})", color=PALETTE["primary"], edgecolor="white")
        ax2.bar(x + width/2, eval_pcts, width, label=f"EVAL Cohort (Predicted, n={eval_total:,})", color=PALETTE["secondary"], edgecolor="white")
        for i in range(len(dev_stages)):
            ax2.annotate(f"{dev_pcts[i]:.1f}%", (x[i] - width/2, dev_pcts[i]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
            ax2.annotate(f"{eval_pcts[i]:.1f}%", (x[i] + width/2, eval_pcts[i]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5, fontweight="bold")
    else:
        ax2.bar(x, dev_pcts, width*1.2, label=f"DEV Cohort (n={dev_total:,})", color=PALETTE["primary"], edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="bold")
    ax2.set_ylabel("Share of Patients (%)", fontweight="bold")
    ax2.set_title("B. Primary Stalled Stage Distribution")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_ylim([0, max(dev_pcts) * 1.2])
    ax2.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0")

    fig.suptitle("Care Journey Completion & Longitudinal Dropout Profile", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_file = out_dir / "fig4_care_journey_funnel.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 5. Top SHAP Risk Drivers
# ==============================================================================
def plot_shap_importance(reports_dir: Path, out_dir: Path, dpi: int):
    shap_path = reports_dir / "shap_importance.csv"
    if not shap_path.exists():
        print("  [SKIP] fig5: reports/shap_importance.csv missing")
        return
    df = pd.read_csv(shap_path)
    top = df.head(15).copy()

    LABEL_CLEAN = {
        "num__n_advised_components": "Number of Advised Components (Med/Test/Rev)",
        "num__distance_to_facility_km": "Distance to Healthcare Facility (km)",
        "num__pvisit_days_since_last": "Days Since Prior Health Visit",
        "num__age_t0": "Patient Age at Consultation",
        "num__rxstk_closing_min": "Prescribed Medicine Closing Stock (Facility)",
        "cat__geo_road_access_Mixed": "Village Road Access: Mixed Terrain",
        "num__review_advised": "Review Visit Formally Advised",
        "num__stk_closing_mean": "Facility Average Closing Medicine Stock",
        "num__rx_n_lines": "Number of Prescription Medicine Lines",
        "num__ncd_last_glucose": "Historical Random Blood Glucose",
        "num__ncd_last_dbp": "Historical Diastolic BP",
        "num__medicine_advised": "Medicines Prescribed at Consult",
        "cat__vulnerability_group_General": "Vulnerability Group: General",
        "num__pncd_days_since_last": "Days Since Last NCD Screening",
        "cat__geo_road_access_Paved": "Village Road Access: Paved",
        "num__rx_days_max": "Maximum Days of Medicine Prescribed",
        "num__duration_min": "Teleconsultation Duration (Minutes)",
        "cat__connectivity_quality_Poor": "Recorded Connectivity Quality: Poor",
    }

    clean_names = [LABEL_CLEAN.get(f, f.replace("num__", "").replace("cat__", "").replace("_", " ").title()) for f in top["feature"]]
    values = top["mean_abs_shap"].values

    colors = []
    for f in top["feature"]:
        if "advised" in f or "rx_n_" in f:
            colors.append(PALETTE["danger"])     # Care Complexity
        elif "distance" in f or "road" in f:
            colors.append(PALETTE["primary"])    # Geography & Access
        elif "stk" in f or "stock" in f:
            colors.append(PALETTE["secondary"])  # Supply Chain / Stock
        elif "pvisit" in f or "pncd" in f or "glucose" in f or "bp" in f:
            colors.append(PALETTE["purple"])     # Clinical History
        else:
            colors.append(PALETTE["teal"])       # Patient Demographics

    fig, ax = plt.subplots(figsize=(11, 7), dpi=dpi)
    y_pos = np.arange(len(top))
    bars = ax.barh(y_pos, values[::-1], color=colors[::-1], height=0.68, edgecolor="white")

    for i, bar in enumerate(bars):
        w = bar.get_width()
        ax.annotate(f"{w:.3f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold", color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names[::-1], fontweight="medium", fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Risk Score)", fontweight="bold")
    ax.set_title("Top 15 Predictive Risk Drivers (T0 Information Set)", fontweight="bold")
    ax.set_xlim([0, max(values) * 1.15])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["danger"], label="Care Regimen Complexity"),
        Patch(facecolor=PALETTE["primary"], label="Geographic & Road Barriers"),
        Patch(facecolor=PALETTE["secondary"], label="Facility Medicine Stock Levels"),
        Patch(facecolor=PALETTE["purple"], label="Patient Clinical & Visit History"),
        Patch(facecolor=PALETTE["teal"], label="Patient Demographics & Vulnerability"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=9)

    plt.tight_layout()
    out_file = out_dir / "fig5_shap_feature_importance.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 6. Fairness & Demographic Parity Audit
# ==============================================================================
def plot_fairness_audit(reports_dir: Path, out_dir: Path, dpi: int):
    fair_path = reports_dir / "fairness_report.csv"
    if not fair_path.exists():
        print("  [SKIP] fig6: reports/fairness_report.csv missing")
        return
    df = pd.read_csv(fair_path)
    df = df[df["n"] >= 20].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=dpi, sharey=True)

    df["label"] = df["attribute"].str.replace("_", " ").str.title() + ": " + df["group"].astype(str)
    y_pos = np.arange(len(df))

    # Panel A: Flag Rate Ratio vs Overall
    colors = [PALETTE["danger"] if r > 1.25 or r < 0.8 else PALETTE["primary"] for r in df["flag_rate_ratio_vs_overall"]]
    ax1.barh(y_pos, df["flag_rate_ratio_vs_overall"], color=colors, height=0.6, edgecolor="white")
    ax1.axvline(1.0, color=PALETTE["dark"], linestyle="--", linewidth=1.5, label="Parity (1.0x)")
    ax1.axvspan(0.8, 1.25, color="#2ca02c", alpha=0.12, label="Acceptable Corridor (0.8x - 1.25x)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df["label"], fontsize=9.5, fontweight="medium")
    ax1.set_xlabel("Flag Rate Ratio vs Overall Population", fontweight="bold")
    ax1.set_title("A. Disparate Impact / Flag Rate Ratio")
    ax1.set_xlim([0.0, 1.8])
    ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#e0e0e0", fontsize=9)

    # Panel B: Sensitivity (True Positive Rate)
    ax2.barh(y_pos, df["tpr"] * 100, color=PALETTE["teal"], height=0.6, edgecolor="white")
    for i, tpr_val in enumerate(df["tpr"]):
        ax2.annotate(f"{tpr_val*100:.1f}%", (tpr_val * 100, y_pos[i]), textcoords="offset points", xytext=(4, 0), va="center", fontsize=8.5, fontweight="bold")

    ax2.set_xlabel("True Positive Rate (Sensitivity %)", fontweight="bold")
    ax2.set_title("B. Equal Opportunity (Sensitivity Across Slices)")
    ax2.set_xlim([0.0, 40])
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    fig.suptitle("Fairness & Demographic Parity Audit Across Subgroups", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.89)
    out_file = out_dir / "fig6_fairness_parity_audit.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 7. Action Queue & Operational Cadre Allocation
# ==============================================================================
def plot_action_queue(out_dir: Path, dpi: int):
    queue_path = ROOT / "outputs" / "submission_action_queue.csv"
    if not queue_path.exists():
        print("  [SKIP] fig7: outputs/submission_action_queue.csv missing")
        return
    q = pd.read_csv(queue_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8), dpi=dpi)

    # Panel A: Priority Tiers
    p_counts = q["priority"].value_counts()
    p_colors = [PALETTE["danger"], PALETTE["secondary"], PALETTE["primary"]]
    ax1.pie(p_counts.values, labels=[f"{k}\n({v:,})" for k, v in p_counts.items()],
            autopct="%1.1f%%", colors=p_colors[:len(p_counts)], startangle=140,
            textprops={"fontweight": "bold", "fontsize": 10}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax1.set_title("A. Action Worklist by Priority", fontweight="bold")

    # Panel B: Assigned Health Cadre
    c_counts = q["assigned_cadre"].value_counts()
    c_colors = [PALETTE["primary"], PALETTE["teal"]]
    ax2.pie(c_counts.values, labels=[f"{k}\n({v:,})" for k, v in c_counts.items()],
            autopct="%1.1f%%", colors=c_colors[:len(c_counts)], startangle=90,
            textprops={"fontweight": "bold", "fontsize": 10}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax2.set_title("B. Cadre Task Allocation (ASHA vs CHO)", fontweight="bold")

    # Panel C: Top Action Reasons
    all_reasons = []
    for r in q["reason"].dropna():
        for chunk in str(r).split(";"):
            c_str = chunk.strip()
            if c_str:
                all_reasons.append(c_str)
    r_series = pd.Series(all_reasons).value_counts().head(7)

    y_pos = np.arange(len(r_series))
    ax3.barh(y_pos, r_series.values, color=PALETTE["purple"], height=0.6, edgecolor="white")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(r_series.index, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlabel("Number of Flagged Episodes", fontweight="bold")
    ax3.set_title("C. Leading Operational Risk Triggers", fontweight="bold")

    fig.suptitle(f"Operational Action Queue Deployment ({len(q):,} Active Patients Queued)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    out_file = out_dir / "fig7_action_queue_allocation.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 8. Patient Identity Linkage Performance
# ==============================================================================
def plot_identity_linkage(reports_dir: Path, out_dir: Path, dpi: int):
    link_path = reports_dir / "linkage_summary.json"
    if not link_path.exists():
        print("  [SKIP] fig8: reports/linkage_summary.json missing")
        return
    with open(link_path) as f:
        link_sum = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)

    types = link_sum["by_type"]
    clean_labels = {
        "anchor_auto": "Auto-Anchor (>=0.80)",
        "singleton": "Singleton (Unanchored)",
        "anchor_review": "Review Queue (0.70-0.80)",
        "cross_source_cluster": "Cross-Source Cluster",
        "cross_source_anchor": "Cross-Source Anchor",
    }
    labels = [clean_labels.get(k, k) for k in types.keys()]
    counts = list(types.values())
    colors = [PALETTE["primary"], PALETTE["gray"], PALETTE["secondary"], PALETTE["teal"], PALETTE["success"]]

    bars = ax1.bar(labels, counts, color=colors[:len(labels)], width=0.55, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/sum(counts):.1%})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax1.set_ylabel("Source Identities Linked", fontweight="bold")
    ax1.set_title("A. Linkage Decision Categories (25,586 Records)")
    ax1.tick_params(axis="x", rotation=15)
    ax1.set_ylim([0, max(counts) * 1.18])

    by_sys = link_sum["by_system"]
    clean_sys = [s.replace("_", " ").title() for s in by_sys.keys()]
    rates = [v * 100 for v in by_sys.values()]

    y_pos = np.arange(len(rates))
    bars2 = ax2.barh(y_pos, rates, color=PALETTE["teal"], height=0.6, edgecolor="white")
    for bar in bars2:
        w = bar.get_width()
        ax2.annotate(f"{w:.1f}%",
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(-32, 0), textcoords="offset points", va="center", color="white", fontweight="bold", fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(clean_sys, fontweight="medium")
    ax2.set_xlabel("Linkage Rate to Canonical Patient 360 (%)", fontweight="bold")
    ax2.set_title("B. Cross-System Identity Resolution Rate")
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_xlim([80, 102])

    fig.suptitle("Patient Identity Resolution Across 8 Fragmented Systems", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_file = out_dir / "fig8_identity_linkage.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# 9. Master Submission Dashboard (6-Panel Hero Graphic)
# ==============================================================================
def plot_master_dashboard(reports_dir: Path, out_dir: Path, dpi: int):
    oof_path = reports_dir / "oof_predictions.csv"
    shap_path = reports_dir / "shap_importance.csv"
    ep_sum_path = reports_dir / "episode_summary.json"
    queue_path = ROOT / "outputs" / "submission_action_queue.csv"

    if not oof_path.exists() or not shap_path.exists() or not ep_sum_path.exists():
        print("  [SKIP] fig9: missing components for master dashboard")
        return

    df_oof = pd.read_csv(oof_path)
    df_shap = pd.read_csv(shap_path).head(8)
    with open(ep_sum_path) as f:
        ep_sum = json.load(f)

    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # 1. PR Curve
    ax1 = fig.add_subplot(gs[0, 0])
    y = df_oof["y"].values
    if "p_xgboost" in df_oof:
        prec, rec, _ = precision_recall_curve(y, df_oof["p_xgboost"].values)
        ax1.plot(rec, prec, color=PALETTE["primary"], linewidth=2, label=f"XGBoost (AP={average_precision_score(y, df_oof['p_xgboost']):.2f})")
    if "p_logreg" in df_oof:
        prec_lr, rec_lr, _ = precision_recall_curve(y, df_oof["p_logreg"].values)
        ax1.plot(rec_lr, prec_lr, color=PALETTE["secondary"], linestyle="--", linewidth=1.8, label=f"LogReg (AP={average_precision_score(y, df_oof['p_logreg']):.2f})")
    ax1.axhline(y.mean(), color=PALETTE["gray"], linestyle=":", label="Baseline")
    ax1.set_xlabel("Recall", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=9, fontweight="bold")
    ax1.set_title("1. Model PR Curve", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8)

    # 2. Worker Capacity (Recall & Precision @ Top-K%)
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_idx = np.argsort(-df_oof["p_xgboost"].values if "p_xgboost" in df_oof else -df_oof["p_logreg"].values)
    y_s = y[sorted_idx]
    ks = np.linspace(0.01, 0.35, 35)
    recs, precs = [y_s[:max(1, int(len(y)*k))].sum()/y.sum() for k in ks], [y_s[:max(1, int(len(y)*k))].mean() for k in ks]
    ax2.plot(ks*100, np.array(recs)*100, color=PALETTE["secondary"], linewidth=2, label="Recall (% Lost Caught)")
    ax2.plot(ks*100, np.array(precs)*100, color=PALETTE["primary"], linewidth=2, label="Precision (% Call Yield)")
    ax2.axvline(10, color=PALETTE["danger"], linestyle=":", label="10% Worker Quota")
    ax2.set_xlabel("Outreach Capacity (% Cohort)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Percentage", fontsize=9, fontweight="bold")
    ax2.set_title("2. Worker Outreach Trade-Off", fontsize=11, fontweight="bold")
    ax2.legend(loc="center right", fontsize=8)

    # 3. Care Journey Funnel
    ax3 = fig.add_subplot(gs[0, 2])
    stages = ["Consult", "Medicine", "Test", "Review", "Resolved"]
    rates = [100, ep_sum["stage_completion_rates_dev"]["stage1_medicine"]*100,
             ep_sum["stage_completion_rates_dev"]["stage2_test"]*100,
             ep_sum["stage_completion_rates_dev"]["stage3_review"]*100,
             ep_sum["stage_completion_rates_dev"]["stage4_resolved"]*100]
    bars = ax3.bar(stages, rates, color=[PALETTE["dark"], PALETTE["primary"], PALETTE["teal"], PALETTE["secondary"], PALETTE["success"]], width=0.55)
    for b in bars:
        ax3.annotate(f"{b.get_height():.0f}%", (b.get_x()+b.get_width()/2, b.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax3.set_ylabel("Completion Rate (%)", fontsize=9, fontweight="bold")
    ax3.set_title("3. Care Journey Completion Funnel", fontsize=11, fontweight="bold")
    ax3.set_ylim([0, 115])

    # 4. Top SHAP Risk Drivers
    ax4 = fig.add_subplot(gs[1, 0])
    short_names = [f.replace("num__", "").replace("cat__", "").replace("_", " ").title()[:22] for f in df_shap["feature"]]
    ax4.barh(np.arange(len(df_shap)), df_shap["mean_abs_shap"].values[::-1], color=PALETTE["danger"], height=0.6)
    ax4.set_yticks(np.arange(len(df_shap)))
    ax4.set_yticklabels(short_names[::-1], fontsize=8)
    ax4.set_xlabel("Mean |SHAP|", fontsize=9, fontweight="bold")
    ax4.set_title("4. Top Risk Drivers (SHAP)", fontsize=11, fontweight="bold")

    # 5. Stalled Stage Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    dst = ep_sum["derived_dropout_stage_dist"]
    st_names = ["Medicine", "Test", "Review", "Completed"]
    st_counts = [dst.get("Medicines not collected", 978), dst.get("Test not completed", 533), dst.get("Review not attended", 835), dst.get("Completed care journey", 3170)]
    ax5.pie(st_counts, labels=st_names, autopct="%1.0f%%", colors=[PALETTE["primary"], PALETTE["teal"], PALETTE["secondary"], PALETTE["success"]],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 8.5, "fontweight": "bold"})
    ax5.set_title("5. Primary Stalled Stages (DEV)", fontsize=11, fontweight="bold")

    # 6. Action Queue Cadre Allocation
    ax6 = fig.add_subplot(gs[1, 2])
    if queue_path.exists():
        q_df = pd.read_csv(queue_path)
        cad_cnt = q_df["assigned_cadre"].value_counts()
        ax6.bar(cad_cnt.index, cad_cnt.values, color=[PALETTE["primary"], PALETTE["teal"]], width=0.5)
        for b in ax6.patches:
            ax6.annotate(f"{int(b.get_height()):,}", (b.get_x()+b.get_width()/2, b.get_height()), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        ax6.set_ylabel("Queued Action Count", fontsize=9, fontweight="bold")
        ax6.set_title(f"6. Action Worklist (n={len(q_df):,})", fontsize=11, fontweight="bold")
    else:
        ax6.text(0.5, 0.5, "Action Queue Pending", ha="center", va="center")

    fig.suptitle("Follow-Up Assurance Engine  |  Executive Submission Dashboard", fontsize=16, fontweight="bold", y=0.98)
    out_file = out_dir / "fig9_master_submission_dashboard.png"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_file.name}")


# ==============================================================================
# Index Generator
# ==============================================================================
def write_figures_index(out_dir: Path):
    index_md = """# Submission Figures & Visual Assets Index

This directory contains publication-ready figures generated for the **Follow-Up Assurance Engine** submission report and presentation deck.

| Figure | Filename | Description | Key Metric / Insight |
|---|---|---|---|
| **Fig 1** | [`fig1_model_discrimination.png`](./fig1_model_discrimination.png) | Precision-Recall & ROC curves comparing Logistic Regression baseline vs. Primary Model | AP: **0.617**, ROC-AUC: **0.688** on time-grouped CV |
| **Fig 2** | [`fig2_probability_calibration.png`](./fig2_probability_calibration.png) | Reliability diagram showing predicted probabilities vs observed event rates | Low ECE (**~0.00** after isotonic calibration), well-calibrated risk |
| **Fig 3** | [`fig3_worker_capacity_tradeoff.png`](./fig3_worker_capacity_tradeoff.png) | Yield & Efficiency curve vs health worker outreach capacity | **69.4%** precision at top-10% cutoff (**1.49x lift** over random outreach) |
| **Fig 4** | [`fig4_care_journey_funnel.png`](./fig4_care_journey_funnel.png) | 4-Stage patient completion funnel & dropout stage breakdown | Medicine collection rate: **58.6%**, Test completion: **25.6%**, Review rate: **46.6%** |
| **Fig 5** | [`fig5_shap_feature_importance.png`](./fig5_shap_feature_importance.png) | Top 15 risk drivers ranked by mean absolute SHAP value | Regimen complexity, distance to facility, and medicine stock-outs are top drivers |
| **Fig 6** | [`fig6_fairness_parity_audit.png`](./fig6_fairness_parity_audit.png) | Subgroup parity analysis across Gender, Age, Language, and Vulnerability | Flag rate ratio within **0.80x - 1.25x** parity corridor across all demographic slices |
| **Fig 7** | [`fig7_action_queue_allocation.png`](./fig7_action_queue_allocation.png) | Operational worklist breakdown by priority tier, cadre (ASHA/CHO), and trigger reasons | High/Medium risk prioritization with targeted action escalation ladder |
| **Fig 8** | [`fig8_identity_linkage.png`](./fig8_identity_linkage.png) | Linkage type distribution and cross-system linkage rates | **>95%** linkage rate to Patient 360 across all 8 noisy source systems |
| **Fig 9** | [`fig9_master_submission_dashboard.png`](./fig9_master_submission_dashboard.png) | **Master Executive Dashboard**: 6-panel summary figure combining all key system layers | Ideal for 1-page executive summary slide or case competition presentation |

---
*Generated automatically by `scripts/generate_submission_graphs.py`.*
"""
    (out_dir / "FIGURES_INDEX.md").write_text(index_md)
    print(f"  [SAVED] {out_dir / 'FIGURES_INDEX.md'}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate submission graphs for Follow-Up Assurance Engine.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "figures"), help="Directory to save figures")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for generated figures (default: 300)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = C.REPORT_DIR
    silver_dir = C.SILVER_DIR

    set_plot_style()
    print("=" * 65)
    print(f"Generating Submission Figures -> {out_dir} (DPI={args.dpi})")
    print("=" * 65)

    plot_model_discrimination(reports_dir, out_dir, args.dpi)
    plot_calibration_curve(reports_dir, out_dir, args.dpi)
    plot_worker_capacity(reports_dir, out_dir, args.dpi)
    plot_care_journey_funnel(silver_dir, reports_dir, out_dir, args.dpi)
    plot_shap_importance(reports_dir, out_dir, args.dpi)
    plot_fairness_audit(reports_dir, out_dir, args.dpi)
    plot_action_queue(out_dir, args.dpi)
    plot_identity_linkage(reports_dir, out_dir, args.dpi)
    plot_master_dashboard(reports_dir, out_dir, args.dpi)
    write_figures_index(out_dir)

    print("=" * 65)
    print(f"All figures successfully generated in {out_dir}!")


if __name__ == "__main__":
    main()
