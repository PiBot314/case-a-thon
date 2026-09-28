#!/usr/bin/env python
"""
make_figures.py - presentation-ready charts from the pipeline outputs.

    python scripts/make_figures.py                 # writes figures/*.png (16:9-friendly, 200 dpi)
    python scripts/make_figures.py --outdir slides_img --dpi 300

Reads whatever exists (a missing input just skips that chart):
  reports/episode_summary.json, linkage_summary.json, model_metrics.json,
  oof_predictions.csv, calibration_table.csv, shap_importance.csv, fairness_report.csv
  outputs/episode_predictions.csv, submission_action_queue.csv

Only needs: pip install matplotlib pandas numpy
"""
import argparse, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("FUE_ROOT", Path(__file__).resolve().parents[1]))
REP, OUT = ROOT / "reports", ROOT / "outputs"

BLUE, RED, AMBER, GREEN, GREY, LIGHT, INK = "#1F6F8B", "#D1495B", "#EDAE49", "#2A9D8F", "#8A8F98", "#DDE3EA", "#1B2733"
TIER_COL = {"High": RED, "Medium": AMBER, "Low": LIGHT}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})
SIZE = (11, 6.2)
DONE, SKIPPED = [], []


def jload(p):
    return json.loads(Path(p).read_text())


def save(fig, name, a):
    a.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.outdir / name, dpi=a.dpi)
    plt.close(fig)
    DONE.append(name)
    print("  wrote", name)


def step(fn):
    def w(a):
        try:
            fn(a)
        except FileNotFoundError as e:
            SKIPPED.append(f"{fn.__name__}: missing {Path(str(e).split(chr(39))[1] if chr(39) in str(e) else str(e)).name}")
        except Exception as e:                                       # keep going, report at the end
            SKIPPED.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    return w


def need(p):
    if not Path(p).exists():
        raise FileNotFoundError(f"'{p}'")
    return p


def barlabels(ax, bars, fmt="{:.0f}", pad=3, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", ha="left", fontsize=11)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), va="bottom", ha="center", fontsize=11)


# ------------------------------------------------------------------------------------------------ 00 headline
@step
def fig_headline(a):
    ep = jload(need(REP / "episode_summary.json"))
    tiles = [(f"{ep['total_episodes']:,}", "episodes tracked\n(1 teleconsult + its care journey)")]
    try:
        lk = jload(REP / "linkage_summary.json")
        tiles.append((f"{sum(lk['by_type'].values()):,} \u2192 {lk['patients']:,}", "source records linked\nto unique patients"))
    except FileNotFoundError:
        pass
    d = ep["derived_dropout_stage_dist"]
    lost = 1 - d.get("Completed care journey", 0) / sum(d.values())
    tiles.append((f"{lost:.0%}", "of episodes end in dropout\n(medicine, test or review missed)"))
    try:
        m = jload(REP / "model_metrics.json")
        o = m["oof_pooled_primary"]
        pk = [k for k in o if k.startswith("precision@")][0]
        tiles.append((f"{o[pk]:.0%}", f"of the top {pk.split('top')[1]} we flag are\ntrue dropouts (random: {o['prevalence']:.0%})"))
        tiles.append((f"{o['lift_at_frac']:.1f}\u00d7", "lift over random calling"))
    except FileNotFoundError:
        pass
    try:
        q = pd.read_csv(OUT / "submission_action_queue.csv")
        tiles.append((f"{len(q):,}", "episodes on the ASHA/CHO\naction queue"))
    except FileNotFoundError:
        pass
    n = len(tiles); cols = 3; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(figsize=(11, 3.1 * rows)); ax.axis("off"); ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    for i, (big, small) in enumerate(tiles):
        x, y = i % cols, rows - 1 - i // cols
        ax.add_patch(plt.Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, color="#F3F6F9", ec=LIGHT))
        ax.text(x + 0.5, y + 0.62, big, ha="center", va="center", fontsize=30 if len(big) <= 7 else 23, fontweight="bold", color=BLUE)
        ax.text(x + 0.5, y + 0.26, small, ha="center", va="center", fontsize=11, color=INK)
    ax.set_title("Follow-Up Assurance Engine at a glance")
    save(fig, "00_headline.png", a)


# ------------------------------------------------------------------------------------------------ 01 cascade
@step
def fig_cascade(a):
    ep = jload(need(REP / "episode_summary.json"))
    r = ep["stage_completion_rates_dev"]
    names = ["Medicines\ncollected", "Tests\ncompleted", "Review visit\nattended", "Episode\nresolved"]
    vals = [r["stage1_medicine"], r["stage2_test"], r["stage3_review"], r["stage4_resolved"]]
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.bar(names, [v * 100 for v in vals], color=[BLUE, BLUE, BLUE, GREEN], width=0.6)
    barlabels(ax, bars, "{:.0f}%")
    ax.set_ylim(0, 100); ax.set_ylabel("% of episodes (development cohort)")
    ax.set_title("Most patients stall somewhere in the care journey")
    save(fig, "01_care_cascade.png", a)


# ------------------------------------------------------------------------------------------------ 02 stages
@step
def fig_stage_dist(a):
    ep = jload(need(REP / "episode_summary.json"))
    d = ep["derived_dropout_stage_dist"]
    tot = sum(d.values())
    items = sorted(d.items(), key=lambda kv: kv[1])
    cols = [GREEN if "omplete" in k else RED for k, _ in items]
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh([k for k, _ in items], [v / tot * 100 for _, v in items], color=cols, height=0.55)
    barlabels(ax, bars, "{:.0f}%", horizontal=True)
    ax.set_xlim(0, max(v for _, v in items) / tot * 100 * 1.18); ax.set_xlabel("% of episodes")
    lost = 1 - d.get("Completed care journey", 0) / tot
    ax.set_title(f"{lost:.0%} of episodes drop out; medicines and reviews are the main gaps")
    save(fig, "02_dropout_stage_distribution.png", a)


# ------------------------------------------------------------------------------------------------ 03 linkage
@step
def fig_linkage(a):
    lk = jload(need(REP / "linkage_summary.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    t = sorted(lk["by_type"].items(), key=lambda kv: kv[1])
    pretty = {"anchor_auto": "Auto-linked (high confidence)", "anchor_review": "Linked, flagged for review",
              "singleton": "Single-system patient", "cross_source_cluster": "Cross-system cluster",
              "cross_source_anchor": "Cross-system anchor"}
    bars = a1.barh([pretty.get(k, k) for k, _ in t], [v for _, v in t], color=BLUE, height=0.55)
    a1.set_xscale("log"); a1.set_xlim(1, max(v for _, v in t) * 60)
    for b in bars:
        a1.text(b.get_width() * 1.15, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,}", va="center", fontsize=11)
    a1.set_xlabel("source records (log scale)")
    a1.set_title("How records were linked")
    s = sorted(lk["by_system"].items(), key=lambda kv: kv[1])
    bars = a2.barh([k.replace("_", " ") for k, _ in s], [v * 100 for _, v in s], color=GREEN, height=0.55)
    barlabels(a2, bars, "{:.1f}%", horizontal=True)
    a2.set_xlim(80, 103); a2.set_xlabel("% linked to a patient_360 identity")
    a2.set_title("Coverage by source system")
    fig.suptitle(f"{sum(lk['by_type'].values()):,} fragmented records resolved into {lk['patients']:,} patients",
                 x=0.01, ha="left", fontsize=16, fontweight="bold", y=1.02)
    save(fig, "03_patient_linkage.png", a)


# ------------------------------------------------------------------------------------------------ 04 models
def _primary_cols(m):
    o = m["oof_pooled_primary"]
    pk = [k for k in o if k.startswith("precision@")][0]
    return pk


@step
def fig_models(a):
    m = jload(need(REP / "model_metrics.json"))
    by = m["mean_by_model"]; pk = _primary_cols(m); prev = m["oof_pooled_primary"]["prevalence"]
    panels = [("auc_pr", "AUC-PR", prev), ("roc_auc", "ROC-AUC", 0.5), (pk, f"Precision @ {pk.split('top')[1]}", prev)]
    names = list(by.keys()); colors = [GREY if n == "logreg" else BLUE for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, (k, title, base) in zip(axes, panels):
        vals = [by[n][k] for n in names]
        bars = ax.bar([n.replace("logreg", "Logistic\nregression").replace("xgboost", "XGBoost") for n in names], vals, color=colors, width=0.55)
        barlabels(ax, bars, "{:.2f}")
        ax.axhline(base, color=RED, ls="--", lw=1.5)
        ax.text(len(names) - 0.5, base, "random", color=RED, va="bottom", ha="right", fontsize=10)
        ax.set_ylim(0, max(vals + [base]) * 1.25); ax.set_title(title, fontsize=13)
    fig.suptitle("Both models beat random; gradient boosting edges out the baseline", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", y=1.03)
    save(fig, "04_model_comparison.png", a)


# ------------------------------------------------------------------------------------------------ 05 gains
@step
def fig_gains(a):
    m = jload(need(REP / "model_metrics.json"))
    oof = pd.read_csv(need(REP / "oof_predictions.csv"))
    col = f"p_{m['primary_model']}"
    o = oof.dropna(subset=[col]).sort_values(col, ascending=False)
    y = o["y"].to_numpy(); n = len(y); tot = y.sum(); kf = m.get("k_frac", 0.10)
    x = np.arange(1, n + 1) / n
    cap = np.cumsum(y) / tot; prec = np.cumsum(y) / np.arange(1, n + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))
    a1.plot(x * 100, cap * 100, color=BLUE, lw=2.5, label="Model ranking")
    a1.plot([0, 100], [0, 100], color=GREY, ls="--", label="Random calling")
    i = int(round(kf * n)) - 1
    a1.scatter([kf * 100], [cap[i] * 100], color=RED, zorder=5)
    a1.annotate(f"Call top {kf:.0%} \u2192 reach {cap[i]:.0%}\nof all dropouts", (kf * 100, cap[i] * 100),
                xytext=(kf * 100 + 22, cap[i] * 100 - 12), arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=11)
    a1.set_xlabel("% of episodes contacted (highest risk first)"); a1.set_ylabel("% of dropouts reached")
    a1.set_title("Cumulative gains"); a1.legend(frameon=False, loc="upper left")
    lo = max(int(0.02 * n), 20)
    a2.plot(x[lo:] * 100, prec[lo:] * 100, color=BLUE, lw=2.5)
    a2.axhline(y.mean() * 100, color=GREY, ls="--"); a2.text(2, y.mean() * 100 - 2, "random", ha="left", va="top", color=GREY)
    a2.scatter([kf * 100], [prec[i] * 100], color=RED, zorder=5)
    a2.annotate(f"{prec[i]:.0%} precision\nat top {kf:.0%}", (kf * 100, prec[i] * 100), xytext=(kf * 100 + 14, prec[i] * 100 + 3),
                arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=11)
    a2.set_ylim(0, 100); a2.set_xlabel("% of episodes contacted"); a2.set_ylabel("% of contacted who are true dropouts")
    a2.set_title("Precision as the call list grows")
    fig.suptitle("Prioritised calling finds more dropouts per call than random", x=0.01, ha="left", fontsize=16, fontweight="bold", y=1.03)
    save(fig, "05_gains_and_precision.png", a)


# ------------------------------------------------------------------------------------------------ 06 calibration
@step
def fig_calibration(a):
    c = pd.read_csv(need(REP / "calibration_table.csv"))
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", label="Perfect calibration")
    ax.plot(c["mean_pred"], c["observed"], color=BLUE, lw=1.5, alpha=0.6)
    ax.scatter(c["mean_pred"], c["observed"], s=c["n"] / c["n"].max() * 260 + 40, color=BLUE, zorder=3, label="Risk decile")
    lim = [0, max(1.0, float(max(c["mean_pred"].max(), c["observed"].max())) * 1.05)]
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed dropout rate"); ax.legend(frameon=False, loc="upper left")
    ax.set_title("Predicted risk tracks observed risk")
    ax.text(0.99, 0.02, "isotonic calibration fitted on out-of-fold\npredictions (not independently validated)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=GREY)
    save(fig, "06_calibration.png", a)


# ------------------------------------------------------------------------------------------------ 07 shap
PRETTY = {"n_advised_components": "Number of steps advised", "rx_n_lines": "Prescription lines", "rx_n_medicines": "Medicines prescribed",
          "rx_n_classes": "Medicine classes", "distance_to_facility_km": "Distance to facility (km)", "medicine_advised": "Medicines advised",
          "review_advised": "Review advised", "test_advised": "Test advised", "geo_road_access": "Village road access",
          "rx_days_max": "Longest supply (days)", "review_due_days": "Review due (days)", "age_t0": "Patient age",
          "duration_min": "Consult duration", "stk_stockout_share": "Facility stock-out share (prior month)",
          "rxstk_any_stockout": "Prescribed medicine out of stock", "facility_id": "Facility"}


@step
def fig_shap(a):
    s = pd.read_csv(need(REP / "shap_importance.csv"))
    s["base"] = s["feature"].str.replace(r"^(num|cat)__", "", regex=True)
    g = s.groupby("base")["mean_abs_shap"].sum().sort_values(ascending=False).head(12).iloc[::-1]
    labels = [PRETTY.get(k, k.replace("_", " ")) for k in g.index]
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(labels, g.values, color=BLUE, height=0.6)
    ax.set_xlabel("Mean |SHAP| (average influence on predicted risk)")
    ax.set_title("What drives dropout risk: the size of the care plan, and access")
    save(fig, "07_top_risk_drivers.png", a)


# ------------------------------------------------------------------------------------------------ 08 fairness
@step
def fig_fairness(a):
    f = pd.read_csv(need(REP / "fairness_report.csv"))
    f = f[f["n"] >= 30].copy()
    f["label"] = f["attribute"].str.replace("_", " ").str.capitalize() + ": " + f["group"].astype(str)
    order = {"gender": 0, "age_band": 1, "preferred_language": 2, "vulnerability_group": 3}
    f["o"] = f["attribute"].map(order).fillna(9)
    f = f.sort_values(["o", "flag_rate"], ascending=[False, True])
    pal = {"gender": BLUE, "age_band": GREEN, "preferred_language": AMBER, "vulnerability_group": RED}
    cols = [pal.get(x, GREY) for x in f["attribute"]]
    tot = f[f["attribute"] == f["attribute"].iloc[0]]
    overall_flag = np.average(tot["flag_rate"], weights=tot["n"]); overall_tpr = np.average(tot["tpr"], weights=tot["n"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    for ax, col, ov, ttl in ((a1, "flag_rate", overall_flag, "Share flagged for follow-up"), (a2, "tpr", overall_tpr, "Dropouts caught (recall)")):
        bars = ax.barh(f["label"], f[col] * 100, color=cols, height=0.65)
        barlabels(ax, bars, "{:.0f}%", horizontal=True)
        ax.axvline(ov * 100, color=INK, ls="--", lw=1); ax.set_title(ttl, fontsize=13)
        ax.set_xlim(0, f[col].max() * 100 * 1.2); ax.set_xlabel("%")
    fig.suptitle("Fairness check: who gets flagged, and who is caught (top-10% call list)", x=0.01, ha="left", fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.01, -0.02, "Dashed line = overall rate. Groups with fewer than 30 episodes omitted.", fontsize=10, color=GREY)
    save(fig, "08_fairness.png", a)


# ------------------------------------------------------------------------------------------------ 09 risk tiers
@step
def fig_tiers(a):
    p = pd.read_csv(need(OUT / "episode_predictions.csv"))
    fig, ax = plt.subplots(figsize=SIZE)
    bins = np.linspace(p["risk_probability"].min(), p["risk_probability"].max(), 31)
    bottom = np.zeros(len(bins) - 1)
    for t in ("Low", "Medium", "High"):
        h, _ = np.histogram(p.loc[p["priority_tier"] == t, "risk_probability"], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", bottom=bottom, color=TIER_COL[t],
               label=f"{t}  ({(p['priority_tier'] == t).sum():,})", edgecolor="white", lw=0.5)
        bottom += h
    ax.set_xlabel("Predicted dropout risk"); ax.set_ylabel("Episodes (evaluation cohort)")
    ax.legend(title="Priority tier", frameon=False)
    ax.set_title(f"{len(p):,} upcoming episodes ranked into High / Medium / Low priority")
    save(fig, "09_risk_tiers.png", a)


# ------------------------------------------------------------------------------------------------ 10 action queue
THEMES = [("Facility / medicine stock-out", r"out of stock|stock-out|stock was low"), ("Long distance to facility", r"km from facility"),
          ("Many follow-up steps advised", r"follow-up steps"), ("Test advised", r"test advised"),
          ("Medicines to collect", r"medicines to collect"), ("Vulnerable patient", r"vulnerab"),
          ("Past partial fills", r"partial"), ("NCD follow-up overdue", r"ncd follow-up")]


@step
def fig_queue(a):
    q = pd.read_csv(need(OUT / "submission_action_queue.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ct = pd.crosstab(q["priority"], q["assigned_cadre"]).reindex(["High", "Medium"]).fillna(0)
    bottom = np.zeros(len(ct)); pal = {"ASHA": BLUE, "CHO": AMBER}
    for cad in ct.columns:
        bars = a1.bar(ct.index, ct[cad], bottom=bottom, color=pal.get(cad, GREY), label=cad, width=0.55)
        for b, v, bt in zip(bars, ct[cad], bottom):
            if v > 0:
                a1.text(b.get_x() + b.get_width() / 2, bt + v / 2, f"{int(v)}", ha="center", va="center", color="white", fontweight="bold")
        bottom += ct[cad].to_numpy()
    a1.set_ylabel("Episodes on the worklist"); a1.legend(title="Assigned cadre", frameon=False); a1.set_title("Worklist by priority and cadre")
    share = []
    for name, rx in THEMES:
        share.append((name, q["reason"].astype(str).str.lower().str.contains(rx, regex=True).mean() * 100))
    share = sorted([s for s in share if s[1] > 0], key=lambda kv: kv[1])
    bars = a2.barh([s[0] for s in share], [s[1] for s in share], color=GREEN, height=0.6)
    barlabels(a2, bars, "{:.0f}%", horizontal=True)
    a2.set_xlim(0, max(s[1] for s in share) * 1.2 if share else 100); a2.set_xlabel("% of worklist rows citing this reason")
    a2.set_title("Why patients are flagged")
    fig.suptitle(f"Daily action queue: {len(q):,} episodes, each with a reason and a suggested action", x=0.01, ha="left", fontsize=16, fontweight="bold", y=1.02)
    save(fig, "10_action_queue.png", a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=ROOT / "figures")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    print(f"writing figures to {a.outdir}")
    for fn in (fig_headline, fig_cascade, fig_stage_dist, fig_linkage, fig_models, fig_gains, fig_calibration,
               fig_shap, fig_fairness, fig_tiers, fig_queue):
        fn(a)
    print(f"\n{len(DONE)} figures written.")
    for s in SKIPPED:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
