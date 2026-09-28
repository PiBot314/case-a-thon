#!/usr/bin/env python
"""
build_submission.py - build/validate the three submission files.

    python scripts/build_submission.py                 # High + Medium tiers in the queue
    python scripts/build_submission.py --tiers High    # only High
    python scripts/build_submission.py --top 300       # top-300 by risk instead of tiers

Inputs (already produced by earlier steps)
  outputs/episode_predictions.csv      (train_models.py)
  outputs/submission_linkage.csv       (run_linkage.py)
  data/features/features_t0.*          (build_features.py)
  models/lost_to_followup_*.joblib     (train_models.py; used for per-episode SHAP reasons)
  reports/eval_stage_probabilities.csv (train_models.py; optional, picks the likely stalled stage)

Output (new)
  outputs/submission_action_queue.csv
    episode_id, predicted_patient_id, priority, reason, recommended_action, assigned_cadre
Also validates the episode-predictions and linkage files and prints a PASS/FAIL list.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.io_utils import load_df, audit

FEAT_DIR, OUT, MODELS, REPORTS = C.ROOT / "data" / "features", C.ROOT / "outputs", C.ROOT / "models", C.REPORT_DIR

PRED_COLS = ["episode_id", "risk_probability", "predicted_lost_to_followup", "predicted_dropout_stage", "priority_tier"]
LINK_COLS = ["source_system", "source_patient_id", "predicted_patient_id", "confidence"]
QUEUE_COLS = ["episode_id", "predicted_patient_id", "priority", "reason", "recommended_action", "assigned_cadre"]

VULNERABLE = {"elderly living alone", "low digital access", "mobility limitation", "seasonal migrant"}
SKIP_REASON_FEATS = {"facility_id", "block", "district", "consult_dow", "consult_is_weekend", "stk_snapshot_lag_months"}

LABELS = {
    "distance_to_facility_km": "Distance to facility", "rx_n_lines": "Prescription lines",
    "rx_n_medicines": "Number of medicines", "rx_n_classes": "Medicine classes", "rx_days_max": "Longest supply (days)",
    "review_due_days": "Review due (days)", "geo_road_access": "Road access", "geo_mobile_connectivity": "Village mobile coverage",
    "connectivity_quality": "Connectivity", "consult_mode": "Consult mode", "duration_min": "Consult duration (min)",
    "diagnosis_group": "Diagnosis group", "chief_complaint": "Complaint", "known_ncd_status": "NCD status",
    "vulnerability_group": "Vulnerability group", "age_t0": "Age", "ptele_n": "Earlier teleconsults",
    "pvisit_n": "Earlier visits", "pdisp_n": "Earlier dispensing records", "plab_n": "Earlier lab orders",
    "pncd_n": "Earlier NCD screenings", "fac_facility_type": "Facility type", "gender": "Gender",
    "preferred_language": "Language", "stk_stockout_days_mean": "Facility stock-out days last month",
}


def describe(col, v):
    """Human phrase for one driver; None if it should not be shown."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = None
    if col == "rxstk_any_stockout":
        return "Prescribed medicine out of stock at facility last month" if f and f > 0 else None
    if col == "rxstk_share_stockout":
        return f"{f:.0%} of prescribed medicines out of stock last month" if f and f > 0 else None
    if col == "stk_stockout_share":
        return "Facility had medicine stock-outs last month" if f and f > 0 else None
    if col == "stk_low_share":
        return "Facility stock was low last month" if f and f > 0 else None
    if col == "distance_to_facility_km":
        return f"Lives {f:.0f} km from facility" if f is not None else None
    if col == "n_advised_components":
        return f"{int(f)} follow-up steps advised (medicine/test/review)" if f is not None else None
    if col == "medicine_advised":
        return "Medicines to collect after consult" if f == 1 else None
    if col == "test_advised":
        return "Test advised, must be completed" if f == 1 else None
    if col == "review_advised":
        return "Review visit due" if f == 1 else None
    if col == "pdisp_partial_rate":
        return "History of partial medicine fills" if f and f > 0 else None
    if col == "ncd_followup_overdue_at_t0":
        return "NCD follow-up already overdue" if f == 1 else None
    if col == "vulnerability_group":
        return None if str(v).lower() in ("general", "unknown") else f"Vulnerability: {v}"
    if col in ("ptele_n", "pvisit_n", "pdisp_n", "plab_n", "pncd_n") and f == 0:
        return f"No {LABELS[col].lower().replace('earlier ', '')} on record"
    label = LABELS.get(col, col.replace("_", " ").capitalize())
    if f is not None:
        return f"{label}: {int(f) if float(f).is_integer() else round(f, 1)}"
    return f"{label}: {v}"


def shap_contributions(ev):
    """Per-episode, per-original-feature contribution to risk. Returns (DataFrame, cols) or (None, None)."""
    try:
        import joblib
        paths = sorted(MODELS.glob("lost_to_followup_*.joblib"))
        if not paths:
            raise FileNotFoundError("no model in models/")
        m = joblib.load(paths[0])
        pre, clf = m.named_steps["pre"], m.named_steps["clf"]
        cols = list(pre.feature_names_in_)
        Xt = np.asarray(pre.transform(ev[cols]), dtype=float)
        names = list(pre.get_feature_names_out())
        if hasattr(clf, "get_booster") or hasattr(clf, "feature_importances_"):
            import shap
            sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
        else:                                              # logistic regression: coef * centred value
            sv = (Xt - Xt.mean(0)) * clf.coef_[0]
        base = {}
        for j, n in enumerate(names):
            s = n.split("__", 1)[-1]
            hit = max((c for c in cols if s == c or s.startswith(c + "_")), key=len, default=None)
            base.setdefault(hit, []).append(j)
        contrib = pd.DataFrame({c: sv[:, idx].sum(1) for c, idx in base.items() if c is not None}, index=ev.index)
        print(f"reasons: using per-episode contributions from {paths[0].name}")
        return contrib
    except Exception as e:
        print(f"reasons: contributions unavailable ({type(e).__name__}: {e}); using rule-based reasons")
        return None


def rule_scores(ev):
    """Fallback ranking of drivers when SHAP is unavailable: simple operational flags."""
    s = pd.DataFrame(0.0, index=ev.index, columns=[c for c in ev.columns])
    def put(col, mask, w):
        if col in ev.columns:
            s.loc[mask.fillna(False), col] += w
    num = lambda c: pd.to_numeric(ev[c], errors="coerce") if c in ev.columns else pd.Series(np.nan, index=ev.index)
    put("rxstk_any_stockout", num("rxstk_any_stockout") > 0, 3)
    put("stk_stockout_share", num("stk_stockout_share") > 0, 2)
    put("distance_to_facility_km", num("distance_to_facility_km") > num("distance_to_facility_km").quantile(0.75), 2)
    put("n_advised_components", num("n_advised_components") >= 3, 2)
    put("test_advised", num("test_advised") == 1, 1)
    put("medicine_advised", num("medicine_advised") == 1, 1)
    put("pdisp_partial_rate", num("pdisp_partial_rate") > 0, 1.5)
    put("ncd_followup_overdue_at_t0", num("ncd_followup_overdue_at_t0") == 1, 1.5)
    if "vulnerability_group" in ev.columns:
        put("vulnerability_group", ~ev["vulnerability_group"].astype(str).str.lower().isin(["general", "unknown", "none", "nan"]), 1.5)
    return s


def top_reasons(ev, contrib, k=3):
    scores = contrib if contrib is not None else rule_scores(ev)
    out = []
    cols = [c for c in scores.columns if c not in SKIP_REASON_FEATS]
    S = scores[cols].to_numpy()
    for i in range(len(ev)):
        order = np.argsort(-S[i])
        phr = []
        for j in order:
            if S[i, j] <= 0:
                break
            p = describe(cols[j], ev.iloc[i][cols[j]] if cols[j] in ev.columns else None)
            if p and p not in phr:
                phr.append(p)
            if len(phr) == k:
                break
        out.append("; ".join(phr) if phr else "Elevated overall dropout risk score")
    return out


def stage_kind(text):
    t = str(text).lower()
    for k in ("medicine", "test", "review"):
        if k in t:
            return k
    return "other"


def action_for(tier, stage, reason, vulnerable, far):
    """Escalation ladder: SMS/IVR -> ASHA call -> home visit -> CHO / village health day."""
    stock = "out of stock" in reason.lower() or "stock-out" in reason.lower()
    if stock:
        stage = "medicine"
    what = {"medicine": "confirm medicines were collected" + ("; arrange alternate facility or delivery (stock-out)" if stock else ""),
            "test": "help book and complete the advised test (village health day or facility)",
            "review": "remind and schedule the review visit; arrange transport or a teleconsult review",
            "other": "check that medicines, test and review steps are on track"}[stage]
    if tier == "High":
        if stage == "test" or stock:
            return f"CHO/village health day: {what}", "CHO"
        if vulnerable or far:
            return f"ASHA home visit within 3 days: {what}", "ASHA"
        return f"ASHA phone call within 48h: {what}", "ASHA"
    return f"SMS/IVR reminder now; ASHA call in 3 days if no response: {what}", "ASHA"


def build_queue(a):
    pred = pd.read_csv(OUT / "episode_predictions.csv")
    fx = load_df(FEAT_DIR, "features_t0")
    ev = fx[fx["cohort"] == "EVALUATION"].copy()
    ev = ev.merge(pred, on="episode_id", how="inner").sort_values("risk_probability", ascending=False).reset_index(drop=True)
    if len(ev) != len(pred):
        print(f"[WARN] {len(pred) - len(ev)} predictions have no matching EVALUATION feature row")

    if (REPORTS / "eval_stage_probabilities.csv").exists():
        sp = pd.read_csv(REPORTS / "eval_stage_probabilities.csv").set_index("episode_id")
        sp.columns = [c.replace("p_stage_", "") for c in sp.columns]
        nc = [c for c in sp.columns if "complete" not in c.lower()]
        likely = sp.loc[ev["episode_id"], nc].idxmax(axis=1).values if nc else ev["predicted_dropout_stage"].values
    else:
        likely = ev["predicted_dropout_stage"].values
    ev["_stage"] = [stage_kind(s) for s in likely]

    sel = ev.index[ev["priority_tier"].isin(a.tiers)] if not a.top else ev.index[: a.top]
    q = ev.loc[sel].copy()
    contrib = shap_contributions(ev)
    q["reason"] = top_reasons(q, contrib.loc[q.index] if contrib is not None else None)

    dist = pd.to_numeric(ev.get("distance_to_facility_km"), errors="coerce")
    far_cut = dist.quantile(0.75) if dist is not None else np.inf
    acts, cad = [], []
    for _, r in q.iterrows():
        vul = str(r.get("vulnerability_group", "")).lower() in VULNERABLE
        far = pd.notna(r.get("distance_to_facility_km")) and r["distance_to_facility_km"] >= far_cut
        tier = r["priority_tier"] if r["priority_tier"] in ("High", "Medium") else "Medium"
        act, c = action_for(tier, r["_stage"], r["reason"], vul, far)
        acts.append(act); cad.append(c)
    q["recommended_action"], q["assigned_cadre"] = acts, cad
    q = q.rename(columns={"patient_id": "predicted_patient_id", "priority_tier": "priority"})
    q = q[QUEUE_COLS]
    q.to_csv(OUT / "submission_action_queue.csv", index=False)
    print(f"wrote outputs/submission_action_queue.csv ({len(q)} rows; "
          f"{q.priority.value_counts().to_dict()}; cadre {q.assigned_cadre.value_counts().to_dict()})")
    return q


def validate(q):
    res = []
    def chk(name, ok, detail=""):
        res.append(ok); print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    pred = pd.read_csv(OUT / "episode_predictions.csv")
    chk("predictions columns", list(pred.columns) == PRED_COLS)
    chk("predictions episode_id unique", pred.episode_id.is_unique, f"n={len(pred)}")
    chk("predictions no nulls", not pred.isna().any().any())
    chk("risk_probability in [0,1]", pred.risk_probability.between(0, 1).all())
    fx = load_df(FEAT_DIR, "features_t0")
    chk("predictions cover all EVALUATION episodes", set(pred.episode_id) == set(fx.loc[fx.cohort == "EVALUATION", "episode_id"]))
    lk = pd.read_csv(OUT / "submission_linkage.csv")
    chk("linkage columns", list(lk.columns) == LINK_COLS)
    chk("linkage no null patient ids", lk.predicted_patient_id.notna().all(), f"n={len(lk)}")
    chk("linkage (system, source id) unique", not lk.duplicated(["source_system", "source_patient_id"]).any())
    chk("linkage confidence in [0,1]", lk.confidence.between(0, 1).all())
    chk("queue columns", list(q.columns) == QUEUE_COLS)
    chk("queue episodes are in predictions", q.episode_id.isin(pred.episode_id).all())
    chk("queue no null patient/reason/action", not q[["predicted_patient_id", "reason", "recommended_action"]].isna().any().any())
    return all(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="High,Medium", help="tiers included in the queue")
    ap.add_argument("--top", type=int, default=0, help="instead of tiers, take the top-N by risk")
    a = ap.parse_args()
    a.tiers = [t.strip() for t in a.tiers.split(",")]
    q = build_queue(a)
    ok = validate(q)
    audit("submission_built", queue_rows=len(q), all_checks_passed=ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()