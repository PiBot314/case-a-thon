#!/usr/bin/env python
"""
Independent validation of the T0 feature store.   Run:  python scripts/validate_features.py
Optional:  --sample 300  --auc-warn 0.70 --auc-fail 0.85 --seed 7

Writes reports/feature_validation.md and exits non-zero if any check FAILS.
Checks 1-4 are structural, 5 re-derives features by brute force from the silver tables
(different code path from src/features.py), 6-8 are statistical leakage / drift screens.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.io_utils import load_df

FEAT_DIR = C.ROOT / "data" / "features"
results = []


def rec(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status:4}] {name}" + (f" - {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--auc-warn", type=float, default=0.70)
    ap.add_argument("--auc-fail", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    fx = load_df(FEAT_DIR, "features_t0")
    man = json.loads((FEAT_DIR / "features_manifest.json").read_text())
    feats, cats, nums = man["feature_columns"], man["categorical_columns"], man["numeric_columns"]
    fx["consult_date"] = pd.to_datetime(fx["consult_date"])
    eo = load_df(C.SILVER_DIR, "episode_outcomes")

    # 1 ---- shape / keys ---------------------------------------------------------------
    rec("PASS" if len(fx) == len(eo) else "FAIL", "1a row count == episode_outcomes", f"{len(fx)} vs {len(eo)}")
    rec("PASS" if fx["episode_id"].is_unique else "FAIL", "1b episode_id unique")
    rec("PASS" if set(fx["episode_id"]) == set(eo["episode_id"]) else "FAIL", "1c episode_id set matches outcomes")
    pn = fx["patient_id"].isna().mean()
    rec("PASS" if pn == 0 else ("WARN" if pn < 0.02 else "FAIL"), "1d patient_id null share", f"{pn:.2%}")

    # 2 ---- cohorts / labels -----------------------------------------------------------
    dev, ev = fx[fx.cohort == "DEVELOPMENT"], fx[fx.cohort == "EVALUATION"]
    rec("PASS" if dev["lost_to_followup_label"].notna().all() else "FAIL", "2a DEV labels all present")
    rec("PASS" if ev["lost_to_followup_label"].isna().all() else "FAIL", "2b EVAL labels all blank")
    rec("PASS" if dev.consult_date.max() <= pd.Timestamp(C.DEV_END) else "FAIL", "2c DEV dates <= cutoff", str(dev.consult_date.max().date()))
    ok = ev.consult_date.between(pd.Timestamp(C.EVAL_START), pd.Timestamp(C.EVAL_END)).all()
    rec("PASS" if ok else "FAIL", "2d EVAL dates inside window")
    rec("PASS", "2e development prevalence", f"lost_to_followup = {dev['lost_to_followup_label'].mean():.3f}, n_dev={len(dev)}, n_eval={len(ev)}")

    # 3 ---- forbidden columns ------------------------------------------------------------
    bad = [c for c in feats if re.search(man["forbidden_feature_regex"], c, flags=re.I)]
    rec("PASS" if not bad else "FAIL", "3a no forbidden feature names", ", ".join(bad))
    overlap = set(feats) & set(man["label_columns"] + man["id_columns"] + man["meta_columns"] + man["audit_columns"])
    rec("PASS" if not overlap else "FAIL", "3b features disjoint from ids/labels/audit", ", ".join(overlap))
    inf = [c for c in nums if np.isinf(pd.to_numeric(fx[c], errors="coerce")).any()]
    rec("PASS" if not inf else "FAIL", "3c no inf values", ", ".join(inf))

    # 4 ---- audit dates strictly before T0 ---------------------------------------------
    for c in man["audit_columns"]:
        d = pd.to_datetime(fx[c], errors="coerce")
        if c == "_audit_stock_snapshot_month":
            cm = fx["consult_date"].dt.to_period("M").dt.to_timestamp()
            viol = int((d.notna() & (d >= cm)).sum())
        else:
            viol = int((d.notna() & (d >= fx["consult_date"])).sum())
        rec("PASS" if viol == 0 else "FAIL", f"4 {c} strictly < T0", f"violations={viol}, coverage={d.notna().mean():.1%}")

    # 5 ---- brute-force recompute on a random sample ---------------------------------
    linkage = load_df(C.SILVER_DIR, "patient_linkage")

    def pid(df, col, sysname):
        m = linkage[linkage.source_system == sysname][["source_patient_id", "patient_id"]].drop_duplicates("source_patient_id")
        return df.drop(columns=["patient_id"], errors="ignore").merge(m.rename(columns={"source_patient_id": col}), on=col, how="left")

    tele = pid(load_df(C.SILVER_DIR, "teleconsultations"), "tele_source_patient_id", "teleconsultations")
    tele["consult_date"] = pd.to_datetime(tele["consult_date"])
    vh = pid(load_df(C.SILVER_DIR, "visit_history"), "visit_source_patient_id", "visit_history")
    vh["visit_date"] = pd.to_datetime(vh["visit_date"])
    dsp = pid(load_df(C.SILVER_DIR, "medicine_dispensing"), "pharm_source_patient_id", "medicine_dispensing")
    dsp["dispense_date"] = pd.to_datetime(dsp["dispense_date"])
    lab = pid(load_df(C.SILVER_DIR, "lab_tests"), "lab_source_patient_id", "lab_tests")
    lab["order_date"] = pd.to_datetime(lab["order_date"])
    ncd = pid(load_df(C.SILVER_DIR, "ncd_screening"), "ncd_source_patient_id", "ncd_screening")
    ncd["screening_date"] = pd.to_datetime(ncd["screening_date"])
    ep_pid = tele.set_index("teleconsult_id")["patient_id"]

    rng = np.random.default_rng(a.seed)
    samp = fx[fx.patient_id.notna()].sample(min(a.sample, len(fx)), random_state=a.seed)
    mism = {k: 0 for k in ("ptele_n", "pvisit_n", "pdisp_n", "plab_n", "pncd_n")}
    for _, r in samp.iterrows():
        p, t0, eid = r["patient_id"], r["consult_date"], r["episode_id"]
        exp = {
            "ptele_n": int(((tele.patient_id == p) & (tele.consult_date < t0)).sum()),
            "pvisit_n": int(((vh.patient_id == p) & (vh.visit_date < t0) & (vh.get("episode_id") != eid)).sum()),
            "pdisp_n": int(((dsp.patient_id == p) & (dsp.dispense_date < t0)).sum()),
            "plab_n": int(((lab.patient_id == p) & (lab.order_date < t0)).sum()),
            "pncd_n": int(((ncd.patient_id == p) & (ncd.screening_date < t0)).sum()),
        }
        for k, v in exp.items():
            if k in r.index and int(r[k]) != v:
                mism[k] += 1
    for k, v in mism.items():
        if k in fx.columns:
            rec("PASS" if v == 0 else "FAIL", f"5 recompute {k} on {len(samp)} episodes", f"mismatches={v}")

    # 6 ---- single-feature leakage screen ------------------------------------------------
    from sklearn.metrics import roc_auc_score
    y = dev["lost_to_followup_label"].astype(int)
    rows = []
    for c in feats:
        x = dev[c]
        x = x.astype("category").cat.codes if c in cats else pd.to_numeric(x, errors="coerce")
        ok = x.notna()
        if ok.sum() < 50 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
            continue
        au = roc_auc_score(y[ok], x[ok])
        rows.append((c, max(au, 1 - au), ok.mean()))
    top = sorted(rows, key=lambda r: -r[1])[:10]
    worst = top[0][1] if top else 0
    st = "FAIL" if worst >= a.auc_fail else ("WARN" if worst >= a.auc_warn else "PASS")
    rec(st, "6 max single-feature AUC vs label", "; ".join(f"{c}={v:.3f}" for c, v, _ in top[:5]))
    (C.REPORT_DIR).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows, columns=["feature", "single_feature_auc", "coverage"]).sort_values(
        "single_feature_auc", ascending=False).to_csv(C.REPORT_DIR / "feature_single_auc.csv", index=False)

    # 7 ---- missingness / constants ---------------------------------------------------
    miss = fx[feats].isna().mean().sort_values(ascending=False)
    rec("WARN" if (miss > 0.6).any() else "PASS", "7a features >60% missing", ", ".join(f"{c}={v:.0%}" for c, v in miss[miss > 0.6].items()))
    const = [c for c in feats if fx[c].nunique(dropna=True) <= 1]
    rec("WARN" if const else "PASS", "7b constant features", ", ".join(const))
    hc = [c for c in cats if fx[c].nunique() > 50]
    rec("WARN" if hc else "PASS", "7c high-cardinality categoricals (>50)", ", ".join(hc))
    ms = (dev[feats].isna().mean() - ev[feats].isna().mean()).abs()
    rec("WARN" if (ms > 0.15).any() else "PASS", "7d missingness DEV vs EVAL gap >15pp", ", ".join(f"{c}={v:.0%}" for c, v in ms[ms > 0.15].items()))

    # 8 ---- DEV vs EVAL drift ------------------------------------------------------------
    from scipy.stats import ks_2samp
    dr = []
    for c in nums:
        a1, b1 = pd.to_numeric(dev[c], errors="coerce").dropna(), pd.to_numeric(ev[c], errors="coerce").dropna()
        if len(a1) > 30 and len(b1) > 30 and a1.nunique() > 1:
            dr.append((c, ks_2samp(a1, b1).statistic))
    dr.sort(key=lambda r: -r[1])
    rec("WARN" if dr and dr[0][1] > 0.2 else "PASS", "8 numeric drift DEV vs EVAL (KS>0.2)",
        "; ".join(f"{c}={v:.2f}" for c, v in dr[:5]))
    cat_shift = []
    for c in cats:
        unseen = set(ev[c].dropna().unique()) - set(dev[c].dropna().unique())
        if unseen:
            cat_shift.append(f"{c}:{len(unseen)} unseen")
    rec("WARN" if cat_shift else "PASS", "8b categories in EVAL never seen in DEV", ", ".join(cat_shift))

    # ---- report ---------------------------------------------------------------------
    nf = sum(1 for s, *_ in results if s == "FAIL")
    nw = sum(1 for s, *_ in results if s == "WARN")
    lines = ["# Feature Store Validation", f"FAIL: **{nf}**  WARN: **{nw}**  PASS: **{len(results)-nf-nw}**", "",
             "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {s} | {n} | {d} |" for s, n, d in results]
    (C.REPORT_DIR / "feature_validation.md").write_text("\n".join(lines))
    print(f"\nSummary: {nf} FAIL, {nw} WARN -> reports/feature_validation.md")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()