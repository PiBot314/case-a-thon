"""
features.py - Leakage-safe T0 feature store
===========================================
Prediction time T0 = END OF THE TELECONSULTATION DAY (consult_date).
Only information that exists at T0 is allowed:

  * consult attributes (advice given, mode, duration, distance, connectivity, ...)
  * canonical patient attributes (age at T0, vulnerability, known NCD status)
  * geography + facility context
  * prescription *existence/content* (issued at consult time; fills are NOT known)
  * facility / prescribed-medicine stock from the last snapshot month STRICTLY BEFORE
    the consult month
  * patient history with event date STRICTLY BEFORE consult_date
    (earlier teleconsults, NCD screenings, visits, dispensing, lab orders)

Explicitly excluded (post-T0 / outcome-side): consult_status, anything from
dispensing / lab / visits / outreach dated on or after consult_date, stage flags,
days_to_*, outreach counts, derived_dropout_stage, both labels.

Columns starting with `_audit_` record the latest source date used for each history
block; scripts/validate_features.py asserts they are all < consult_date. They are NOT
model features.

Output: data/features/features_t0.parquet (+ features_manifest.json)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config as C
from .io_utils import save_df, load_df, audit

FEATURE_DIR = C.ROOT / "data" / "features"
ID_COLS = ["episode_id", "teleconsult_id", "patient_id"]
META_COLS = ["consult_date", "cohort", "group_household"]
LABEL_COLS = ["lost_to_followup_label", "dropout_stage_label"]
FORBIDDEN_FEATURE_REGEX = (r"^(stage\d|days_to_|n_outreach|last_outreach|derived_|lost_to|dropout|"
                           r"consult_status|outreach|label)")

TELE_FEATS = ["age", "gender", "chief_complaint", "diagnosis_group", "consult_mode",
              "duration_min", "medicine_advised", "test_advised", "review_advised",
              "review_due_days", "distance_to_facility_km", "preferred_language",
              "connectivity_quality", "facility_id", "block", "district"]
P360_FEATS = ["date_of_birth", "age_as_of_2026", "village_id", "known_ncd_status",
              "vulnerability_group", "household_id", "gender", "preferred_language"]
GEO_FEATS = ["population", "road_access", "mobile_connectivity"]
FAC_FEATS = ["facility_type", "network_context", "cho_count", "asha_linked_count"]


def _load(name):
    try:
        return load_df(C.SILVER_DIR, name)
    except FileNotFoundError:
        return None


def _dt(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _attach_pid(df, pid_col, system, linkage):
    df = df.drop(columns=["patient_id"], errors="ignore")
    m = (linkage.loc[linkage["source_system"] == system, ["source_patient_id", "patient_id"]]
         .drop_duplicates("source_patient_id")
         .rename(columns={"source_patient_id": pid_col}))
    return df.merge(m, on=pid_col, how="left")


def _yes(s):
    return s.astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"]).astype(int)


def _prior(ep, ev, date_col, keep=(), drop_same_episode=False):
    """Event rows of the same patient with event date STRICTLY before consult_date."""
    left = ep[["episode_id", "patient_id", "consult_date"]].dropna(subset=["patient_id"])
    cols = ["patient_id", date_col] + [c for c in keep if c in ev.columns and c not in (date_col, "episode_id")]
    e = ev[cols].dropna(subset=["patient_id"]).rename(columns={date_col: "ev_date"})
    if drop_same_episode and "episode_id" in ev.columns:
        e["ev_episode_id"] = ev.loc[e.index, "episode_id"].values
    m = left.merge(e, on="patient_id", how="inner")
    m = m[m["ev_date"] < m["consult_date"]].copy()
    if drop_same_episode and "ev_episode_id" in m.columns:
        m = m[m["ev_episode_id"] != m["episode_id"]]
    m["days_before"] = (m["consult_date"] - m["ev_date"]).dt.days
    return m


def _agg_prior(ep, m, prefix, sums=()):
    g = m.groupby("episode_id")
    out = pd.DataFrame({
        f"{prefix}_n": g.size(),
        f"{prefix}_n_180d": m[m["days_before"] <= 180].groupby("episode_id").size(),
        f"{prefix}_days_since_last": g["days_before"].min(),
        f"_audit_{prefix}_last_date": g["ev_date"].max(),
    })
    for s in sums:
        out[f"{prefix}_{s}_sum"] = g[s].sum()
    out = out.reset_index()
    ep = ep.merge(out, on="episode_id", how="left")
    for c in [f"{prefix}_n", f"{prefix}_n_180d"] + [f"{prefix}_{s}_sum" for s in sums]:
        ep[c] = ep[c].fillna(0)
    return ep


def _stock_features(ep, stock, rx):
    stock = stock.copy()
    stock["snap"] = pd.to_datetime(stock["snapshot_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    stock = stock.dropna(subset=["snap"])
    stock["facility_id"] = stock["facility_id"].astype(str)
    for c in ["stockout_flag_month", "stockout_days", "closing_stock"]:
        stock[c] = pd.to_numeric(stock[c], errors="coerce") if c in stock.columns else np.nan
    if "stock_status" in stock.columns:
        stock["low"] = (~stock["stock_status"].astype(str).str.lower().str.contains("adequate")).astype(float)
    else:
        stock["low"] = np.nan
    fm = (stock.groupby(["facility_id", "snap"])
          .agg(stk_stockout_share=("stockout_flag_month", "mean"),
               stk_stockout_days_mean=("stockout_days", "mean"),
               stk_low_share=("low", "mean"),
               stk_closing_mean=("closing_stock", "mean")).reset_index().sort_values("snap"))
    left = ep[["episode_id", "facility_id", "consult_date"]].copy()
    left["facility_id"] = left["facility_id"].astype(str)
    left["consult_month"] = left["consult_date"].dt.to_period("M").dt.to_timestamp()
    left = left.sort_values("consult_month")
    m = pd.merge_asof(left, fm, left_on="consult_month", right_on="snap", by="facility_id",
                      allow_exact_matches=False)          # strictly earlier month
    m["_audit_stock_snapshot_month"] = m["snap"]
    m["stk_snapshot_lag_months"] = ((m["consult_month"].dt.year - m["snap"].dt.year) * 12
                                    + (m["consult_month"].dt.month - m["snap"].dt.month))
    out = m[["episode_id", "stk_stockout_share", "stk_stockout_days_mean", "stk_low_share",
             "stk_closing_mean", "stk_snapshot_lag_months", "_audit_stock_snapshot_month"]]

    # prescribed-medicine-specific stock (prior month, same facility)
    if rx is not None and "medicine_name" in rx.columns and "medicine_name" in stock.columns:
        r = rx[["teleconsult_id", "medicine_name"]].drop_duplicates()
        r = r.merge(ep[["episode_id", "teleconsult_id", "facility_id", "consult_date"]], on="teleconsult_id")
        r["facility_id"] = r["facility_id"].astype(str)
        r["med"] = r["medicine_name"].astype(str).str.strip().str.lower()
        r["consult_month"] = r["consult_date"].dt.to_period("M").dt.to_timestamp()
        r = r.sort_values("consult_month")
        sm = stock.assign(med=stock["medicine_name"].astype(str).str.strip().str.lower())
        sm = sm[["facility_id", "med", "snap", "stockout_flag_month", "stockout_days", "closing_stock"]].sort_values("snap")
        mm = pd.merge_asof(r, sm, left_on="consult_month", right_on="snap", by=["facility_id", "med"],
                           allow_exact_matches=False)
        agg = (mm.groupby("episode_id")
               .agg(rxstk_any_stockout=("stockout_flag_month", "max"),
                    rxstk_share_stockout=("stockout_flag_month", "mean"),
                    rxstk_stockout_days_max=("stockout_days", "max"),
                    rxstk_closing_min=("closing_stock", "min")).reset_index())
        out = out.merge(agg, on="episode_id", how="left")
    return out


def run() -> pd.DataFrame:
    audit("features_start")
    linkage, tele, eo = _load("patient_linkage"), _load("teleconsultations"), _load("episode_outcomes")
    if linkage is None or tele is None or eo is None:
        raise FileNotFoundError("Need silver patient_linkage, teleconsultations, episode_outcomes")
    p360, geo, fac = _load("patient_360_reference"), _load("geography_reference"), _load("facility_reference")
    rx, ncd = _load("prescriptions"), _load("ncd_screening")
    disp, lab, vh = _load("medicine_dispensing"), _load("lab_tests"), _load("visit_history")
    stock = _load("medicine_stock_status")

    tele = _attach_pid(_dt(tele, ["consult_date"]), "tele_source_patient_id", "teleconsultations", linkage)
    eo = _dt(eo, ["consult_date"])
    tcols = [c for c in TELE_FEATS if c in tele.columns]
    ep = eo[["episode_id", "teleconsult_id", "consult_date", "cohort"] + LABEL_COLS].merge(
        tele[["teleconsult_id", "patient_id"] + tcols], on="teleconsult_id", how="left")
    ep["lost_to_followup_label"] = pd.to_numeric(ep["lost_to_followup_label"], errors="coerce")
    n0 = len(ep)

    # ---- canonical patient attributes -------------------------------------------------
    if p360 is not None:
        pc = [c for c in P360_FEATS if c in p360.columns]
        p = p360[["patient_id"] + pc].drop_duplicates("patient_id").rename(
            columns={c: f"{c}_ref" for c in pc})
        ep = ep.merge(p, on="patient_id", how="left")
        for c in ("gender", "preferred_language"):
            if f"{c}_ref" in ep.columns:
                ep[c] = ep[c].fillna(ep[f"{c}_ref"]) if c in ep.columns else ep[f"{c}_ref"]
        dob = (pd.to_datetime(ep["date_of_birth_ref"], errors="coerce") if "date_of_birth_ref" in ep.columns
               else pd.Series(pd.NaT, index=ep.index))
        age_dob = (ep["consult_date"] - dob).dt.days / 365.25
        ep["age_t0"] = age_dob.fillna(pd.to_numeric(ep.get("age"), errors="coerce"))
        for c in ("known_ncd_status", "vulnerability_group"):
            if f"{c}_ref" in ep.columns:
                ep[c] = ep[f"{c}_ref"]
        hh = ep["household_id_ref"] if "household_id_ref" in ep.columns else None
        vid = ep["village_id_ref"] if "village_id_ref" in ep.columns else None
        ep = ep.drop(columns=[c for c in ep.columns if c.endswith("_ref")])
    else:
        ep["age_t0"] = pd.to_numeric(ep.get("age"), errors="coerce")
        hh, vid = None, None
    ep = ep.drop(columns=["age"], errors="ignore")
    ep["group_household"] = (pd.Series(hh, index=ep.index) if hh is not None else pd.Series(np.nan, index=ep.index))
    ep["group_household"] = ep["group_household"].fillna(ep["patient_id"]).fillna(ep["episode_id"]).astype(str)

    # ---- geography / facility ---------------------------------------------------------
    if geo is not None and vid is not None:
        g = geo[["village_id"] + [c for c in GEO_FEATS if c in geo.columns]].drop_duplicates("village_id")
        g = g.rename(columns={c: f"geo_{c}" for c in GEO_FEATS})
        ep["_vid"] = vid.values
        ep = ep.merge(g, left_on="_vid", right_on="village_id", how="left").drop(columns=["_vid", "village_id"])
    if fac is not None:
        f = fac[["facility_id"] + [c for c in FAC_FEATS if c in fac.columns]].drop_duplicates("facility_id")
        f = f.rename(columns={c: f"fac_{c}" for c in FAC_FEATS})
        ep = ep.merge(f, on="facility_id", how="left")

    # ---- consult attributes (T0-known) -----------------------------------------------
    for c in ("medicine_advised", "test_advised", "review_advised"):
        if c in ep.columns:
            ep[c] = _yes(ep[c])
    ep["n_advised_components"] = ep[[c for c in ("medicine_advised", "test_advised", "review_advised") if c in ep.columns]].sum(axis=1)
    for c in ("duration_min", "review_due_days", "distance_to_facility_km"):
        if c in ep.columns:
            ep[c] = pd.to_numeric(ep[c], errors="coerce")
    ep["consult_dow"] = ep["consult_date"].dt.dayofweek
    ep["consult_is_weekend"] = (ep["consult_dow"] >= 5).astype(int)

    # ---- prescription content (issued at consult time) --------------------------------
    if rx is not None:
        r = rx.copy()
        r["days_prescribed"] = pd.to_numeric(r.get("days_prescribed"), errors="coerce")
        agg = (r.groupby("teleconsult_id")
               .agg(rx_n_lines=("prescription_line_id", "count"),
                    rx_n_medicines=("medicine_name", "nunique"),
                    rx_n_classes=("medicine_class", "nunique"),
                    rx_days_max=("days_prescribed", "max"),
                    rx_days_mean=("days_prescribed", "mean")).reset_index())
        ep = ep.merge(agg, on="teleconsult_id", how="left")
        for c in ("rx_n_lines", "rx_n_medicines", "rx_n_classes"):
            ep[c] = ep[c].fillna(0)

    # ---- stock (prior month only) -----------------------------------------------------
    if stock is not None:
        ep = ep.merge(_stock_features(ep, stock, rx), on="episode_id", how="left")

    # ---- patient history, strictly before T0 -------------------------------------------
    t = tele[["patient_id", "consult_date", "teleconsult_id"]].copy()
    m = _prior(ep.assign(_k=1), t, "consult_date")
    m = m[m["ev_date"] < m["consult_date"]]
    ep = _agg_prior(ep, m, "ptele")

    if ncd is not None:
        n = _attach_pid(_dt(ncd, ["screening_date", "next_followup_due"]), "ncd_source_patient_id", "ncd_screening", linkage)
        for c in ("systolic_bp", "diastolic_bp", "random_glucose_mg_dl", "bmi"):
            if c in n.columns:
                n[c] = pd.to_numeric(n[c], errors="coerce")
        keep = [c for c in ("systolic_bp", "diastolic_bp", "random_glucose_mg_dl", "bmi", "ncd_status",
                            "control_status", "next_followup_due") if c in n.columns]
        m = _prior(ep, n, "screening_date", keep)
        ep = _agg_prior(ep, m, "pncd")
        last = m.sort_values("ev_date").groupby("episode_id").tail(1).set_index("episode_id")
        ren = {"systolic_bp": "ncd_last_sbp", "diastolic_bp": "ncd_last_dbp", "random_glucose_mg_dl": "ncd_last_glucose",
               "bmi": "ncd_last_bmi", "ncd_status": "ncd_last_status", "control_status": "ncd_last_control"}
        lastf = last[[c for c in ren if c in last.columns]].rename(columns=ren)
        if "next_followup_due" in last.columns:
            lastf["ncd_followup_overdue_at_t0"] = (last["next_followup_due"] < last["consult_date"]).astype(float)
        ep = ep.merge(lastf.reset_index(), on="episode_id", how="left")

    if vh is not None:
        v = _attach_pid(_dt(vh, ["visit_date"]), "visit_source_patient_id", "visit_history", linkage)
        v["ref_flag"] = _yes(v["referral_flag"]) if "referral_flag" in v.columns else 0
        m = _prior(ep, v, "visit_date", ["ref_flag", "episode_id"], drop_same_episode=True)
        ep = _agg_prior(ep, m, "pvisit", sums=["ref_flag"])

    if disp is not None:
        d = _attach_pid(_dt(disp, ["dispense_date"]), "pharm_source_patient_id", "medicine_dispensing", linkage)
        for c in ("partial_fill", "stockout_flag"):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0) if c in d.columns else 0
        m = _prior(ep, d, "dispense_date", ["partial_fill", "stockout_flag"])
        ep = _agg_prior(ep, m, "pdisp", sums=["partial_fill", "stockout_flag"])

    if lab is not None:
        l = _attach_pid(_dt(lab, ["order_date", "result_date"]), "lab_source_patient_id", "lab_tests", linkage)
        m = _prior(ep, l, "order_date", ["result_date"])
        m["done_by_t0"] = (m["result_date"].notna() & (m["result_date"] < m["consult_date"])).astype(float)
        ep = _agg_prior(ep, m, "plab", sums=["done_by_t0"])
        ep["plab_completion_rate"] = np.where(ep["plab_n"] > 0, ep["plab_done_by_t0_sum"] / ep["plab_n"].replace(0, np.nan), np.nan)

    if "pdisp_n" in ep.columns:
        ep["pdisp_partial_rate"] = np.where(ep["pdisp_n"] > 0, ep["pdisp_partial_fill_sum"] / ep["pdisp_n"].replace(0, np.nan), np.nan)

    assert len(ep) == n0 and ep["episode_id"].is_unique, "row explosion / duplicate episodes"

    # ---- finalise ---------------------------------------------------------------------
    audit_cols = [c for c in ep.columns if c.startswith("_audit_")]
    feat_cols = [c for c in ep.columns if c not in ID_COLS + META_COLS + LABEL_COLS + audit_cols]
    for c in feat_cols:
        if ep[c].dtype == bool:
            ep[c] = ep[c].astype(int)
        elif ep[c].dtype == object or str(ep[c].dtype) == "category":
            ep[c] = ep[c].astype(object).where(ep[c].notna(), None).map(lambda x: None if x is None else str(x).strip())
    cat_cols = [c for c in feat_cols if ep[c].dtype == object]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    ep = ep[ID_COLS + META_COLS + LABEL_COLS + feat_cols + audit_cols].sort_values(["consult_date", "episode_id"]).reset_index(drop=True)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    save_df(ep, FEATURE_DIR, "features_t0")
    manifest = dict(prediction_time="T0 = end of teleconsultation day (consult_date)",
                    id_columns=ID_COLS, meta_columns=META_COLS, label_columns=LABEL_COLS,
                    audit_columns=audit_cols, feature_columns=feat_cols,
                    categorical_columns=cat_cols, numeric_columns=num_cols,
                    forbidden_feature_regex=FORBIDDEN_FEATURE_REGEX,
                    excluded_by_design=["consult_status", "stage flags", "days_to_*", "outreach (same episode)",
                                        "dispensing/lab/visits on or after consult_date", "derived_dropout_stage", "labels"])
    (FEATURE_DIR / "features_manifest.json").write_text(json.dumps(manifest, indent=2))
    audit("features_end", rows=len(ep), n_features=len(feat_cols))
    print(f"features_t0: {len(ep):,} rows x {len(feat_cols)} features "
          f"({len(cat_cols)} categorical, {len(num_cols)} numeric) -> {FEATURE_DIR}")
    return ep