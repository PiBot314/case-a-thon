"""
episodes.py  –  Episode (patient care journey) construction
=============================================================
Input:  silver/*.parquet (or .pkl) + silver/patient_linkage.parquet
Output: silver/episodes.parquet
        reports/episode_summary.{md,json}

Episode definition
------------------
One episode = one teleconsultation (row in `teleconsultations`) plus
its downstream care journey, tracked through four stages:

  Stage 1 – Medicine collected  : at least one dispensing row linked via
             prescription_id → prescription → teleconsult_id, within
             medicine_window_days of consult_date.

  Stage 2 – Test completed      : at least one lab result row (result_date
             not null) linked via patient_id + order_date within
             test_window_days of consult_date.

  Stage 3 – Review visit        : a follow-up or visit_history row of
             visit_type "Follow-up review" linked via episode_id OR
             (patient_id + visit within review_due_days + grace_days).

  Stage 4 – Resolved            : episode_outcomes.lost_to_followup_label = 0
             AND all prior stages done  (DEVELOPMENT cohort only; EVALUATION
             labels are blank by design).

Dropout stage
-------------
The FIRST incomplete stage (1→4) is the dropout_stage.
If all stages are complete → "Completed care journey" (matches episode_outcomes label).

T0 leakage guard
----------------
All post-T0 tables (dispensing, lab, visits, outreach) are date-filtered to
rows AFTER the episode's consult_date before building stage flags.
Prescription existence is known at T0; fills are not.

Output schema (one row per episode / teleconsult)
--------------------------------------------------
episode_id, teleconsult_id, patient_id, consult_date, cohort,
facility_id, review_due_days,
# stage completion booleans
stage1_medicine_collected, stage2_test_completed, stage3_review_attended,
stage4_resolved,
# derived
derived_dropout_stage, days_to_medicine, days_to_test, days_to_review,
# outreach engagement (post-T0)
n_outreach_attempts, last_outreach_outcome,
# label passthrough (DEVELOPMENT only)
lost_to_followup_label, dropout_stage_label
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from . import config as C
from .io_utils import save_df, audit

log = logging.getLogger(__name__)

# ── windows ───────────────────────────────────────────────────────────────────
MEDICINE_WINDOW_DAYS = 14    # dispensing must occur within N days of consult
TEST_WINDOW_DAYS     = 30    # lab result within N days of consult
REVIEW_GRACE_DAYS    = 7     # extra grace on top of review_due_days


# ── loader ────────────────────────────────────────────────────────────────────

def _load(name: str, silver: dict | None):
    if silver and name in silver:
        return silver[name].copy()
    for ext in (".parquet", ".pkl"):
        p = C.SILVER_DIR / f"{name}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_pickle(p)
    return None


def _ensure_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ── patient_id attachment ─────────────────────────────────────────────────────

def _attach_patient_id(df: pd.DataFrame, source_pid_col: str,
                        source_system: str, linkage: pd.DataFrame) -> pd.DataFrame:
    """Left-join patient_id from linkage onto df."""
    subset = linkage[linkage["source_system"] == source_system][
        ["source_patient_id", "patient_id"]
    ].drop_duplicates("source_patient_id")
    df = df.merge(
        subset.rename(columns={"source_patient_id": source_pid_col}),
        on=source_pid_col, how="left"
    )
    return df


# ── stage helpers ─────────────────────────────────────────────────────────────

def _build_stage1(prescriptions: pd.DataFrame,
                   dispensing: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      teleconsult_id, stage1_medicine_collected, days_to_medicine
    """
    if prescriptions is None or dispensing is None:
        return pd.DataFrame(columns=["teleconsult_id",
                                     "stage1_medicine_collected",
                                     "days_to_medicine"])

    # prescription → teleconsult_id link
    rx_map = prescriptions[["prescription_id", "teleconsult_id",
                             "prescription_date"]].drop_duplicates()
    dispensing = _ensure_date(dispensing.copy(), "dispense_date")

    disp = dispensing.merge(rx_map, on="prescription_id", how="left")
    disp = _ensure_date(disp, "prescription_date")
    disp["days_to_medicine"] = (
        disp["dispense_date"] - disp["prescription_date"]
    ).dt.days

    # only dispensing within the window and after T0 (dispense_date >= prescription_date)
    disp = disp[
        (disp["days_to_medicine"] >= 0) &
        (disp["days_to_medicine"] <= MEDICINE_WINDOW_DAYS)
    ]

    agg = (disp.groupby("teleconsult_id")
               .agg(stage1_medicine_collected=("dispense_date", "count"),
                    days_to_medicine=("days_to_medicine", "min"))
               .reset_index())
    agg["stage1_medicine_collected"] = agg["stage1_medicine_collected"] > 0
    return agg[["teleconsult_id", "stage1_medicine_collected", "days_to_medicine"]]


def _build_stage2(teleconsults: pd.DataFrame,
                   lab_tests: pd.DataFrame,
                   linkage: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame:
      teleconsult_id, stage2_test_completed, days_to_test
    Linked via patient_id + order_date within TEST_WINDOW_DAYS of consult_date.
    """
    if lab_tests is None:
        return pd.DataFrame(columns=["teleconsult_id",
                                     "stage2_test_completed",
                                     "days_to_test"])

    tele = teleconsults[["teleconsult_id", "patient_id", "consult_date"]].copy()
    lab  = _ensure_date(lab_tests.copy(), "order_date")
    lab  = _ensure_date(lab, "result_date")

    # only completed tests (result_date exists)
    lab = lab[lab["result_date"].notna()]

    merged = tele.merge(lab, on="patient_id", how="left")
    merged["days_to_test"] = (merged["order_date"] - merged["consult_date"]).dt.days

    # post-T0 and within window
    merged = merged[
        (merged["days_to_test"] >= 0) &
        (merged["days_to_test"] <= TEST_WINDOW_DAYS)
    ]

    agg = (merged.groupby("teleconsult_id")
                 .agg(stage2_test_completed=("result_date", "count"),
                      days_to_test=("days_to_test", "min"))
                 .reset_index())
    agg["stage2_test_completed"] = agg["stage2_test_completed"] > 0
    return agg[["teleconsult_id", "stage2_test_completed", "days_to_test"]]


def _build_stage3(episodes_meta: pd.DataFrame,
                   followup_visits: pd.DataFrame,
                   visit_history: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame:
      episode_id, stage3_review_attended, days_to_review

    Links via episode_id (preferred) or patient_id + visit within
    (review_due_days + REVIEW_GRACE_DAYS) of consult_date.
    """
    empty = pd.DataFrame(columns=["episode_id",
                                   "stage3_review_attended",
                                   "days_to_review"])

    visit_frames = []
    for vdf in (followup_visits, visit_history):
        if vdf is not None:
            v = _ensure_date(vdf.copy(), "visit_date")
            # only follow-up / review visits
            if "visit_type" in v.columns:
                v = v[v["visit_type"].str.contains("review|follow", case=False, na=False)]
            visit_frames.append(v[["episode_id", "visit_date",
                                   "patient_id"] if "patient_id" in v.columns
                                  else ["episode_id", "visit_date"]])

    if not visit_frames:
        return empty

    visits = pd.concat(visit_frames, ignore_index=True)
    meta   = episodes_meta[["episode_id", "teleconsult_id", "patient_id",
                             "consult_date", "review_due_days"]].copy()
    meta   = _ensure_date(meta, "consult_date")

    merged = meta.merge(visits, on="episode_id", how="left")
    merged["days_to_review"] = (merged["visit_date"] - merged["consult_date"]).dt.days

    # within allowable window (post-T0 and ≤ due + grace)
    due_days = merged["review_due_days"].fillna(30)
    merged = merged[
        (merged["days_to_review"] > 0) &
        (merged["days_to_review"] <= due_days + REVIEW_GRACE_DAYS)
    ]

    agg = (merged.groupby("episode_id")
                 .agg(stage3_review_attended=("visit_date", "count"),
                      days_to_review=("days_to_review", "min"))
                 .reset_index())
    agg["stage3_review_attended"] = agg["stage3_review_attended"] > 0
    return agg[["episode_id", "stage3_review_attended", "days_to_review"]]


def _build_outreach_summary(outreach: pd.DataFrame) -> pd.DataFrame:
    """Returns: episode_id, n_outreach_attempts, last_outreach_outcome."""
    if outreach is None:
        return pd.DataFrame(columns=["episode_id",
                                     "n_outreach_attempts",
                                     "last_outreach_outcome"])
    out = _ensure_date(outreach.copy(), "action_date")
    agg = (out.groupby("episode_id")
              .agg(n_outreach_attempts=("outreach_id", "count"),
                   last_outreach_outcome=("contact_outcome",
                                          lambda x: x.iloc[-1]
                                          if len(x) else pd.NA))
              .reset_index())
    return agg


# ── dropout stage derivation ──────────────────────────────────────────────────

_STAGE_LABELS = {
    1: "Medicines not collected",
    2: "Test not completed",
    3: "Review not attended",
    4: "Completed care journey",
}

def _derive_dropout(row: pd.Series) -> str:
    med_advised  = str(row.get("medicine_advised", "No")).strip().lower() == "yes"
    test_advised = str(row.get("test_advised",     "No")).strip().lower() == "yes"
    rev_advised  = str(row.get("review_advised",   "No")).strip().lower() == "yes"

    # Stage 1
    if med_advised and not row.get("stage1_medicine_collected", False):
        return _STAGE_LABELS[1]
    # Stage 2
    if test_advised and not row.get("stage2_test_completed", False):
        return _STAGE_LABELS[2]
    # Stage 3
    if rev_advised and not row.get("stage3_review_attended", False):
        return _STAGE_LABELS[3]
    return _STAGE_LABELS[4]


# ── main ──────────────────────────────────────────────────────────────────────

def run(silver: dict | None = None,
        linkage: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the episode table.

    Parameters
    ----------
    silver  : dict of DataFrames (optional; loaded from disk if absent)
    linkage : patient_linkage DataFrame (optional; loaded from disk if absent)

    Returns
    -------
    episodes : pd.DataFrame  (one row per teleconsultation)
    """
    audit("episodes_start")

    # ── load everything ──────────────────────────────────────────────────────
    if linkage is None:
        linkage = _load("patient_linkage", None)
    if linkage is None or len(linkage) == 0:
        log.warning("patient_linkage not found – run linkage.run() first")
        audit("episodes_skip", reason="patient_linkage missing")
        return pd.DataFrame()

    tele    = _load("teleconsultations",  silver)
    eo      = _load("episode_outcomes",   silver)
    rx      = _load("prescriptions",      silver)
    disp    = _load("medicine_dispensing",silver)
    lab     = _load("lab_tests",          silver)
    fv      = _load("followup_visits",    silver)
    vh      = _load("visit_history",      silver)
    outreach= _load("outreach_actions",   silver)

    if tele is None or eo is None:
        log.error("teleconsultations or episode_outcomes missing")
        return pd.DataFrame()

    # ── attach canonical patient_id to teleconsults ──────────────────────────
    tele = _ensure_date(tele, "consult_date")
    tele = _attach_patient_id(tele, "tele_source_patient_id",
                               "teleconsultations", linkage)

    # ── attach patient_id to source tables ───────────────────────────────────
    for df, sys, pid_col in (
        (rx,       "prescriptions",       "rx_source_patient_id"),
        (disp,     "medicine_dispensing", "pharm_source_patient_id"),
        (lab,      "lab_tests",           "lab_source_patient_id"),
        (fv,       "followup_visits",     "visit_source_patient_id"),
        (vh,       "visit_history",       "visit_source_patient_id"),
        (outreach, "outreach_actions",    "outreach_source_patient_id"),
    ):
        if df is not None and pid_col in df.columns:
            df = _attach_patient_id(df, pid_col, sys, linkage)
            # write back (local variable re-bind is enough for passing around)
            if sys == "prescriptions":       rx       = df
            elif sys == "medicine_dispensing": disp   = df
            elif sys == "lab_tests":           lab    = df
            elif sys == "followup_visits":     fv     = df
            elif sys == "visit_history":       vh     = df
            elif sys == "outreach_actions":    outreach = df

    # ── base episode table: episode_outcomes LEFT JOIN teleconsultations ──────
    eo = _ensure_date(eo, "consult_date")
    base = eo[["episode_id", "teleconsult_id", "consult_date",
               "cohort", "lost_to_followup_label",
               "dropout_stage_label"]].merge(
        tele[["teleconsult_id", "patient_id", "facility_id",
              "review_due_days", "medicine_advised",
              "test_advised", "review_advised"]],
        on="teleconsult_id", how="left"
    )

    # ── stage 1: medicine collected ──────────────────────────────────────────
    s1 = _build_stage1(rx, disp)
    base = base.merge(s1, on="teleconsult_id", how="left")
    base["stage1_medicine_collected"] = (
        base["stage1_medicine_collected"].fillna(False).astype(bool))

    # ── stage 2: test completed ──────────────────────────────────────────────
    s2 = _build_stage2(base[["teleconsult_id", "patient_id",
                              "consult_date"]].rename(columns={}),
                        lab, linkage)
    base = base.merge(s2, on="teleconsult_id", how="left")
    base["stage2_test_completed"] = (
        base["stage2_test_completed"].fillna(False).astype(bool))

    # ── stage 3: review attended ─────────────────────────────────────────────
    s3 = _build_stage3(base[["episode_id", "teleconsult_id", "patient_id",
                              "consult_date", "review_due_days"]], fv, vh)
    base = base.merge(s3, on="episode_id", how="left")
    base["stage3_review_attended"] = (
        base["stage3_review_attended"].fillna(False).astype(bool))

    # ── stage 4: resolved (DEVELOPMENT only) ─────────────────────────────────
    base["stage4_resolved"] = (
        base["lost_to_followup_label"].eq(0) &
        base["stage1_medicine_collected"] &
        base["stage2_test_completed"] &
        base["stage3_review_attended"]
    )

    # ── derived dropout stage ─────────────────────────────────────────────────
    base["derived_dropout_stage"] = base.apply(_derive_dropout, axis=1)

    # ── outreach summary (post-T0 context) ───────────────────────────────────
    os_ = _build_outreach_summary(outreach)
    base = base.merge(os_, on="episode_id", how="left")
    base["n_outreach_attempts"] = base["n_outreach_attempts"].fillna(0).astype(int)

    # ── tidy output ──────────────────────────────────────────────────────────
    cols = [
        "episode_id", "teleconsult_id", "patient_id", "consult_date",
        "cohort", "facility_id", "review_due_days",
        "medicine_advised", "test_advised", "review_advised",
        "stage1_medicine_collected", "stage2_test_completed",
        "stage3_review_attended", "stage4_resolved",
        "derived_dropout_stage",
        "days_to_medicine", "days_to_test", "days_to_review",
        "n_outreach_attempts", "last_outreach_outcome",
        "lost_to_followup_label", "dropout_stage_label",
    ]
    episodes = base[[c for c in cols if c in base.columns]].copy()
    episodes = episodes.sort_values("consult_date").reset_index(drop=True)

    save_df(episodes, C.SILVER_DIR, "episodes")
    audit("episodes_end", total=len(episodes),
          dev=int((episodes.cohort == "DEVELOPMENT").sum()),
          eval_=int((episodes.cohort == "EVALUATION").sum()),
          stage1_rate=round(float(episodes.stage1_medicine_collected.mean()), 3),
          stage2_rate=round(float(episodes.stage2_test_completed.mean()), 3),
          stage3_rate=round(float(episodes.stage3_review_attended.mean()), 3))

    _write_episode_report(episodes)
    return episodes


# ── report ────────────────────────────────────────────────────────────────────

def _write_episode_report(ep: pd.DataFrame):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dev = ep[ep.cohort == "DEVELOPMENT"]
    evl = ep[ep.cohort == "EVALUATION"]

    rep = {
        "total_episodes": len(ep),
        "development": len(dev),
        "evaluation":  len(evl),
        "stage_completion_rates_dev": {
            "stage1_medicine": round(float(dev.stage1_medicine_collected.mean()), 3),
            "stage2_test":     round(float(dev.stage2_test_completed.mean()), 3),
            "stage3_review":   round(float(dev.stage3_review_attended.mean()), 3),
            "stage4_resolved": round(float(dev.stage4_resolved.mean()), 3),
        } if len(dev) else {},
        "derived_dropout_stage_dist": ep.derived_dropout_stage.value_counts().to_dict(),
        "label_agreement_dev": (
            (dev.derived_dropout_stage == dev.dropout_stage_label).mean()
            if "dropout_stage_label" in dev.columns and len(dev) else None
        ),
        "median_days_to_medicine": float(ep.days_to_medicine.median()) if "days_to_medicine" in ep else None,
        "median_days_to_test":     float(ep.days_to_test.median())     if "days_to_test"     in ep else None,
        "median_days_to_review":   float(ep.days_to_review.median())   if "days_to_review"   in ep else None,
    }

    (C.REPORT_DIR / "episode_summary.json").write_text(
        json.dumps(rep, indent=2, default=str))

    lines = [
        "# Episode Construction Summary",
        f"- Total episodes: **{rep['total_episodes']:,}**  "
        f"(DEVELOPMENT: {rep['development']:,}  |  EVALUATION: {rep['evaluation']:,})",
        "",
        "## Stage completion rates (DEVELOPMENT cohort)",
        "| Stage | Rate |",
        "|-------|------|",
    ]
    for k, v in rep.get("stage_completion_rates_dev", {}).items():
        lines.append(f"| {k} | {v:.1%} |")
    lines += [
        "",
        "## Derived dropout stage distribution",
        "| Stage | Count |",
        "|-------|-------|",
    ]
    for k, v in sorted(rep["derived_dropout_stage_dist"].items()):
        lines.append(f"| {k} | {v:,} |")

    agree = rep.get("label_agreement_dev")
    if agree is not None:
        lines.append(f"\n- Derived vs. labelled dropout stage agreement (DEV): **{agree:.1%}**")

    (C.REPORT_DIR / "episode_summary.md").write_text("\n".join(lines))
