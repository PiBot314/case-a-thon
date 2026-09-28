"""Ingestion: raw CSV -> bronze (typed) -> silver (normalised, minimum-necessary) + unified source_identities."""
import pandas as pd
from . import config as C
from .io_utils import save_df, audit
from .normalize import norm_name, norm_phone, norm_village, norm_gender

def find_csv(name):
    for cand in (C.RAW_DIR / f"{name}.csv", C.RAW_DIR / f"preview_{name}.csv"):
        if cand.exists(): return cand
    return None

def read_raw(name):
    p = find_csv(name)
    if p is None: return None, None
    df = pd.read_csv(p, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df, p

def type_columns(df, name, spec):
    for c in spec.get("dates", []):
        if c in df: df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in C.NUMERIC_HINTS.get(name, []):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_silver(df, name, spec):
    df = df.copy()
    if spec["kind"] == "source":
        df["name_norm"] = df["patient_name"].map(norm_name)
        df["phone_norm"] = df["mobile"].map(norm_phone) if "mobile" in df else pd.NA
        df["phone_last4"] = df["phone_norm"].str[-4:]
        df["phone_valid"] = df["phone_norm"].notna()
        df["village_norm"] = df["village"].map(norm_village)
        if "gender" in df: df["gender"] = df["gender"].map(norm_gender)
        df = df.drop(columns=[c for c in ("mobile",) if c in df])   # raw phone stays in bronze only
    if name == "patient_360_reference":
        df["name_norm"] = df["canonical_name"].map(norm_name)
        df["phone_last4"] = df["masked_mobile"].str[-4:]
        df["village_norm"] = df["village"].map(norm_village)
        df["gender"] = df["gender"].map(norm_gender)
    if name == "geography_reference":
        df["village_norm"] = df["village"].map(norm_village)
    if name == "episode_outcomes":
        df["is_dev"] = df["cohort"].eq("DEVELOPMENT")
    df["_post_t0_table"] = bool(spec["post_t0"])
    df["_source_table"] = name
    return df

def _mode(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else pd.NA

def build_source_identities(silver):
    """One row per (source_system, source_patient_id): linkage input."""
    frames = []
    for name, spec in C.TABLES.items():
        if spec["kind"] != "source" or name not in silver: continue
        d = silver[name].rename(columns={spec["pid"]: "source_patient_id"})
        keep = [c for c in ("name_norm","phone_norm","village_norm","gender","age","block","district") if c in d]
        g = d.groupby("source_patient_id", sort=False)
        out = g[keep].agg(_mode)
        out["n_records"] = g.size()
        out.insert(0, "source_system", name)
        frames.append(out.reset_index())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def run():
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit("ingest_start", raw_dir=str(C.RAW_DIR), note=C.PREDICTION_TIME_NOTE)
    silver, status = {}, {}
    for name, spec in C.TABLES.items():
        df, p = read_raw(name)
        if df is None:
            status[name] = {"loaded": False}; audit("table_missing", table=name); continue
        rows_raw = len(df)
        typed = type_columns(df, name, spec)
        bp = save_df(typed, C.BRONZE_DIR, name)
        sv = to_silver(typed, name, spec)
        sp = save_df(sv, C.SILVER_DIR, name)
        silver[name] = sv
        status[name] = {"loaded": True, "rows": rows_raw, "source_file": p.name,
                        "bronze": bp.name, "silver": sp.name, "bronze_bytes": bp.stat().st_size}
        audit("table_ingested", table=name, rows=rows_raw, file=p.name)
    ident = build_source_identities(silver)
    if len(ident): save_df(ident, C.SILVER_DIR, "source_identities"); audit("identities_built", rows=len(ident))
    audit("ingest_end", tables_loaded=sum(v["loaded"] for v in status.values()))
    return silver, ident, status
