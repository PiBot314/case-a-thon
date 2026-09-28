"""Data-quality + leakage-guard checks and report writers (JSON + Markdown)."""
import json
import pandas as pd
from . import config as C
from .io_utils import HAS_PARQUET

def table_profile(df, name, spec):
    p = {"rows": len(df), "cols": len(df.columns)}
    p["missing_required_cols"] = [c for c in spec["required"] if c not in df.columns]
    pk = spec["pk"]
    p["pk_duplicates"] = int(df[pk].duplicated().sum()) if pk in df else None
    p["null_pct"] = {c: round(float(v), 1) for c, v in (df.isna().mean() * 100).items() if v > 0 and not c.startswith("_")}
    dr = {}
    for c in spec.get("dates", []):
        if c in df and df[c].notna().any(): dr[c] = [str(df[c].min().date()), str(df[c].max().date())]
    p["date_ranges"] = dr
    if "phone_valid" in df: p["invalid_or_missing_phone_pct"] = round(float((~df["phone_valid"]).mean() * 100), 1)
    return p

def fk_checks(silver):
    out = []
    for name, spec in C.TABLES.items():
        if name not in silver: continue
        for col, (ot, oc) in spec.get("fk", {}).items():
            if ot not in silver or col not in silver[name]: continue
            s = silver[name][col].dropna(); valid = set(silver[ot][oc].dropna())
            out.append({"child": f"{name}.{col}", "parent": f"{ot}.{oc}", "checked": int(len(s)), "orphans": int((~s.isin(valid)).sum())})
    return out

def village_match(silver):
    if "geography_reference" not in silver: return {}
    geo = set(silver["geography_reference"]["village_norm"].dropna()); out = {}
    for name, spec in C.TABLES.items():
        if spec["kind"] == "source" and name in silver:
            v = silver[name]["village_norm"].dropna()
            out[name] = round(float((~v.isin(geo)).mean() * 100), 1) if len(v) else None
    return out

def cohort_checks(silver):
    if "episode_outcomes" not in silver: return {}
    e = silver["episode_outcomes"]; r = {"cohort_counts": e["cohort"].value_counts().to_dict()}
    dev, ev = e[e.cohort == "DEVELOPMENT"], e[e.cohort == "EVALUATION"]
    r["dev_max_date_ok"] = bool((dev.consult_date <= C.DEV_END).all()) if len(dev) else None
    r["eval_date_window_ok"] = bool(ev.consult_date.between(C.EVAL_START, C.EVAL_END).all()) if len(ev) else None
    r["dev_label_null"] = int(dev.lost_to_followup_label.isna().sum())
    r["eval_label_nonnull"] = int(ev.lost_to_followup_label.notna().sum())
    if len(dev):
        r["dev_lost_rate_pct"] = round(float(dev.lost_to_followup_label.mean() * 100), 1)
        r["dropout_stage_dist"] = dev.dropout_stage_label.value_counts().to_dict()
    return r

def leakage_register():
    return {"prediction_time": C.PREDICTION_TIME_NOTE,
            "post_t0_tables_excluded_from_T0_features": [n for n, s in C.TABLES.items() if s["post_t0"]],
            "handle_with_care": {n: s["note"] for n, s in C.TABLES.items() if s.get("note")}}

def build_report(silver, ident, status):
    rep = {"storage_format": "parquet" if HAS_PARQUET else "pickle (pyarrow not installed - fallback)",
           "tables": {}, "fk_checks": fk_checks(silver), "village_string_mismatch_pct_vs_geography": village_match(silver),
           "cohorts": cohort_checks(silver), "leakage_register": leakage_register()}
    for n, spec in C.TABLES.items():
        rep["tables"][n] = {**status.get(n, {"loaded": False}), **(table_profile(silver[n], n, spec) if n in silver else {})}
    if len(ident):
        rep["source_identities"] = {"total": len(ident), "by_system": ident.source_system.value_counts().to_dict(),
            "missing_phone_pct": round(float(ident.phone_norm.isna().mean() * 100), 1),
            "missing_name_pct": round(float(ident.name_norm.isna().mean() * 100), 1)}
    issues = [f"{n}: missing required cols {t['missing_required_cols']}" for n, t in rep["tables"].items() if t.get("missing_required_cols")]
    issues += [f"{n}: {t['pk_duplicates']} duplicate PKs" for n, t in rep["tables"].items() if t.get("pk_duplicates")]
    issues += [f"FK {f['child']}->{f['parent']}: {f['orphans']} orphans" for f in rep["fk_checks"] if f["orphans"]]
    issues += [f"{n}: file not found" for n, t in rep["tables"].items() if not t.get("loaded")]
    rep["issues"] = issues; rep["status"] = "OK" if not issues else "ATTENTION"
    return rep

def write_reports(rep):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_DIR / "ingestion_summary.json").write_text(json.dumps(rep, indent=2, default=str))
    L = [f"# Ingestion Summary - {rep['status']}", f"Storage: {rep['storage_format']}", "",
         "| table | rows | cols | pk dups | phone bad % | date range |", "|---|---|---|---|---|---|"]
    for n, t in rep["tables"].items():
        dr = "; ".join(f"{k}: {v[0]} to {v[1]}" for k, v in t.get("date_ranges", {}).items())
        L.append(f"| {n} | {t.get('rows','-')} | {t.get('cols','-')} | {t.get('pk_duplicates','-')} | {t.get('invalid_or_missing_phone_pct','-')} | {dr} |")
    L += ["", "## Referential integrity"] + [f"- {f['child']} -> {f['parent']}: {f['orphans']}/{f['checked']} orphans" for f in rep["fk_checks"]]
    L += ["", "## Village strings not matching geography_reference (%, need fuzzy repair)"] + [f"- {k}: {v}%" for k, v in rep["village_string_mismatch_pct_vs_geography"].items()]
    L += ["", "## Cohorts", "```", json.dumps(rep["cohorts"], indent=1, default=str), "```"]
    if "source_identities" in rep: L += ["", "## Source identities (linkage input)", "```", json.dumps(rep["source_identities"], indent=1), "```"]
    L += ["", "## Leakage register", f"- {rep['leakage_register']['prediction_time']}",
          f"- Post-T0 tables: {', '.join(rep['leakage_register']['post_t0_tables_excluded_from_T0_features'])}",
          "", "## Issues"] + ([f"- {i}" for i in rep["issues"]] or ["- none"])
    (C.REPORT_DIR / "ingestion_summary.md").write_text("\n".join(L))
