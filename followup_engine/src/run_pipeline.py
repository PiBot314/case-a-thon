#!/usr/bin/env python
"""
run_pipeline.py  –  Full pipeline runner
=========================================
Stages run in order:
  1. Ingest       (raw CSV → bronze → silver + source_identities)
  2. Linkage      (source_identities → patient_linkage)
  3. Episodes     (patient_linkage + silver → episodes)

Usage
-----
  python scripts/run_pipeline.py [--raw-dir PATH] [--stage STAGE]

--raw-dir PATH   override the default data/raw directory
--stage  STAGE   run only from this stage onwards:
                   ingest   (default, runs all three)
                   linkage  (skips re-ingest, loads silver from disk)
                   episodes (skips ingest + linkage, loads from disk)
"""
import sys, argparse, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(description="Follow-Up Assurance Engine – full pipeline")
ap.add_argument("--raw-dir", help="path to raw CSV directory (default: data/raw)")
ap.add_argument("--stage",   default="ingest",
                choices=["ingest", "linkage", "episodes"],
                help="start from this stage (all downstream stages also run)")
args = ap.parse_args()

if args.raw_dir:
    os.environ["FUE_RAW_DIR"] = args.raw_dir

from src import ingest, quality, linkage, episodes

silver, ident = {}, None

# ── Stage 1: Ingest ───────────────────────────────────────────────────────────
if args.stage == "ingest":
    print("=" * 60)
    print("STAGE 1 — Data ingestion")
    print("=" * 60)
    silver, ident, status = ingest.run()
    rep = quality.build_report(silver, ident, status)
    quality.write_reports(rep)
    print(f"[{rep['status']}] "
          f"{sum(v['loaded'] for v in status.values())}/{len(status)} tables | "
          f"storage={rep['storage_format']}")
    for issue in rep["issues"]:
        print(" -", issue)
    print("→ Reports: reports/ingestion_summary.md  reports/ingestion_summary.json")
else:
    print("(skipping ingest – loading silver from disk)")

# ── Stage 2: Linkage ──────────────────────────────────────────────────────────
if args.stage in ("ingest", "linkage"):
    print()
    print("=" * 60)
    print("STAGE 2 — Patient identity linkage")
    print("=" * 60)
    lnk = linkage.run(silver if silver else None)
    if len(lnk):
        n_canon  = lnk["patient_id"].nunique()
        n_source = len(lnk)
        by_type  = lnk["link_type"].value_counts().to_dict()
        print(f"Resolved {n_canon:,} canonical patients from {n_source:,} source rows")
        for lt, cnt in sorted(by_type.items()):
            print(f"  {lt}: {cnt:,}")
        print(f"  review_flagged: {lnk.review_flag.sum():,}")
        print("→ Reports: reports/linkage_summary.md  reports/linkage_summary.json")
    else:
        print("[WARNING] linkage returned empty – check source_identities")
        sys.exit(1)
else:
    print("(skipping linkage – will load patient_linkage from disk)")
    lnk = None

# ── Stage 3: Episodes ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STAGE 3 — Episode construction")
print("=" * 60)
ep = episodes.run(silver if silver else None, linkage=lnk)
if len(ep):
    dev = ep[ep.cohort == "DEVELOPMENT"]
    evl = ep[ep.cohort == "EVALUATION"]
    print(f"Built {len(ep):,} episodes  "
          f"(DEV: {len(dev):,}  |  EVAL: {len(evl):,})")
    if len(dev):
        print(f"  Stage-1 medicine rate : {dev.stage1_medicine_collected.mean():.1%}")
        print(f"  Stage-2 test rate     : {dev.stage2_test_completed.mean():.1%}")
        print(f"  Stage-3 review rate   : {dev.stage3_review_attended.mean():.1%}")
    print("→ Reports: reports/episode_summary.md  reports/episode_summary.json")
else:
    print("[WARNING] episodes returned empty")

print()
print("Pipeline complete.")
print("Next stages: T0 feature store → XGBoost + LR → action queue → submission CSVs")
