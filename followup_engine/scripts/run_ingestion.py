#!/usr/bin/env python
"""Usage: python scripts/run_ingestion.py [--raw-dir PATH]   (default data/raw; accepts preview_*.csv names)"""
import sys, argparse, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument("--raw-dir"); a = ap.parse_args()
if a.raw_dir: os.environ["FUE_RAW_DIR"] = a.raw_dir
from src import ingest, quality
silver, ident, status = ingest.run()
rep = quality.build_report(silver, ident, status)
quality.write_reports(rep)
print(f"[{rep['status']}] {sum(v['loaded'] for v in status.values())}/{len(status)} tables | storage={rep['storage_format']}")
for i in rep["issues"]: print(" -", i)
print("Reports: reports/ingestion_summary.md, reports/ingestion_summary.json, reports/audit_log.jsonl")
