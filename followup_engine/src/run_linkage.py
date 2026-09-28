#!/usr/bin/env python
"""Stage 2 only: build source_id -> patient_id map from silver/source_identities.
Usage: python scripts/run_linkage.py [--benchmark]
  --benchmark   also run the synthetic known-truth benchmark and add it to reports/linkage_summary.*
After hand-labelling reports/linkage_hand_check_template.csv (is_correct Y/N, true_patient_id if known),
save it as reports/linkage_hand_check_labelled.csv and re-run to get precision/recall."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument("--benchmark", action="store_true"); a = ap.parse_args()
from src import linkage, linkage_eval
lk = linkage.run()
if not len(lk): sys.exit("source_identities / patient_360 / geography silver tables missing - run scripts/run_ingestion.py first")
print(f"{len(lk):,} source ids -> {lk.patient_id.nunique():,} patients")
print(lk.link_type.value_counts().to_string())
if a.benchmark:
    bench, _ = linkage_eval.benchmark()
    linkage_eval.write_report(lk, linkage_eval.diagnostics(lk, __import__('src.io_utils', fromlist=['x']).load_df(linkage.C.SILVER_DIR, 'source_identities')), bench=bench)
    print("benchmark:", bench)
print("Outputs: data/silver/patient_linkage, outputs/submission_linkage.csv, reports/linkage_*.{md,json,csv}")
