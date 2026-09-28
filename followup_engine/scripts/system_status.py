#!/usr/bin/env python
"""Quick health check of pipeline state."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.io_utils import HAS_PARQUET
def layer(d):
    fs = sorted(d.glob("*.*")) if d.exists() else []
    return len(fs), sum(f.stat().st_size for f in fs), (time.strftime("%F %T", time.localtime(max(f.stat().st_mtime for f in fs))) if fs else "-")
print(f"parquet engine: {'yes' if HAS_PARQUET else 'NO (pickle fallback) -> pip install pyarrow'}")
print(f"raw CSVs: {len(list(C.RAW_DIR.glob('*.csv')))} in {C.RAW_DIR}")
for n, d in (("bronze", C.BRONZE_DIR), ("silver", C.SILVER_DIR)):
    c, b, t = layer(d); print(f"{n}: {c} files, {b/1024:.0f} KB, last write {t}")
s = C.REPORT_DIR / "ingestion_summary.json"
if s.exists():
    r = json.loads(s.read_text()); print("last ingestion status:", r["status"]); [print(" -", i) for i in r["issues"]]
else: print("no ingestion summary yet -> run scripts/run_ingestion.py")
print("next stages: linkage -> episodes -> features(T0) -> models -> action queue  (not yet built)")
