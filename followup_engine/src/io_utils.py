"""Storage helpers: parquet when an engine exists, else pickle fallback. Audit logging."""
import json, time, getpass
from pathlib import Path
import pandas as pd
from . import config

def _has_parquet():
    for m in ("pyarrow", "fastparquet"):
        try: __import__(m); return True
        except ImportError: pass
    return False
HAS_PARQUET = _has_parquet()

def save_df(df: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        p = directory / f"{name}.parquet"; df.to_parquet(p, index=False)
    else:
        p = directory / f"{name}.pkl"; df.to_pickle(p)
    return p

def load_df(directory: Path, name: str) -> pd.DataFrame:
    for ext, fn in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        p = directory / f"{name}{ext}"
        if p.exists(): return fn(p)
    raise FileNotFoundError(f"{name} not found in {directory}")

def audit(event: str, **kw):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "user": getpass.getuser(), "event": event, **kw}
    with open(config.AUDIT_LOG, "a", encoding="utf-8") as f: f.write(json.dumps(rec, default=str) + "\n")
