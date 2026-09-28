# Ingestion pathway
1. `pip install -r requirements.txt`
2. Put the full CSVs in `data/raw/` (or `--raw-dir`).
3. `python scripts/run_ingestion.py` then `python scripts/system_status.py`

Outputs: `data/bronze/`, `data/silver/` (parquet, or .pkl if no engine), `reports/ingestion_summary.md|json`, `reports/audit_log.jsonl`.
Checks: required columns, PK duplicates, nulls, date ranges, FK orphans, village mismatch vs geography, phone validity, cohort/label sanity, leakage register.
Table specs live in `src/config.py`; add a table there and it flows through.
