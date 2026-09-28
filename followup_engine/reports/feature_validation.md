# Feature Store Validation
FAIL: **0**  WARN: **4**  PASS: **26**

| Status | Check | Detail |
|---|---|---|
| PASS | 1a row count == episode_outcomes | 5516 vs 5516 |
| PASS | 1b episode_id unique |  |
| PASS | 1c episode_id set matches outcomes |  |
| PASS | 1d patient_id null share | 0.00% |
| PASS | 2a DEV labels all present |  |
| PASS | 2b EVAL labels all blank |  |
| PASS | 2c DEV dates <= cutoff | 2026-06-30 |
| PASS | 2d EVAL dates inside window |  |
| PASS | 2e development prevalence | lost_to_followup = 0.481, n_dev=4132, n_eval=1384 |
| PASS | 3a no forbidden feature names |  |
| PASS | 3b features disjoint from ids/labels/audit |  |
| PASS | 3c no inf values |  |
| PASS | 4 _audit_stock_snapshot_month strictly < T0 | violations=0, coverage=89.3% |
| PASS | 4 _audit_ptele_last_date strictly < T0 | violations=0, coverage=23.4% |
| PASS | 4 _audit_pncd_last_date strictly < T0 | violations=0, coverage=60.1% |
| PASS | 4 _audit_pvisit_last_date strictly < T0 | violations=0, coverage=66.3% |
| PASS | 4 _audit_pdisp_last_date strictly < T0 | violations=0, coverage=13.2% |
| PASS | 4 _audit_plab_last_date strictly < T0 | violations=0, coverage=8.6% |
| PASS | 5 recompute ptele_n on 300 episodes | mismatches=0 |
| PASS | 5 recompute pvisit_n on 300 episodes | mismatches=0 |
| PASS | 5 recompute pdisp_n on 300 episodes | mismatches=0 |
| PASS | 5 recompute plab_n on 300 episodes | mismatches=0 |
| PASS | 5 recompute pncd_n on 300 episodes | mismatches=0 |
| PASS | 6 max single-feature AUC vs label | n_advised_components=0.664; rx_n_lines=0.621; rx_n_medicines=0.619; rx_n_classes=0.619; distance_to_facility_km=0.600 |
| WARN | 7a features >60% missing | plab_completion_rate=91%, plab_days_since_last=91%, pdisp_partial_rate=87%, pdisp_days_since_last=87%, ptele_days_since_last=77% |
| WARN | 7b constant features | stk_snapshot_lag_months |
| PASS | 7c high-cardinality categoricals (>50) |  |
| WARN | 7d missingness DEV vs EVAL gap >15pp | ptele_days_since_last=21%, pncd_days_since_last=16%, ncd_last_sbp=16%, ncd_last_dbp=16%, ncd_last_glucose=16%, ncd_last_bmi=16%, ncd_last_status=16%, ncd_last_control=16%, ncd_followup_overdue_at_t0=16%, pvisit_days_since_last=18% |
| WARN | 8 numeric drift DEV vs EVAL (KS>0.2) | pncd_days_since_last=0.25; pvisit_n=0.22; ptele_n=0.21; plab_days_since_last=0.20; pdisp_days_since_last=0.19 |
| PASS | 8b categories in EVAL never seen in DEV |  |