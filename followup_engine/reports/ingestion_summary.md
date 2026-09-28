# Ingestion Summary - OK
Storage: parquet

| table | rows | cols | pk dups | phone bad % | date range |
|---|---|---|---|---|---|
| patient_360_reference | 5000 | 19 | 0 | - |  |
| geography_reference | 36 | 12 | 0 | - |  |
| facility_reference | 27 | 12 | 0 | - |  |
| medicine_stock_status | 1296 | 13 | 0 | - | snapshot_month: 2026-01-01 to 2026-08-01 |
| teleconsultations | 5516 | 29 | 0 | 7.9 | consult_date: 2026-01-01 to 2026-08-15 |
| ncd_screening | 4748 | 24 | 0 | 8.4 | screening_date: 2025-07-01 to 2026-08-15; next_followup_due: 2025-07-31 to 2026-11-13 |
| prescriptions | 5151 | 20 | 0 | 7.9 | prescription_date: 2026-01-01 to 2026-08-15 |
| medicine_dispensing | 3756 | 20 | 0 | 8.0 | dispense_date: 2026-01-01 to 2026-08-20 |
| lab_tests | 2005 | 19 | 0 | 7.4 | order_date: 2026-01-01 to 2026-08-15; sample_date: 2026-01-02 to 2026-08-22; result_date: 2026-01-03 to 2026-08-24 |
| visit_history | 13061 | 17 | 0 | 8.2 | visit_date: 2025-07-01 to 2026-09-21 |
| followup_visits | 2575 | 17 | 0 | 7.1 | visit_date: 2026-01-08 to 2026-09-21 |
| outreach_actions | 6088 | 18 | 0 | 7.7 | action_date: 2026-01-04 to 2026-09-15 |
| episode_outcomes | 5516 | 9 | 0 | - | consult_date: 2026-01-01 to 2026-08-15 |

## Referential integrity
- medicine_stock_status.facility_id -> facility_reference.facility_id: 0/1296 orphans
- teleconsultations.facility_id -> facility_reference.facility_id: 0/5516 orphans
- prescriptions.teleconsult_id -> teleconsultations.teleconsult_id: 0/5151 orphans
- prescriptions.facility_id -> facility_reference.facility_id: 0/5151 orphans
- medicine_dispensing.prescription_id -> prescriptions.prescription_id: 0/3756 orphans
- medicine_dispensing.facility_id -> facility_reference.facility_id: 0/3756 orphans
- lab_tests.facility_id -> facility_reference.facility_id: 0/2005 orphans
- visit_history.episode_id -> episode_outcomes.episode_id: 0/8091 orphans
- visit_history.facility_id -> facility_reference.facility_id: 0/13061 orphans
- followup_visits.episode_id -> episode_outcomes.episode_id: 0/2575 orphans
- followup_visits.facility_id -> facility_reference.facility_id: 0/2575 orphans
- outreach_actions.episode_id -> episode_outcomes.episode_id: 0/6088 orphans
- outreach_actions.facility_id -> facility_reference.facility_id: 0/6088 orphans
- episode_outcomes.teleconsult_id -> teleconsultations.teleconsult_id: 0/5516 orphans

## Village strings not matching geography_reference (%, need fuzzy repair)
- teleconsultations: 5.1%
- ncd_screening: 5.0%
- prescriptions: 4.8%
- medicine_dispensing: 5.1%
- lab_tests: 5.2%
- visit_history: 5.1%
- followup_visits: 5.1%
- outreach_actions: 4.9%

## Cohorts
```
{
 "cohort_counts": {
  "DEVELOPMENT": 4132,
  "EVALUATION": 1384
 },
 "dev_max_date_ok": true,
 "eval_date_window_ok": true,
 "dev_label_null": 0,
 "eval_label_nonnull": 0,
 "dev_lost_rate_pct": 48.1,
 "dropout_stage_dist": {
  "Completed care journey": 2144,
  "Medicine not collected": 1170,
  "Review not attended": 543,
  "Test not completed": 275
 }
}
```

## Source identities (linkage input)
```
{
 "total": 25586,
 "by_system": {
  "visit_history": 4747,
  "teleconsultations": 4231,
  "ncd_screening": 3589,
  "prescriptions": 3449,
  "medicine_dispensing": 2781,
  "outreach_actions": 2705,
  "followup_visits": 2270,
  "lab_tests": 1814
 },
 "missing_phone_pct": 4.8,
 "missing_name_pct": 0.0
}
```

## Leakage register
- T0 = end of teleconsultation. Tables flagged post_t0=True must NOT feed T0 features.
- Post-T0 tables: medicine_dispensing, lab_tests, visit_history, followup_visits, outreach_actions

## Issues
- none