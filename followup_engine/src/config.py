"""Central configuration: paths + declarative table specs for the 8 source systems + references."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("FUE_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = Path(os.environ.get("FUE_RAW_DIR", ROOT / "data" / "raw"))   # drop the CSVs here
BRONZE_DIR = ROOT / "data" / "bronze"     # typed, otherwise untouched (RESTRICTED: contains raw phones)
SILVER_DIR = ROOT / "data" / "silver"     # cleaned + normalised, raw phone dropped (minimum-necessary)
REPORT_DIR = ROOT / "reports"
AUDIT_LOG = REPORT_DIR / "audit_log.jsonl"

PREDICTION_TIME_NOTE = "T0 = end of teleconsultation. Tables flagged post_t0=True must NOT feed T0 features."
DEV_END = "2026-06-30"
EVAL_START, EVAL_END = "2026-07-01", "2026-08-15"

TABLES = {
 "patient_360_reference": dict(kind="reference", pk="patient_id", dates=[], post_t0=False,
    required=["patient_id","canonical_name","village_id","household_id"]),
 "geography_reference": dict(kind="reference", pk="village_id", dates=[], post_t0=False,
    required=["village_id","village","district","block"]),
 "facility_reference": dict(kind="reference", pk="facility_id", dates=[], post_t0=False,
    required=["facility_id","facility_name"]),
 "medicine_stock_status": dict(kind="reference", pk="stock_record_id", dates=["snapshot_month"], post_t0=False,
    required=["stock_record_id","facility_id","snapshot_month"],
    fk={"facility_id":("facility_reference","facility_id")},
    note="Use only snapshots from months BEFORE the consult month (prior-month stock)."),
 "teleconsultations": dict(kind="source", pk="teleconsult_id", pid="tele_source_patient_id", dates=["consult_date"], post_t0=False,
    required=["teleconsult_id","tele_source_patient_id","consult_date","facility_id"],
    fk={"facility_id":("facility_reference","facility_id")},
    note="consult_status may be finalised after T0 - treat with care."),
 "ncd_screening": dict(kind="source", pk="ncd_record_id", pid="ncd_source_patient_id", dates=["screening_date","next_followup_due"],
    post_t0=False, required=["ncd_record_id","ncd_source_patient_id","screening_date"],
    note="History: only records dated before T0 are usable."),
 "prescriptions": dict(kind="source", pk="prescription_line_id", pid="rx_source_patient_id", dates=["prescription_date"], post_t0=False,
    required=["prescription_line_id","prescription_id","teleconsult_id"],
    fk={"teleconsult_id":("teleconsultations","teleconsult_id"),"facility_id":("facility_reference","facility_id")},
    note="Issued at consult time; existence of a prescription is known at T0, fills are not."),
 "medicine_dispensing": dict(kind="source", pk="dispense_id", pid="pharm_source_patient_id", dates=["dispense_date"], post_t0=True,
    required=["dispense_id","pharm_source_patient_id","prescription_id","dispense_date"],
    fk={"prescription_id":("prescriptions","prescription_id"),"facility_id":("facility_reference","facility_id")}),
 "lab_tests": dict(kind="source", pk="lab_record_id", pid="lab_source_patient_id", dates=["order_date","sample_date","result_date"],
    post_t0=True, required=["lab_record_id","lab_source_patient_id","order_date"],
    fk={"facility_id":("facility_reference","facility_id")}),
 "visit_history": dict(kind="source", pk="visit_id", pid="visit_source_patient_id", dates=["visit_date"], post_t0=True,
    required=["visit_id","visit_source_patient_id","visit_date"],
    fk={"episode_id":("episode_outcomes","episode_id"),"facility_id":("facility_reference","facility_id")},
    note="Pre-T0 rows are valid history; rows on/after the episode consult date are outcome-side."),
 "followup_visits": dict(kind="source", pk="visit_id", pid="visit_source_patient_id", dates=["visit_date"], post_t0=True,
    required=["visit_id","visit_source_patient_id","visit_date"],
    fk={"episode_id":("episode_outcomes","episode_id"),"facility_id":("facility_reference","facility_id")}),
 "outreach_actions": dict(kind="source", pk="outreach_id", pid="outreach_source_patient_id", dates=["action_date"], post_t0=True,
    required=["outreach_id","outreach_source_patient_id","action_date","episode_id"],
    fk={"episode_id":("episode_outcomes","episode_id"),"facility_id":("facility_reference","facility_id")}),
 "episode_outcomes": dict(kind="label", pk="episode_id", dates=["consult_date"], post_t0=False,
    required=["episode_id","teleconsult_id","consult_date","cohort"],
    fk={"teleconsult_id":("teleconsultations","teleconsult_id")},
    note="Labels are DEVELOPMENT only; blank for EVALUATION by design."),
}
NUMERIC_HINTS = {
 "teleconsultations":["age","duration_min","review_due_days","distance_to_facility_km"],
 "ncd_screening":["age","systolic_bp","diastolic_bp","random_glucose_mg_dl","bmi"],
 "prescriptions":["days_prescribed"],
 "medicine_dispensing":["prescribed_qty_proxy","dispensed_qty","partial_fill","stockout_flag"],
 "medicine_stock_status":["opening_stock","received_qty","dispensed_qty_month","closing_stock","stockout_flag_month","stockout_days"],
 "facility_reference":["latitude","longitude","cho_count","asha_linked_count"],
 "geography_reference":["latitude","longitude","population"],
 "patient_360_reference":["age_as_of_2026"],
 "episode_outcomes":["lost_to_followup_label"],
}
