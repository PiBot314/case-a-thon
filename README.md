# Overview

The system is a "Follow-Up Assurance Engine". It ensures that a patient doesn't drop out partway through a treatment by linking each teleconsultation to the medicine pickup, test, and review visit that should follow it. 
It then ranks open _episodes_ by dropout risk so ASHAs and CHOs know whom to contact each day.

The prediction unit is the episode, one teleconsultation and its downstream care journey. There are three parts:
- Link the 8 fragmented source systems to one patient identity.
- Track each episode through four stages: medicines collected → tests completed → review visit → resolved. Where it first stalls gives the dropout stage.
- Predict and act. A model scores each episode, and a short prioritised call list goes to the health worker.

## Architecture

### Data Ingestion, and Basics
#### 1. Loading the data
- Load all source tables and reference tables (patient_360_reference, facilities, geography).
- Use role-based access, minimum-necessary fields, and masked phones. Keep an audit log and ABDM-aligned consent flags (privacy maxxing)

#### 2. Constructing the Patient from the scattered data
- Every system has its own source_patient_id and there is no crosswalk, so the linkage is built from noisy name, phone, age, village, and gender.
- Normalize the fields: transliterate and lowercase names, strip phone formatting, and standardize village spellings against geography_reference.
- Use blocking on district, block, and phone last-4. Score candidate pairs with fuzzy name similarity (Jaro-Winkler or phonetic), phone match, age within ±2 years, and village match. (Essentially check if a person is the same on multiple fields)
- Accept high-confidence links automatically and send borderline ones to a review queue (would only happen with human review, otherwise, directly best match taken). Report precision and recall on a hand-checked sample.

The output is a source_id → patient_id map for every table.

#### 3. Constructing an Episode (Patient Journey)
- Prioritize "teleconsultations" and "episode_outcomes"
- Join prescriptions to dispensing via prescription_id, lab orders to results, and visits and outreach via episode_id or patient plus time window.
- Derive stage completion (is each step over) and time-to-event. Stage outcomes are medicine collected, test completed, and review attended, judged against review_due_days.

#### 4. Feature store (ML Modelling Stuff) (leakage-safe)
- Prediction time = end of the teleconsultation (T0). Only data known at T0 is allowed. This includes consult attributes, patient history before T0, distance, connectivity, facility and village context, and stock status for the prior month.
- Exclude anything dated after T0 from the features. That covers dispensing, lab, visit, and outreach events for the same episode, and consult_status if it is finalised later.
- Option for a second model: a "day 7 / day 14 rescore" that uses early signals such as an unfilled prescription or a failed outreach. It must be timestamped and reported separately.

#### 5. Modelling (Finally)
- Model 1 predicts lost_to_followup_label with gradient boosting (XGBoost because of familiarity), with logistic regression as an explainable baseline.
- Model 2 predicts dropout_stage_labeel, which tells the health worker what kind of help is needed.
- Validation: train on the development cohort (consults on or before 30-Jun-2026). Validate with a time-based split, with folds grouped by patient and household to avoid leakage. Track AUC-PR, recall at the top-k that a health worker can realistically call, and calibration. 
i.e Train-Test split based on time
- Add SHAP explanations and fairness checks across gender, age, language, and vulnerability group.
- Score the evaluation cohort (1-Jul to 15-Aug 2026) and export predictions.

#### 6. Action layer
- A ranked worklist per ASHA/CHO, sized to their capacity. Each row shows the patient, village, reason for the flag ("medicine stock-out at the facility", "no test done"), and a suggested action.
- The escalation ladder runs from SMS or IVR reminder to ASHA call, then home visit, then CHO or village health day.
- Low-end phone design: a lightweight, offline-first view that syncs when connectivity returns, with local-language text.

#### 7. Measurement/Evaluation
- Effectiveness: compare the lost-to-follow-up rate and the stage completion rate between targeted and usual-care groups, using a stepped-wedge or block-randomised pilot.
- Feasibility: contact success rate, calls per worker per day, and alert acceptance.
- Cost: cost per follow-up completed, and cost per lost-to-follow-up case averted.