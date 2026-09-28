# Patient Linkage Summary
- Source identities: **25,586** -> patients: **5,970** (4,917 anchored to patient_360)

| link_type | n |
|---|---|
| anchor_auto | 24,351 |
| singleton | 1,043 |
| anchor_review | 170 |
| cross_source_cluster | 20 |
| cross_source_anchor | 2 |

Share linked to a patient_360 id, by system: followup_visits 91.1%, lab_tests 91.1%, medicine_dispensing 93.1%, ncd_screening 99.9%, outreach_actions 95.8%, prescriptions 93.2%, teleconsultations 99.9%, visit_history 97.0%
- Review-zone links (best match taken, in reports/linkage_review_queue.csv): 170
- Patient x system pairs holding >1 source id (possible wrong merges/dupes): 17
- Villages repaired against geography_reference: 5.5%

## Precision / recall
```
"pending - label reports/linkage_hand_check_template.csv, save as linkage_hand_check_labelled.csv"
```