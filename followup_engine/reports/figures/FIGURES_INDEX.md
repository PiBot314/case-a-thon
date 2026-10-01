# Submission Figures & Visual Assets Index

This directory contains publication-ready figures generated for the **Follow-Up Assurance Engine** submission report and presentation deck.

| Figure | Filename | Description | Key Metric / Insight |
|---|---|---|---|
| **Fig 1** | [`fig1_model_discrimination.png`](./fig1_model_discrimination.png) | Precision-Recall & ROC curves comparing Logistic Regression baseline vs. Primary Model | AP: **0.617**, ROC-AUC: **0.688** on time-grouped CV |
| **Fig 2** | [`fig2_probability_calibration.png`](./fig2_probability_calibration.png) | Reliability diagram showing predicted probabilities vs observed event rates | Low ECE (**~0.00** after isotonic calibration), well-calibrated risk |
| **Fig 3** | [`fig3_worker_capacity_tradeoff.png`](./fig3_worker_capacity_tradeoff.png) | Yield & Efficiency curve vs health worker outreach capacity | **69.4%** precision at top-10% cutoff (**1.49x lift** over random outreach) |
| **Fig 4** | [`fig4_care_journey_funnel.png`](./fig4_care_journey_funnel.png) | 4-Stage patient completion funnel & dropout stage breakdown | Medicine collection rate: **58.6%**, Test completion: **25.6%**, Review rate: **46.6%** |
| **Fig 5** | [`fig5_shap_feature_importance.png`](./fig5_shap_feature_importance.png) | Top 15 risk drivers ranked by mean absolute SHAP value | Regimen complexity, distance to facility, and medicine stock-outs are top drivers |
| **Fig 6** | [`fig6_fairness_parity_audit.png`](./fig6_fairness_parity_audit.png) | Subgroup parity analysis across Gender, Age, Language, and Vulnerability | Flag rate ratio within **0.80x - 1.25x** parity corridor across all demographic slices |
| **Fig 7** | [`fig7_action_queue_allocation.png`](./fig7_action_queue_allocation.png) | Operational worklist breakdown by priority tier, cadre (ASHA/CHO), and trigger reasons | High/Medium risk prioritization with targeted action escalation ladder |
| **Fig 8** | [`fig8_identity_linkage.png`](./fig8_identity_linkage.png) | Linkage type distribution and cross-system linkage rates | **>95%** linkage rate to Patient 360 across all 8 noisy source systems |
| **Fig 9** | [`fig9_master_submission_dashboard.png`](./fig9_master_submission_dashboard.png) | **Master Executive Dashboard**: 6-panel summary figure combining all key system layers | Ideal for 1-page executive summary slide or case competition presentation |

---
*Generated automatically by `scripts/generate_submission_graphs.py`.*
