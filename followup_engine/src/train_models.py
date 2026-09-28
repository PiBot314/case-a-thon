#!/usr/bin/env python
"""
Train Model 1 (lost_to_followup) and Model 2 (dropout stage) on the T0 feature store.

    python scripts/train_models.py                      # defaults
    python scripts/train_models.py --calibrate --folds 3 --k-frac 0.10 --topk 50
    python scripts/train_models.py --exclude-features gender,vulnerability_group

Validation: expanding-window TIME folds over the DEVELOPMENT cohort (last 50% of dates is
validated in --folds chunks). Training rows whose household appears in the validation
fold are dropped (grouped by household => no patient/household leakage).
Metrics: AUC-PR, ROC-AUC, recall@top-k% and @top-N, Brier, ECE, calibration table.
Also: SHAP (or gain importance) and fairness slices by gender, age band, language,
vulnerability group.
Final models are refit on all DEVELOPMENT rows and score the EVALUATION cohort.

Outputs
  outputs/episode_predictions.csv   (submission template columns)
  reports/model_metrics.json|md, oof_predictions.csv, calibration_table.csv,
  fairness_report.csv, shap_importance.csv, stage_model_report.txt
  models/*.joblib
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.io_utils import load_df, audit

warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss, f1_score,
                             accuracy_score, log_loss, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEAT_DIR = C.ROOT / "data" / "features"
REPORTS, MODELS, OUT = C.REPORT_DIR, C.ROOT / "models", C.ROOT / "outputs"


# ----------------------------------------------------------------------------- models
def make_pre(nums, cats, scale):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]) if scale else "passthrough"
    return ColumnTransformer([("num", num, nums),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cats)])


def make_lr(nums, cats):
    return Pipeline([("pre", make_pre(nums, cats, True)),
                     ("clf", LogisticRegression(C=0.5, max_iter=3000))])


def make_xgb(nums, cats, seed, n_classes=None):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=5, reg_lambda=2.0, tree_method="hist", n_jobs=-1, random_state=seed)
    if n_classes and n_classes > 2:
        kw.update(objective="multi:softprob", eval_metric="mlogloss")
    else:
        kw.update(eval_metric="aucpr")
    return Pipeline([("pre", make_pre(nums, cats, False)), ("clf", XGBClassifier(**kw))])


# --------------------------------------------------------------------------- metrics
def recall_at_frac(y, p, f):
    k = max(1, int(round(len(y) * f)))
    return float(y[np.argsort(-p)[:k]].sum() / max(1, y.sum()))


def recall_at_k(y, p, k):
    k = min(k, len(y))
    return float(y[np.argsort(-p)[:k]].sum() / max(1, y.sum()))


def precision_at_frac(y, p, f):
    k = max(1, int(round(len(y) * f)))
    return float(y[np.argsort(-p)[:k]].mean())


def ece(y, p, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def metrics(y, p, kf, topk):
    return dict(n=int(len(y)), prevalence=float(y.mean()), auc_pr=float(average_precision_score(y, p)),
                roc_auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)), ece=ece(y, p),
                **{f"recall@top{int(kf*100)}%": recall_at_frac(y, p, kf),
                   f"precision@top{int(kf*100)}%": precision_at_frac(y, p, kf),
                   f"recall@top{topk}": recall_at_k(y, p, topk)},
                lift_at_frac=float(precision_at_frac(y, p, kf) / max(y.mean(), 1e-9)))


# ---------------------------------------------------------------------------- splits
def make_folds(dev, groups, n_folds):
    t = dev["consult_date"].astype("int64").values
    edges = np.quantile(t, np.linspace(0.5, 1.0, n_folds + 1))
    out = []
    for i in range(n_folds):
        va = (t > edges[i]) & (t <= edges[i + 1])
        tr = (t <= edges[i]) & ~np.isin(groups, np.unique(groups[va]))
        out.append((np.where(tr)[0], np.where(va)[0]))
    return out


def age_band(a):
    return pd.cut(pd.to_numeric(a, errors="coerce"), [-1, 17, 39, 59, 200], labels=["<18", "18-39", "40-59", "60+"]).astype(object)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--k-frac", type=float, default=0.10, help="fraction a health worker can call")
    ap.add_argument("--topk", type=int, default=50, help="fixed call capacity for recall@N")
    ap.add_argument("--calibrate", action="store_true", help="isotonic calibration from OOF predictions")
    ap.add_argument("--exclude-features", default="", help="comma list of features to drop (e.g. gender)")
    ap.add_argument("--tiers", default="0.10,0.30", help="cumulative share High,Medium by rank")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-xgb", action="store_true")
    a = ap.parse_args()
    for d in (REPORTS, MODELS, OUT):
        d.mkdir(parents=True, exist_ok=True)
    import joblib

    fx = load_df(FEAT_DIR, "features_t0")
    man = json.loads((FEAT_DIR / "features_manifest.json").read_text())
    excl = {c.strip() for c in a.exclude_features.split(",") if c.strip()}
    nums = [c for c in man["numeric_columns"] if c not in excl]
    cats = [c for c in man["categorical_columns"] if c not in excl]
    feats = nums + cats
    fx["consult_date"] = pd.to_datetime(fx["consult_date"])
    fx = fx.sort_values(["consult_date", "episode_id"]).reset_index(drop=True)

    dev = fx[(fx.cohort == "DEVELOPMENT") & fx.lost_to_followup_label.notna()].reset_index(drop=True)
    ev = fx[fx.cohort == "EVALUATION"].reset_index(drop=True)
    y = dev["lost_to_followup_label"].astype(int).values
    groups = dev["group_household"].astype(str).values
    X, Xev = dev[feats], ev[feats]
    folds = make_folds(dev, groups, a.folds)
    print(f"dev={len(dev)} eval={len(ev)} features={len(feats)} prevalence={y.mean():.3f}")
    for i, (tr, va) in enumerate(folds):
        print(f"  fold{i+1}: train={len(tr)} valid={len(va)} "
              f"({dev.consult_date.iloc[va].min().date()}..{dev.consult_date.iloc[va].max().date()})")

    # ------------------------------------------------------------ Model 1: CV
    builders = {"logreg": lambda: make_lr(nums, cats)}
    if not a.skip_xgb:
        builders["xgboost"] = lambda: make_xgb(nums, cats, a.seed)
    oof = {k: np.full(len(dev), np.nan) for k in builders}
    fold_id = np.full(len(dev), -1)
    fold_rows = []
    last_models = {}
    for i, (tr, va) in enumerate(folds):
        fold_id[va] = i + 1
        for name, mk in builders.items():
            m = mk().fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            oof[name][va] = p
            last_models[name] = m
            fold_rows.append(dict(model=name, fold=i + 1, **metrics(y[va], p, a.k_frac, a.topk)))
    fm = pd.DataFrame(fold_rows)
    summary = fm.groupby("model").mean(numeric_only=True).drop(columns=["fold"])
    print("\nMean across folds:\n", summary.round(3).to_string())
    primary = summary["auc_pr"].idxmax()
    print(f"primary model (by mean AUC-PR): {primary}")

    scored = fold_id > 0
    yo, po = y[scored], oof[primary][scored]

    # ---- calibration (isotonic on OOF) + threshold
    calibrator = None
    if a.calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(po, yo)
        po_cal = calibrator.predict(po)
    else:
        po_cal = po
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(yo, po_cal >= t) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    cal = pd.DataFrame({"y": yo, "p": po_cal})
    cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
    cal.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean")).to_csv(REPORTS / "calibration_table.csv")

    pd.DataFrame({"episode_id": dev.episode_id[scored], "fold": fold_id[scored], "y": yo,
                  **{f"p_{k}": v[scored] for k, v in oof.items()}}).to_csv(REPORTS / "oof_predictions.csv", index=False)

    # ---- fairness slices on OOF of primary model
    fl = dev.loc[scored, ["gender", "preferred_language", "vulnerability_group", "age_t0"]].copy() if all(
        c in dev.columns for c in ("gender", "preferred_language", "vulnerability_group", "age_t0")) else dev.loc[scored].copy()
    if "age_t0" in fl.columns:
        fl["age_band"] = age_band(fl["age_t0"])
    for _c in ("gender", "age_band", "preferred_language", "vulnerability_group"):
        if _c in fl.columns:
            fl[_c] = fl[_c].astype(object).where(fl[_c].notna(), "Unknown")
    cut = np.sort(po_cal)[::-1][max(0, int(round(len(po_cal) * a.k_frac)) - 1)]
    rows = []
    for col in ("gender", "age_band", "preferred_language", "vulnerability_group"):
        if col not in fl.columns:
            continue
        for lv, idx in fl.groupby(col, dropna=False).groups.items():
            pos = fl.index.get_indexer(idx)
            yy, pp = yo[pos], po_cal[pos]
            flagged = pp >= cut
            rows.append(dict(attribute=col, group=str(lv), n=len(yy), prevalence=yy.mean(), mean_risk=pp.mean(),
                             flag_rate=flagged.mean(),
                             tpr=(flagged & (yy == 1)).sum() / max(1, (yy == 1).sum()),
                             fpr=(flagged & (yy == 0)).sum() / max(1, (yy == 0).sum()),
                             precision=(yy[flagged].mean() if flagged.any() else np.nan),
                             auc_pr=(average_precision_score(yy, pp) if len(yy) >= 30 and 0 < yy.sum() < len(yy) else np.nan)))
    fair = pd.DataFrame(rows)
    if len(fair):
        fair["flag_rate_ratio_vs_overall"] = fair["flag_rate"] / max(float((po_cal >= cut).mean()), 1e-9)
        fair["tpr_gap_vs_best_in_attr"] = fair.groupby("attribute")["tpr"].transform("max") - fair["tpr"]
        fair.to_csv(REPORTS / "fairness_report.csv", index=False)

    # ---- explainability
    try:
        m_last = last_models.get("xgboost") or last_models[primary]
        pre, clf = m_last.named_steps["pre"], m_last.named_steps["clf"]
        names = pre.get_feature_names_out()
        va = folds[-1][1]
        Xv = pre.transform(X.iloc[va])
        try:
            import shap
            sv = shap.TreeExplainer(clf).shap_values(Xv)
            imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
        except Exception:
            imp = pd.DataFrame({"feature": names, "mean_abs_shap": getattr(clf, "feature_importances_", np.zeros(len(names)))})
            print("shap unavailable -> saved model feature importances instead")
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(REPORTS / "shap_importance.csv", index=False)
    except Exception as e:
        print("explainability skipped:", e)

    # ------------------------------------------------------ Model 2: dropout stage
    stage_raw = dev["dropout_stage_label"].astype(object).where(dev["dropout_stage_label"].notna(), "Completed care journey").astype(str).str.strip()
    classes = sorted(stage_raw.unique())
    ys = stage_raw.map({c: i for i, c in enumerate(classes)}).values
    completed_idx = next((i for i, c in enumerate(classes) if "complete" in c.lower() or c.lower() in ("none", "no dropout")), None)

    def fit_stage(kind, tr):
        present = np.unique(ys[tr])
        remap = {c: i for i, c in enumerate(present)}
        yt = np.array([remap[v] for v in ys[tr]])
        m = make_lr(nums, cats) if kind == "logreg" else make_xgb(nums, cats, a.seed, n_classes=len(present))
        m.fit(X.iloc[tr], yt)
        return m, present

    def proba_full(m, present, Xn):
        pr = m.predict_proba(Xn)
        full = np.zeros((len(Xn), len(classes)))
        full[:, present] = pr
        return full

    stage_rows, stage_oof = [], {}
    rep = []
    for kind in builders:
        oo = np.zeros((len(dev), len(classes)))
        for i, (tr, va) in enumerate(folds):
            m, present = fit_stage(kind, tr)
            oo[va] = proba_full(m, present, X.iloc[va])
            pv = oo[va].argmax(1)
            stage_rows.append(dict(model=kind, fold=i + 1, accuracy=accuracy_score(ys[va], pv),
                                   macro_f1=f1_score(ys[va], pv, average="macro"),
                                   log_loss=log_loss(ys[va], np.clip(oo[va], 1e-6, 1), labels=range(len(classes)))))
        stage_oof[kind] = oo
        pv = oo[scored].argmax(1)
        rep.append(f"== {kind} (pooled OOF) ==\n" + classification_report(ys[scored], pv, target_names=classes, zero_division=0))
    sm = pd.DataFrame(stage_rows)
    (REPORTS / "stage_model_report.txt").write_text(sm.to_string(index=False) + "\n\n" + "\n".join(rep))
    stage_primary = sm.groupby("model")["macro_f1"].mean().idxmax()
    print("\nStage model mean:\n", sm.groupby("model").mean(numeric_only=True).round(3).to_string(), f"\nprimary stage model: {stage_primary}")

    # --------------------------------------------------------- final refit + score EVAL
    allidx = np.arange(len(dev))
    final = builders[primary]().fit(X, y)
    p_ev = final.predict_proba(Xev)[:, 1]
    if calibrator is not None:
        p_ev = calibrator.predict(p_ev)
    joblib.dump(final, MODELS / f"lost_to_followup_{primary}.joblib")
    mst, present = fit_stage(stage_primary, allidx)
    joblib.dump((mst, present, classes), MODELS / f"dropout_stage_{stage_primary}.joblib")
    sp = proba_full(mst, present, Xev)

    pred_lost = (p_ev >= thr).astype(int)
    non_completed = [i for i in range(len(classes)) if i != completed_idx]
    stage_pred = []
    for row, flag in zip(sp, pred_lost):
        if flag and completed_idx is not None:
            stage_pred.append(classes[non_completed[int(np.argmax(row[non_completed]))]])
        elif completed_idx is not None:
            stage_pred.append(classes[completed_idx])
        else:
            stage_pred.append(classes[int(np.argmax(row))])
    hi, mid = (float(x) for x in a.tiers.split(","))
    rank_pct = pd.Series(p_ev).rank(ascending=False, method="first") / len(p_ev)
    tier = np.where(rank_pct <= hi, "High", np.where(rank_pct <= mid, "Medium", "Low"))
    pred = pd.DataFrame({"episode_id": ev["episode_id"], "risk_probability": np.round(p_ev, 5),
                         "predicted_lost_to_followup": pred_lost, "predicted_dropout_stage": stage_pred,
                         "priority_tier": tier})
    pred.to_csv(OUT / "episode_predictions.csv", index=False)
    pd.concat([ev[["episode_id"]], pd.DataFrame(sp, columns=[f"p_stage_{c}" for c in classes])], axis=1).to_csv(
        REPORTS / "eval_stage_probabilities.csv", index=False)

    # ------------------------------------------------------------------- reports
    out = dict(primary_model=primary, stage_model=stage_primary, threshold=thr, calibrated=a.calibrate,
               k_frac=a.k_frac, topk=a.topk, n_dev=int(len(dev)), n_eval=int(len(ev)), n_features=len(feats),
               dev_prevalence=float(y.mean()), per_fold=fold_rows,
               mean_by_model=summary.round(4).to_dict(orient="index"),
               oof_pooled_primary=metrics(yo, po_cal, a.k_frac, a.topk),
               eval_predicted_positive_rate=float(pred_lost.mean()))
    (REPORTS / "model_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    md = ["# Model metrics (time-based grouped CV, DEVELOPMENT cohort)", "", f"Primary: **{primary}**, threshold {thr:.2f}, calibrated={a.calibrate}", "",
          summary.round(3).to_markdown(), "", "## Pooled OOF (primary)", pd.Series(out["oof_pooled_primary"]).round(3).to_markdown()]
    try:
        (REPORTS / "model_metrics.md").write_text("\n".join(md))
    except Exception:
        pass
    audit("models_trained", primary=primary, stage_model=stage_primary, n_eval=len(ev))
    print(f"\nWrote outputs/episode_predictions.csv ({len(pred)} rows) and reports/*")


if __name__ == "__main__":
    main()