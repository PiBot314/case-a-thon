"""Linkage evaluation + reporting: review queue, hand-check sample (+scorer), diagnostics, synthetic benchmark."""
import json
import random
import numpy as np
import pandas as pd
from . import config as C

SYSTEMS = ["teleconsultations", "ncd_screening", "prescriptions", "medicine_dispensing",
           "lab_tests", "visit_history", "followup_visits", "outreach_actions"]
_SRC_COLS = ["name_norm", "age", "gender", "village_norm", "block", "district"]


def _ctx(lk, ident, p360):
    d = lk.merge(ident[["source_system", "source_patient_id"] + [c for c in _SRC_COLS + ["phone_norm"] if c in ident]],
                 on=["source_system", "source_patient_id"], how="left")
    d["src_phone_last4"] = d["phone_norm"].astype("string").str[-4:] if "phone_norm" in d else pd.NA
    p = p360.set_index("patient_id")
    for tag, col in (("", "patient_id"), ("runner_up_", "runner_up_patient_id")):
        d[f"{tag}match_name"] = d[col].map(p["canonical_name"])
    d["match_age"] = d["patient_id"].map(p["age_as_of_2026"])
    d["match_village"] = d["patient_id"].map(p["village"])
    d["match_last4"] = d["patient_id"].map(p["masked_mobile"].astype(str).str[-4:])
    return d


def export_review_queue(lk, ident, p360, path=None):
    d = _ctx(lk, ident, p360)
    q = d[d.review_flag].sort_values("link_score")
    cols = ["source_system", "source_patient_id", "name_norm", "age", "village_norm", "src_phone_last4",
            "patient_id", "match_name", "match_age", "match_village", "match_last4", "link_score",
            "runner_up_patient_id", "runner_up_match_name", "runner_up_score"]
    path = path or C.REPORT_DIR / "linkage_review_queue.csv"
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    q[[c for c in cols if c in q]].to_csv(path, index=False)
    return q


def export_hand_check(lk, ident, p360, n=(80, 60, 60), seed=7, path=None):
    """Stratified sample to label by hand: is_correct (Y/N) and, if N / unmatched, true_patient_id if you can tell."""
    d = _ctx(lk, ident, p360)
    d["stratum"] = np.select([d.link_type.eq("anchor_auto"), d.link_type.eq("anchor_review")],
                             ["auto", "review"], "unanchored")
    parts = []
    for s, k in zip(["auto", "review", "unanchored"], n):
        g = d[d.stratum == s]
        parts.append(g.sample(min(k, len(g)), random_state=seed).assign(stratum_size=len(g)))
    out = pd.concat(parts)
    out["is_correct"] = ""
    out["true_patient_id"] = ""
    cols = ["stratum", "stratum_size", "source_system", "source_patient_id", "name_norm", "age", "gender",
            "village_norm", "src_phone_last4", "patient_id", "match_name", "match_age", "match_village",
            "match_last4", "link_score", "runner_up_patient_id", "runner_up_match_name", "is_correct", "true_patient_id"]
    path = path or C.REPORT_DIR / "linkage_hand_check_template.csv"
    out[[c for c in cols if c in out]].to_csv(path, index=False)
    return out


def score_hand_check(path):
    """Stratum-weighted precision/recall from the labelled template.
    linked = auto+review strata. precision = P(is_correct=Y | linked).
    recall  = correct links / all identities that truly belong to a known patient
              (correct links + wrong links with a known true id + unanchored with a known true id)."""
    d = pd.read_csv(path, dtype=str).fillna("")
    d["stratum_size"] = d["stratum_size"].astype(float)
    lab = d[d.is_correct.str.upper().isin(["Y", "N"]) | d.true_patient_id.ne("")]
    if lab.empty:
        return {"status": "no labels filled in yet"}
    tp = fn = linked = 0.0
    for s, g in lab.groupby("stratum"):
        w = g.stratum_size.iloc[0] / len(g)
        if s in ("auto", "review"):
            ok = g.is_correct.str.upper().eq("Y").sum()
            wrong_known = (g.is_correct.str.upper().eq("N") & g.true_patient_id.ne("")).sum()
            tp += w * ok; linked += w * len(g); fn += w * wrong_known
        else:
            fn += w * g.true_patient_id.ne("").sum()
    return {"precision": tp / linked if linked else None, "recall": tp / (tp + fn) if tp + fn else None,
            "n_labelled": int(len(lab))}


def diagnostics(lk, ident):
    dd = {}
    dd["by_type"] = lk.link_type.value_counts().to_dict()
    dd["by_system"] = {k: v for k, v in lk.groupby("source_system").link_type.apply(
        lambda s: (s.isin(["anchor_auto", "anchor_review", "cross_source_anchor"]).mean().round(4))).items()}
    dd["patients"] = int(lk.patient_id.nunique())
    dd["anchored_patients"] = int(lk.patient_id[~lk.patient_id.str.startswith("LINK_")].nunique())
    # identity-consistency diagnostic: one person normally holds one source id per system
    multi = (lk[~lk.patient_id.str.startswith("LINK_")].groupby(["patient_id", "source_system"]).size() > 1)
    dd["patient_system_pairs_with_multiple_ids"] = int(multi.sum())
    dd["review_flagged"] = int(lk.review_flag.sum())
    dd["village_repaired_share"] = round(float(lk.village_repaired.mean()), 4)
    dd["score_quantiles"] = lk.loc[lk.link_type.isin(["anchor_auto", "anchor_review"]), "link_score"].quantile([.05, .25, .5, .75]).round(3).to_dict()
    return dd


def write_report(lk, dd, path=None, bench=None):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    hc = C.REPORT_DIR / "linkage_hand_check_labelled.csv"
    dd = dict(dd); dd["hand_check"] = score_hand_check(hc) if hc.exists() else "pending - label reports/linkage_hand_check_template.csv, save as linkage_hand_check_labelled.csv"
    if bench: dd["synthetic_benchmark"] = bench
    (C.REPORT_DIR / "linkage_summary.json").write_text(json.dumps(dd, indent=2, default=str))
    L = ["# Patient Linkage Summary", f"- Source identities: **{len(lk):,}** -> patients: **{dd['patients']:,}** "
         f"({dd['anchored_patients']:,} anchored to patient_360)", "", "| link_type | n |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in dd["by_type"].items()]
    L += ["", "Share linked to a patient_360 id, by system: " + ", ".join(f"{k} {v:.1%}" for k, v in dd["by_system"].items()),
          f"- Review-zone links (best match taken, in reports/linkage_review_queue.csv): {dd['review_flagged']:,}",
          f"- Patient x system pairs holding >1 source id (possible wrong merges/dupes): {dd['patient_system_pairs_with_multiple_ids']:,}",
          f"- Villages repaired against geography_reference: {dd['village_repaired_share']:.1%}", "",
          f"## Precision / recall\n```\n{json.dumps(dd['hand_check'], indent=1, default=str)}\n```"]
    if bench:
        L.append(f"## Synthetic benchmark (known truth)\n```\n{json.dumps(bench, indent=1)}\n```")
    (path or C.REPORT_DIR / "linkage_summary.md").write_text("\n".join(L))


# ── synthetic benchmark (real raw data is deleted, so truth is injected) ──────
FIRST = "mohan ravi suresh ramesh venkat lakshmi sunita anitha srinivas krishna narsimha yadagiri mallaiah shankar bhaskar kavitha padma sarita rajesh mahesh naresh sathish ganesh swathi vijaya anjali balaraju ramulu".split()
LAST = "basha laxman rao reddy goud naik yadav chary mallaiah kumar devi begum pandit sharma verma mudiraj patel nayak".split()

def _typo(s, rng):
    if len(s) < 4: return s
    i = rng.randrange(1, len(s) - 1); op = rng.choice("dsr")
    return s[:i] + s[i + 1:] if op == "d" else (s[:i] + s[i + 1] + s[i] + s[i + 2:] if op == "s" and i + 2 <= len(s) else s[:i] + rng.choice("aeiou") + s[i + 1:])

_PHON = [("mohan", "mohn"), ("laxman", "lakshman"), ("lakshmi", "laxmi"), ("venkat", "venkata"), ("krishna", "krishn"),
         ("suresh", "sures"), ("srinivas", "sreenivas"), ("shankar", "sankar"), ("bhaskar", "baskar"), ("ganesh", "ganesha"),
         ("mallaiah", "mallayya"), ("reddy", "reddi"), ("yadav", "yadab")]

def make_synthetic(n=5000, seed=1, noise=1.0):
    rng = random.Random(seed)
    geo_rows, vid = [], 0
    for d in ["Nalgonda", "Mahabubnagar", "Sangareddy", "Warangal", "Nizamabad"]:
        for b in range(2):
            for v in range(4):
                vid += 1; geo_rows.append((f"VIL{vid:04d}", f"{d[:3]}village{chr(97+b)}{chr(97+v)}".replace("village", "pally"), d, f"{d[:3]}blk{b}"))
    geo = pd.DataFrame(geo_rows, columns=["village_id", "village", "district", "block"])
    geo["village_norm"] = geo["village"].str.lower()
    pats = []
    for i in range(n):
        g = geo.iloc[rng.randrange(len(geo))]
        pats.append(dict(patient_id=f"P{i+1:06d}", canonical_name=f"{rng.choice(FIRST).title()} {rng.choice(LAST).title()}",
                         gender=rng.choice("MF"), age_as_of_2026=rng.randint(18, 85), phone=f"{rng.randint(6,9)}{rng.randint(0,10**9-1):09d}",
                         village=g.village, village_norm=g.village_norm, block=g.block, district=g.district))
    p360 = pd.DataFrame(pats)
    p360["name_norm"] = p360.canonical_name.str.lower()
    p360["masked_mobile"] = "XXXXXX" + p360.phone.str[-4:]
    p360["phone_last4"] = p360.phone.str[-4:]
    rows = []
    for p in pats:
        for sysname in SYSTEMS:
            if rng.random() > 0.72: continue
            nm = p["canonical_name"].lower().split()
            r = rng.random() / max(noise, 1e-9)
            if r < .12: nm = [_typo(t, rng) if rng.random() < .6 else t for t in nm]
            elif r < .24:
                nm = [next((b for a, b in _PHON if a == t), t) for t in nm]
            elif r < .32: nm = nm[::-1]
            elif r < .37: nm = nm[:1]
            elif r < .41: nm = [nm[0][0]] + nm[1:]
            ph = p["phone"]
            r = rng.random() / max(noise, 1e-9)
            ph = None if r < .08 else (ph[:-2] + f"{rng.randint(0,99):02d}" if r < .13 else ph)
            age = p["age_as_of_2026"]; r = rng.random() / max(noise, 1e-9)
            age = age if r < .6 else age + rng.choice([-2, -1, 1, 2]) if r < .9 else age + rng.choice([-5, -4, -3, 3, 4, 5])
            vil = p["village_norm"]; r = rng.random() / max(noise, 1e-9)
            vil = _typo(vil, rng) if r < .05 else (rng.choice(list(geo.village_norm)) if r < .08 else vil)
            gen = p["gender"]; r = rng.random() / max(noise, 1e-9)
            gen = None if r < .03 else ({"M": "F", "F": "M"}[gen] if r < .05 else gen)
            dist, blk = p["district"], p["block"]
            if rng.random() < .03 * noise:
                x = geo.iloc[rng.randrange(len(geo))]; dist, blk = x.district, x.block
            rows.append(dict(source_system=sysname, source_patient_id=f"{sysname[:3].upper()}{len(rows):07d}",
                             name_norm=" ".join(nm), phone_norm=ph, village_norm=vil, gender=gen, age=age,
                             block=blk, district=dist, true_patient_id=p["patient_id"]))
    # orphans: people who exist in a source system but not in patient_360
    for k in range(int(0.03 * len(rows))):
        g = geo.iloc[rng.randrange(len(geo))]
        rows.append(dict(source_system=rng.choice(SYSTEMS), source_patient_id=f"ORP{k:06d}",
                         name_norm=f"{rng.choice(FIRST)} {rng.choice(LAST)}", phone_norm=f"{rng.randint(6,9)}{rng.randint(0,10**9-1):09d}",
                         village_norm=g.village_norm, gender=rng.choice("MF"), age=rng.randint(18, 85),
                         block=g.block, district=g.district, true_patient_id=None))
    ident = pd.DataFrame(rows)
    return ident, p360, geo


def benchmark(n=5000, seed=1, noise=1.0, accept=None, review=None):
    from . import linkage as L
    ident, p360, geo = make_synthetic(n, seed, noise)
    lk = L.link_identities(ident.drop(columns="true_patient_id"), p360, geo,
                           accept=accept or L.ACCEPT, review=review or L.REVIEW)
    lk = lk.merge(ident[["source_system", "source_patient_id", "true_patient_id"]], on=["source_system", "source_patient_id"])
    anch = ~lk.patient_id.str.startswith("LINK_")
    truth = lk.true_patient_id.notna()
    tp = (anch & truth & (lk.patient_id == lk.true_patient_id)).sum()
    res = {"identities": len(lk), "precision_anchored": round(tp / anch.sum(), 4),
           "recall_of_true_members": round(tp / truth.sum(), 4),
           "orphans_wrongly_anchored": int((anch & ~truth).sum()), "orphans": int((~truth).sum()),
           "auto_precision": round(float(((lk.link_type == "anchor_auto") & (lk.patient_id == lk.true_patient_id)).sum() / (lk.link_type == "anchor_auto").sum()), 4),
           "review_precision": round(float(((lk.link_type == "anchor_review") & (lk.patient_id == lk.true_patient_id)).sum() / max((lk.link_type == "anchor_review").sum(), 1)), 4),
           "review_n": int((lk.link_type == "anchor_review").sum()),
           "link_types": lk.link_type.value_counts().to_dict()}
    return res, lk
