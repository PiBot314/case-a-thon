"""
linkage.py - Stage 2: construct the Patient from scattered source identities
============================================================================
Input : silver/source_identities   one row per (source_system, source_patient_id)
        silver/patient_360_reference   anchor identity list (P000001...)
        silver/geography_reference     for village spelling repair
Output: silver/patient_linkage         source_system, source_patient_id -> patient_id  (every table)
        outputs/submission_linkage.csv source_system, source_patient_id, predicted_patient_id, confidence
        reports/linkage_review_queue.csv   borderline links (best match already taken; human may overturn)
        reports/linkage_hand_check_template.csv   stratified sample to hand-label
        reports/linkage_summary.{md,json}

Method
------
1. Normalise: names (ascii/lowercase/titles stripped, token-sorted, phonetic key), phones (last-4),
   villages repaired against geography_reference (Jaro-Winkler within the record's own block first).
2. Blocking (union of three keys, so one noisy field never hides a true pair):
     (district, block, phone_last4)  |  (phone_last4, name-token phonetic)  |  (district, block, village, name-token phonetic)
3. Score = weighted agreement over the fields that are present in BOTH records (missing != mismatch):
     name 0.35 (max of JW on raw / token-sorted / phonetic key / token-set),  phone last-4 0.30,
     age within +-2 0.15,  village 0.15 (graded),  gender 0.05
4. Each source identity is matched to its best patient_360 candidate.
     score >= ACCEPT  -> auto link          REVIEW <= score < ACCEPT -> best match taken, flagged for review queue
     below REVIEW     -> not anchored; attached to its best already-anchored neighbour in another system
                         (>= ACCEPT), else clustered with other unanchored identities (LINK_<hash>), else singleton.
"""
import json
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import config as C
from .io_utils import save_df, load_df, audit
from .normalize import jaro_winkler as jw, phonetic_token, phonetic_name, standardize_villages

ACCEPT = 0.80
REVIEW = 0.70
W = dict(name=0.35, phone=0.30, age=0.15, village=0.15, gender=0.05)
AGE_TOL = 2


# ── record preparation ────────────────────────────────────────────────────────
def _s(v):
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def prepare(df, age_col="age", has_full_phone=True):
    """DataFrame -> list of light dicts used by the scorer (fast, no pandas in the hot loop)."""
    recs = []
    for r in df.to_dict("records"):
        name = _s(r.get("name_norm"))
        toks = name.split() if name else []
        l4 = _s(r.get("phone_last4"))
        if l4 is None and has_full_phone and _s(r.get("phone_norm")):
            l4 = str(r["phone_norm"])[-4:]
        age = r.get(age_col)
        recs.append(dict(
            name=name, toks=toks, sorted=" ".join(sorted(toks)) if toks else None,
            phon=_s(phonetic_name(name)) if name else None,
            ptoks=[phonetic_token(t) for t in toks if len(t) > 1],
            l4=l4, village=_s(r.get("village_std", r.get("village_norm"))),
            age=None if age is None or pd.isna(age) else float(age),
            gender=_s(r.get("gender")), district=_s(r.get("district")), block=_s(r.get("block"))))
    return recs


# ── similarity ────────────────────────────────────────────────────────────────
def name_sim(a, b):
    if not a["name"] or not b["name"]:
        return None
    if a["name"] == b["name"]:
        return 1.0
    best = max(jw(a["name"], b["name"]), jw(a["sorted"], b["sorted"]))
    if a["phon"] and b["phon"]:
        best = max(best, 0.97 * jw(a["phon"], b["phon"]))
    # token-set: every token of the shorter name found in the longer one (handles dropped surname / initials)
    s, l = (a["toks"], b["toks"]) if len(a["toks"]) <= len(b["toks"]) else (b["toks"], a["toks"])
    if s and l:
        tok = sum(max(max(jw(x, y), 1.0 if (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)) else 0)
                      for y in l) for x in s) / len(s)
        best = max(best, tok * (1.0 if len(s) == len(l) else 0.9))
    return best


def score_pair(a, b):
    """Weighted agreement over fields present in both records. Returns (score, components)."""
    comp = {}
    comp["name"] = name_sim(a, b)
    comp["phone"] = None if not (a["l4"] and b["l4"]) else float(a["l4"] == b["l4"])
    comp["age"] = None if a["age"] is None or b["age"] is None else float(abs(a["age"] - b["age"]) <= AGE_TOL)
    if a["village"] and b["village"]:
        comp["village"] = 1.0 if a["village"] == b["village"] else (jw(a["village"], b["village"]) if jw(a["village"], b["village"]) >= 0.9 else 0.0)
    else:
        comp["village"] = None
    comp["gender"] = None if not (a["gender"] and b["gender"]) else float(a["gender"] == b["gender"])
    num = den = 0.0
    for k, v in comp.items():
        if v is not None:
            num += W[k] * v; den += W[k]
    if den == 0 or comp["name"] is None:
        return 0.0, comp
    # evidence guard: a name-only agreement cannot reach the accept zone
    n_other = sum(v is not None for k, v in comp.items() if k != "name")
    sc = num / den
    if n_other == 0:
        sc = min(sc, REVIEW - 0.01)
    return sc, comp


# ── blocking ──────────────────────────────────────────────────────────────────
def block_keys(r):
    keys = []
    if r["district"] and r["block"] and r["l4"]:
        keys.append(("dbp", r["district"], r["block"], r["l4"]))
    for t in r["ptoks"]:
        if r["l4"]:
            keys.append(("pn", r["l4"], t))
        if r["district"] and r["block"] and r["village"]:
            keys.append(("vn", r["district"], r["block"], r["village"], t))
    return keys


class BlockIndex:
    def __init__(self, recs):
        self.recs, self.idx = recs, defaultdict(list)
        for i, r in enumerate(recs):
            for k in block_keys(r):
                self.idx[k].append(i)

    def candidates(self, r):
        out = set()
        for k in block_keys(r):
            out.update(self.idx.get(k, ()))
        return out


# ── core linker (pure function: no disk) ──────────────────────────────────────
def link_identities(ident: pd.DataFrame, p360: pd.DataFrame, geo: pd.DataFrame,
                    accept=ACCEPT, review=REVIEW) -> pd.DataFrame:
    ident = ident.copy().reset_index(drop=True)
    p360 = p360.copy().reset_index(drop=True)
    if "village_norm" not in geo:
        from .normalize import norm_village
        geo = geo.assign(village_norm=geo["village"].map(norm_village))
    ident["village_std"], ident["village_repaired"] = standardize_villages(ident, geo)
    p360["village_std"], _ = standardize_villages(p360, geo)

    A = prepare(p360, age_col="age_as_of_2026", has_full_phone=False)
    I = prepare(ident, age_col="age")
    anchor_idx = BlockIndex(A)
    pids = p360["patient_id"].tolist()

    n = len(I)
    res = [dict(patient_id=None, link_type=None, score=0.0, runner_up=0.0, n_cand=0, name_sim=None, second_pid=None)
           for _ in range(n)]

    # -- step A: anchor to patient_360 --------------------------------------
    for i, r in enumerate(I):
        cands = anchor_idx.candidates(r)
        scored = sorted(((score_pair(r, A[j])[0], j) for j in cands), reverse=True)
        res[i]["n_cand"] = len(scored)
        if not scored:
            continue
        best, j = scored[0]
        res[i]["score"] = round(best, 4)
        res[i]["runner_up"] = round(scored[1][0], 4) if len(scored) > 1 else 0.0
        res[i]["second_pid"] = pids[scored[1][1]] if len(scored) > 1 else None
        res[i]["name_sim"] = score_pair(r, A[j])[1]["name"]
        if best >= accept:
            res[i].update(patient_id=pids[j], link_type="anchor_auto")
        elif best >= review:
            res[i].update(patient_id=pids[j], link_type="anchor_review")   # best match taken; flagged
        else:
            res[i]["cand_pid"] = pids[j]

    # -- step B: unanchored identities -> attach to anchored neighbour in another system, else cluster ------
    ident_idx = BlockIndex(I)
    systems = ident["source_system"].tolist()
    un = [i for i in range(n) if res[i]["patient_id"] is None]
    still = []
    for i in un:
        best, bj = 0.0, None
        for j in ident_idx.candidates(I[i]):
            if j == i or systems[j] == systems[i] or res[j]["patient_id"] is None:
                continue
            if res[j]["link_type"] != "anchor_auto":
                continue
            sc = score_pair(I[i], I[j])[0]
            if sc > best:
                best, bj = sc, j
        if bj is not None and best >= accept:
            res[i].update(patient_id=res[bj]["patient_id"], link_type="cross_source_anchor", score=round(best, 4))
        else:
            still.append(i)

    parent = {i: i for i in still}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    sset = set(still)
    edge_best = defaultdict(float)
    for i in still:
        for j in ident_idx.candidates(I[i]):
            if j <= i or j not in sset or systems[j] == systems[i]:
                continue
            sc = score_pair(I[i], I[j])[0]
            if sc >= accept:
                parent[find(i)] = find(j)
                edge_best[i] = max(edge_best[i], sc); edge_best[j] = max(edge_best[j], sc)
    comps = defaultdict(list)
    for i in still:
        comps[find(i)].append(i)
    keys = list(zip(ident["source_system"], ident["source_patient_id"]))
    for members in comps.values():
        h = hashlib.sha1("|".join(sorted(f"{keys[m][0]}:{keys[m][1]}" for m in members)).encode()).hexdigest()[:10].upper()
        for m in members:
            res[m].update(patient_id=f"LINK_{h}",
                          link_type="cross_source_cluster" if len(members) > 1 else "singleton",
                          score=round(edge_best.get(m, res[m]["score"]), 4) if len(members) > 1 else res[m]["score"])

    out = pd.DataFrame({
        "source_system": ident["source_system"], "source_patient_id": ident["source_patient_id"],
        "patient_id": [r["patient_id"] for r in res],
        "link_type": [r["link_type"] for r in res],
        "link_score": [r["score"] for r in res],
        "review_flag": [r["link_type"] == "anchor_review" for r in res],
        "runner_up_score": [r["runner_up"] for r in res],
        "runner_up_patient_id": [r["second_pid"] for r in res],
        "n_candidates": [r["n_cand"] for r in res],
        "village_repaired": ident["village_repaired"].values,
    })
    # confidence: score, discounted when the runner-up is close (ambiguous)
    margin = (out["link_score"] - out["runner_up_score"]).clip(lower=0)
    out["confidence"] = (out["link_score"] * (0.85 + 0.15 * (margin / 0.15).clip(upper=1))).round(4)
    out.loc[out.link_type.isin(["singleton"]), "confidence"] = 0.0
    return out


# ── pipeline entry point ──────────────────────────────────────────────────────
def run(silver=None, out_dir=None) -> pd.DataFrame:
    audit("linkage_start")
    def _load(name):
        if silver and name in silver:
            return silver[name]
        return load_df(C.SILVER_DIR, name)
    try:
        ident, p360, geo = _load("source_identities"), _load("patient_360_reference"), _load("geography_reference")
    except FileNotFoundError as e:
        audit("linkage_skip", reason=str(e)); return pd.DataFrame()
    if "village_norm" not in geo:
        from .normalize import norm_village
        geo = geo.assign(village_norm=geo["village"].map(norm_village))

    lk = link_identities(ident, p360, geo)
    save_df(lk, C.SILVER_DIR, "patient_linkage")

    out_dir = out_dir or (C.ROOT / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (lk[["source_system", "source_patient_id", "patient_id", "confidence"]]
        .rename(columns={"patient_id": "predicted_patient_id"})
        .to_csv(out_dir / "submission_linkage.csv", index=False))

    from .linkage_eval import export_review_queue, export_hand_check, diagnostics, write_report
    export_review_queue(lk, ident, p360)
    export_hand_check(lk, ident, p360)
    write_report(lk, diagnostics(lk, ident))
    audit("linkage_end", rows=len(lk), patients=int(lk.patient_id.nunique()),
          by_type=lk.link_type.value_counts().to_dict())
    return lk
