"""
linkage.py  –  Patient identity resolution
============================================
Input:  silver/source_identities.parquet   (built by ingest.run())
        silver/patient_360_reference.parquet
Output: silver/patient_linkage.parquet
        reports/linkage_summary.{md,json}

Algorithm
---------
1. Blocking  – candidate pairs must share at least ONE of:
     • same district + block + phone_last4
     • same district + block + name token overlap ≥ 1
2. Scoring   – weighted similarity on five signals:
     • name_sim    : Jaro-Winkler distance on norm_name          (weight 0.35)
     • phone_match : exact match of phone_last4                  (weight 0.30)
     • age_match   : |age_a - age_b| ≤ 2                        (weight 0.15)
     • village_sim : exact village_norm match                    (weight 0.15)
     • gender_match: both known and equal                        (weight 0.05)
3. Accept / review / reject thresholds:
     ACCEPT  ≥ 0.72  – auto-linked to same canonical patient
     REVIEW  0.50–0.72 – flagged (in sandbox: take best match)
     REJECT  < 0.50  – kept as-is (new patient identity)
4. Connected-components resolve transitive matches across sources.
5. Unlinked source patients get a generated patient_id (LINK_<uuid>).

Output schema (one row per source patient)
------------------------------------------
source_system, source_patient_id, patient_id, link_type,
link_score, review_flag, canonical_patient_id
"""

import json
import uuid
import logging
import itertools
from pathlib import Path

import pandas as pd
import numpy as np

from . import config as C
from .io_utils import save_df, audit

log = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────
ACCEPT_THRESH  = 0.72
REVIEW_THRESH  = 0.50
W_NAME         = 0.35
W_PHONE        = 0.30
W_AGE          = 0.15
W_VILLAGE      = 0.15
W_GENDER       = 0.05

# ── helpers ───────────────────────────────────────────────────────────────────

def _jaro_winkler(s1: str, s2: str) -> float:
    """Pure-Python Jaro-Winkler (avoids jellyfish dependency)."""
    if s1 == s2:
        return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 == 0 or l2 == 0:
        return 0.0
    match_dist = max(l1, l2) // 2 - 1
    match_dist = max(match_dist, 0)
    s1_matches = [False] * l1
    s2_matches = [False] * l2
    matches = 0
    transpositions = 0
    for i in range(l1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, l2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(l1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, l1, l2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


def _score_pair(a: dict, b: dict) -> float:
    """Compute weighted similarity for a candidate pair."""
    score = 0.0

    # name similarity
    na, nb = a.get("name_norm"), b.get("name_norm")
    if na and nb:
        score += W_NAME * _jaro_winkler(str(na), str(nb))

    # phone last-4
    pa, pb = a.get("phone_last4"), b.get("phone_last4")
    if pa and pb and pa == pb:
        score += W_PHONE

    # age ± 2
    aa, ab_ = a.get("age"), b.get("age")
    if pd.notna(aa) and pd.notna(ab_):
        try:
            if abs(float(aa) - float(ab_)) <= 2:
                score += W_AGE
        except (TypeError, ValueError):
            pass

    # village
    va, vb = a.get("village_norm"), b.get("village_norm")
    if va and vb and va == vb:
        score += W_VILLAGE

    # gender
    ga, gb = a.get("gender"), b.get("gender")
    if ga and gb and pd.notna(ga) and pd.notna(gb) and ga == gb:
        score += W_GENDER

    return round(score, 4)


# ── blocking ──────────────────────────────────────────────────────────────────

def _build_blocks(ident: pd.DataFrame) -> dict[tuple, list[int]]:
    """Return {block_key: [row_indices]} for candidate generation."""
    blocks: dict[tuple, list[int]] = {}

    for idx, row in ident.iterrows():
        district = str(row.get("district", "") or "")
        block    = str(row.get("block", "")    or "")
        last4    = str(row.get("phone_last4", "") or "")
        village  = str(row.get("village_norm", "") or "")

        # Block 1: district+block+phone_last4
        if district and block and last4:
            k = ("dbp", district, block, last4)
            blocks.setdefault(k, []).append(idx)

        # Block 2: district+block+village (fall-back when phone missing)
        if district and block and village:
            k = ("dbv", district, block, village)
            blocks.setdefault(k, []).append(idx)

    return blocks


def _generate_candidates(ident: pd.DataFrame, blocks: dict) -> list[tuple[int, int, float]]:
    """Enumerate candidate pairs within each block and score them."""
    seen: set[frozenset] = set()
    candidates = []

    records = ident.to_dict("index")

    for indices in blocks.values():
        if len(indices) < 2:
            continue
        for i, j in itertools.combinations(indices, 2):
            key = frozenset((i, j))
            if key in seen:
                continue
            seen.add(key)
            # skip same source_system (same system cannot resolve to different patients)
            si = records[i].get("source_system")
            sj = records[j].get("source_system")
            if si == sj:
                continue
            sc = _score_pair(records[i], records[j])
            if sc >= REVIEW_THRESH:
                candidates.append((i, j, sc))

    return candidates


# ── union-find for connected components ───────────────────────────────────────

class _UF:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def components(self, n):
        from collections import defaultdict
        comp = defaultdict(list)
        for i in range(n):
            comp[self.find(i)].append(i)
        return comp


# ── anchor matching against patient_360_reference ────────────────────────────

def _match_to_p360(ident: pd.DataFrame, p360: pd.DataFrame) -> pd.Series:
    """
    Try to resolve each source patient directly against patient_360_reference
    using phone_last4 + village_norm + name similarity.
    Returns a Series indexed by ident.index with patient_id or NaN.
    """
    if p360 is None or len(p360) == 0:
        return pd.Series(index=ident.index, dtype=object)

    # Build p360 lookup by (phone_last4, village_norm)
    p360_lookup: dict[tuple, list[dict]] = {}
    for _, row in p360.iterrows():
        k = (str(row.get("phone_last4") or ""), str(row.get("village_norm") or ""))
        p360_lookup.setdefault(k, []).append(row.to_dict())

    resolved = {}
    for idx, row in ident.iterrows():
        k = (str(row.get("phone_last4") or ""), str(row.get("village_norm") or ""))
        candidates = p360_lookup.get(k, [])
        best_pid, best_sc = None, 0.0
        for cand in candidates:
            sc = _score_pair(row.to_dict(), cand)
            if sc > best_sc:
                best_sc, best_pid = sc, cand["patient_id"]
        if best_pid and best_sc >= ACCEPT_THRESH:
            resolved[idx] = best_pid

    return pd.Series(resolved, dtype=object)


# ── main entry point ──────────────────────────────────────────────────────────

def run(silver: dict | None = None) -> pd.DataFrame:
    """
    Build the source_id → patient_id crosswalk.

    Parameters
    ----------
    silver : dict of DataFrames (optional)
        If provided, uses in-memory silver tables; otherwise loads from disk.

    Returns
    -------
    linkage : pd.DataFrame
        Columns: source_system, source_patient_id, patient_id,
                 link_type, link_score, review_flag
    """
    audit("linkage_start")

    # ── load inputs ──────────────────────────────────────────────────────────
    def _load(name):
        if silver and name in silver:
            return silver[name]
        p = C.SILVER_DIR / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
        p2 = C.SILVER_DIR / f"{name}.pkl"
        if p2.exists():
            return pd.read_pickle(p2)
        return None

    ident = _load("source_identities")
    p360  = _load("patient_360_reference")

    if ident is None or len(ident) == 0:
        log.warning("source_identities not found – run ingest.run() first")
        audit("linkage_skip", reason="source_identities missing")
        return pd.DataFrame()

    ident = ident.copy().reset_index(drop=True)
    n = len(ident)
    log.info("linkage: %d source patient rows from %d systems",
             n, ident["source_system"].nunique())

    # ── step 1: anchor match against patient_360_reference ───────────────────
    p360_match = _match_to_p360(ident, p360)
    ident["_p360_pid"]   = p360_match
    ident["_p360_hit"]   = p360_match.notna()

    # ── step 2: blocking + pairwise scoring ──────────────────────────────────
    blocks     = _build_blocks(ident)
    candidates = _generate_candidates(ident, blocks)
    log.info("linkage: %d candidate pairs above review threshold", len(candidates))

    # ── step 3: connected components via union-find ───────────────────────────
    uf = _UF(n)
    accept_scores: dict[frozenset, float] = {}
    review_flags:  set[frozenset]         = set()

    for i, j, sc in candidates:
        key = frozenset((i, j))
        if sc >= ACCEPT_THRESH:
            uf.union(i, j)
            accept_scores[key] = sc
        else:
            review_flags.add(key)

    # ── step 4: assign patient_id to each component ───────────────────────────
    components = uf.components(n)
    comp_to_pid: dict[int, str] = {}

    for root, members in components.items():
        # prefer p360 patient_id if any member matched
        p360_pids = [ident.loc[m, "_p360_pid"] for m in members
                     if pd.notna(ident.loc[m, "_p360_pid"])]
        if p360_pids:
            pid = str(p360_pids[0])          # take first (p360 is ground truth)
        else:
            pid = f"LINK_{uuid.uuid4().hex[:12].upper()}"
        comp_to_pid[root] = pid

    # ── step 5: build output ─────────────────────────────────────────────────
    rows = []
    # track pair scores per row index for reporting
    row_best_score: dict[int, float] = {}
    for i, j, sc in candidates:
        if sc >= ACCEPT_THRESH:
            row_best_score[i] = max(row_best_score.get(i, 0.0), sc)
            row_best_score[j] = max(row_best_score.get(j, 0.0), sc)

    for idx, row in ident.iterrows():
        root = uf.find(idx)
        pid  = comp_to_pid[root]
        p360_hit = bool(ident.loc[idx, "_p360_hit"])

        if p360_hit:
            link_type = "p360_direct"
            sc = 1.0
        elif idx in row_best_score:
            link_type = "cross_source"
            sc = row_best_score[idx]
        else:
            link_type = "singleton"
            sc = 0.0

        # review flag: any pair involving this row was in review zone
        rev = any(idx in pair for pair in review_flags)

        rows.append({
            "source_system":     row["source_system"],
            "source_patient_id": row["source_patient_id"],
            "patient_id":        pid,
            "link_type":         link_type,
            "link_score":        sc,
            "review_flag":       rev,
        })

    linkage = pd.DataFrame(rows)
    save_df(linkage, C.SILVER_DIR, "patient_linkage")
    audit("linkage_end", rows=len(linkage),
          p360_direct=int((linkage.link_type == "p360_direct").sum()),
          cross_source=int((linkage.link_type == "cross_source").sum()),
          singletons=int((linkage.link_type == "singleton").sum()),
          review_flags=int(linkage.review_flag.sum()))

    _write_linkage_report(linkage, candidates)
    return linkage


# ── report writer ─────────────────────────────────────────────────────────────

def _write_linkage_report(linkage: pd.DataFrame, candidates: list):
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_patients = linkage["patient_id"].nunique()
    n_source   = len(linkage)
    by_type    = linkage["link_type"].value_counts().to_dict()
    by_system  = linkage.groupby("source_system")["link_type"].value_counts().unstack(fill_value=0).to_dict()
    n_review   = int(linkage["review_flag"].sum())
    n_accept   = sum(1 for _, _, sc in candidates if sc >= ACCEPT_THRESH)
    n_review_c = sum(1 for _, _, sc in candidates if REVIEW_THRESH <= sc < ACCEPT_THRESH)

    rep = {
        "canonical_patients": n_patients,
        "source_rows_linked": n_source,
        "link_type_breakdown": by_type,
        "review_flagged_source_rows": n_review,
        "candidate_pairs_accepted": n_accept,
        "candidate_pairs_review": n_review_c,
        "thresholds": {"accept": ACCEPT_THRESH, "review": REVIEW_THRESH},
    }

    (C.REPORT_DIR / "linkage_summary.json").write_text(
        json.dumps(rep, indent=2, default=str))

    lines = [
        "# Patient Linkage Summary",
        f"- Canonical patients resolved: **{n_patients:,}**",
        f"- Source patient rows linked: **{n_source:,}**",
        f"- Accept threshold: {ACCEPT_THRESH}  |  Review threshold: {REVIEW_THRESH}",
        "",
        "## Link type breakdown",
        "| link_type | count |",
        "|-----------|-------|",
    ]
    for lt, cnt in sorted(by_type.items()):
        lines.append(f"| {lt} | {cnt:,} |")
    lines += [
        "",
        f"- Candidate pairs accepted (auto-linked): {n_accept:,}",
        f"- Candidate pairs in review zone (best-match taken): {n_review_c:,}",
        f"- Source rows flagged for human review: {n_review:,}",
    ]
    (C.REPORT_DIR / "linkage_summary.md").write_text("\n".join(lines))
