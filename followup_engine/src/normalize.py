"""Field normalisers making noisy source identities linkage-ready (no linkage done here)."""
import re, unicodedata
import pandas as pd

_TITLES = re.compile(r"\b(mr|mrs|ms|smt|shri|sri|dr)\b\.?", re.I)

def norm_name(s):
    if pd.isna(s): return pd.NA
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = _TITLES.sub(" ", s.lower())
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or pd.NA

def norm_phone(s):
    if pd.isna(s): return pd.NA
    d = re.sub(r"\D", "", str(s))
    if len(d) > 10 and d.startswith(("91", "0")): d = d[-10:]
    return d if len(d) == 10 else pd.NA

def norm_village(s):
    if pd.isna(s): return pd.NA
    s = re.sub(r"[^a-z ]", " ", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s or pd.NA

def norm_gender(s):
    if pd.isna(s): return pd.NA
    s = str(s).strip().upper()[:1]
    return s if s in ("M", "F", "O") else pd.NA


# ── linkage helpers (added for stage 2: patient construction) ────────────────
_PHON_SUBS = [("ck", "k"), ("ph", "f"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
              ("gh", "g"), ("sh", "s"), ("ch", "c"), ("aa", "a"), ("ee", "i"), ("oo", "u"),
              ("ou", "u"), ("w", "v"), ("z", "j"), ("q", "k"), ("x", "ks"), ("y", "i")]

def phonetic_token(tok):
    """Cheap Indic-friendly phonetic key: mohan/mohn -> mhn, laxman/lakshman -> lksmn."""
    t = str(tok).lower()
    for a, b in _PHON_SUBS: t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)                       # collapse doubled letters
    return t[:1] + re.sub(r"[aeiou]", "", t[1:])          # keep first letter, drop later vowels

def phonetic_name(s):
    """Order-insensitive phonetic key of a normalised name (sorted token keys)."""
    if pd.isna(s) or not str(s).strip(): return pd.NA
    return " ".join(sorted(phonetic_token(t) for t in str(s).split()))

def jaro_winkler(s1, s2):
    """Jaro-Winkler; uses rapidfuzz when installed, else pure python."""
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.similarity(s1, s2)
    except ImportError:
        pass
    if s1 == s2: return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    md = max(max(l1, l2) // 2 - 1, 0)
    m1, m2 = [False] * l1, [False] * l2
    m = 0
    for i in range(l1):
        for j in range(max(0, i - md), min(i + md + 1, l2)):
            if not m2[j] and s1[i] == s2[j]:
                m1[i] = m2[j] = True; m += 1; break
    if not m: return 0.0
    k = t = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]: k += 1
            if s1[i] != s2[k]: t += 1
            k += 1
    jaro = (m / l1 + m / l2 + (m - t / 2) / m) / 3
    p = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b: break
        p += 1
    return jaro + p * 0.1 * (1 - jaro)

def standardize_villages(df, geo, col="village_norm", min_sim=0.86):
    """Repair village spellings against geography_reference.
    Prefers villages inside the record's own (district, block); falls back to the whole
    reference. Returns (village_std Series, village_repaired bool Series). Unrepairable -> keeps input."""
    vocab_all = sorted(geo["village_norm"].dropna().unique())
    by_blk = {k: sorted(g["village_norm"].dropna().unique()) for k, g in geo.groupby(["district", "block"])}
    cache, out, fixed = {}, [], []
    known = set(vocab_all)
    for v, d, b in zip(df[col], df.get("district", pd.Series(pd.NA, index=df.index)),
                       df.get("block", pd.Series(pd.NA, index=df.index))):
        if pd.isna(v): out.append(pd.NA); fixed.append(False); continue
        if v in known: out.append(v); fixed.append(False); continue
        key = (v, d, b)
        if key not in cache:
            best = None
            for vocab in (by_blk.get((d, b), []), vocab_all):
                sc = [(jaro_winkler(v, w), w) for w in vocab]
                if sc:
                    s, w = max(sc)
                    if s >= min_sim: best = w; break
            cache[key] = best
        r = cache[key]
        out.append(r if r else v); fixed.append(r is not None)
    return pd.Series(out, index=df.index, dtype="object"), pd.Series(fixed, index=df.index)
