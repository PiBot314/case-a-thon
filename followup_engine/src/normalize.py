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
