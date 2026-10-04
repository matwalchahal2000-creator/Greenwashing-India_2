import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path(__file__).resolve().parent.parent / "data"

# ---------- sentences & environmental filter ----------
def load_keywords(path=None):
    df = pd.read_csv(path or DATA / "env_keywords.csv")
    return [k.strip().lower() for k in df["keyword"].dropna()]


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"-\n(?=[a-z])", "", text))


_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z0-9(\"“‘])")


def split_sentences(text, min_words=5, max_words=150):
    return [s.strip() for s in _SPLIT.split(clean(text)) if min_words <= len(s.split()) <= max_words]


def env_sentences(sentences, keywords):
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in keywords) + r")", re.I)
    return [s for s in sentences if pat.search(s)]


# ---------- specificity (rule-based, India-adapted) ----------
STATES = ("Andhra Pradesh|Assam|Bihar|Chhattisgarh|Delhi|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|"
          "Karnataka|Kerala|Madhya Pradesh|Maharashtra|Odisha|Punjab|Rajasthan|Tamil Nadu|Telangana|"
          "Uttar Pradesh|Uttarakhand|West Bengal|Mumbai|Chennai|Pune|Hyderabad|Bengaluru|Kolkata|"
          "Jamnagar|Mundra|Dahej")
_YEAR = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2,4})?\b")
_REF = re.compile(r"\b(?:Principle|Section|Question|Annexure|Note|Table|Figure|Para)\s*\d+\b", re.I)
_PATTERNS = {
    "has_number": re.compile(r"\d"),
    "has_currency": re.compile(r"₹|\bRs\.?\s?\d|\bINR\b|\bUSD\b|\$|\bcrore\b|\blakh\b", re.I),
    "has_unit": re.compile(r"%|\b(?:tCO2e?|MtCO2e|kL|MLD|MW|GW|MWh|GWh|kWh|GJ|TJ|tonnes?|MT|kg)\b"),
    "has_regulator": re.compile(r"\b(?:CPCB|SPCB|MoEFCC|NGT|BEE|ISO ?14001|ISO ?50001|GRI|SBTi|TCFD|CDP|EIA|ZLD|ETP)\b"),
    "has_location": re.compile(rf"\b(?:{STATES})\b"),
    "has_entity": re.compile(r"(?<=[a-z,;] )[A-Z][A-Za-z&.]+(?: (?:of |and )?[A-Z][A-Za-z&.]+)+"),
}


def specificity_flags(s):
    no_years = _REF.sub(" ", _YEAR.sub(" ", s))  # years / 'Principle 6' are not data
    f = {k: bool(p.search(no_years if k == "has_number" else s)) for k, p in _PATTERNS.items()}
    f["specific"] = any(f.values())
    return f


# ---------- novelty vs. the firm's previous filing ----------
def novelty_vs_prior(curr, prior):
    """Per sentence: cosine to nearest prior sentence and number of differing words."""
    if not prior or not curr:
        return [(np.nan, np.nan)] * len(curr)
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(curr + prior)
    sim = (vec.transform(curr) @ vec.transform(prior).T).toarray()
    out = []
    for i, s in enumerate(curr):
        j = int(sim[i].argmax())
        a, b = s.lower().split(), prior[j].lower().split()
        matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
        out.append((float(sim[i, j]), max(len(a), len(b)) - matched))
    return out


def score_sentences(sents, prior_sents=None):
    nov = novelty_vs_prior(sents, prior_sents)
    rows = [{"sentence": s, **specificity_flags(s), "nn_cosine": c, "word_diff": w}
            for s, (c, w) in zip(sents, nov)]
    return pd.DataFrame(rows)


def classify(df, mode="paper", cos_thr=0.80, wd_thr=2):
    """Apply novelty rule -> novel / useful (specific or novel) / fluff (general and repetitive)."""
    if df.empty:
        return df
    df = df.copy()
    if mode == "paper":
        df["novel"] = np.where(df["word_diff"].isna(), np.nan, (df["word_diff"] >= wd_thr).astype(float))
    else:
        df["novel"] = np.where(df["nn_cosine"].isna(), np.nan, (df["nn_cosine"] < cos_thr).astype(float))
    df["useful"] = df["specific"] | (df["novel"] == 1)
    df["fluff"] = (~df["specific"]) & (df["novel"] == 0)
    return df


def filing_metrics(df):
    nan = float("nan")
    n = len(df)
    m = dict(quantity=n, specific=0, general=0, specificity_ratio=nan, novel=nan, repetitive=nan,
             novelty_ratio=nan, useful=nan, fluff=nan, fluff_ratio=nan, signal_to_noise=nan)
    if n == 0:
        return m
    m["specific"] = int(df["specific"].sum())
    m["general"] = n - m["specific"]
    m["specificity_ratio"] = m["specific"] / n
    if df["novel"].notna().any():  # needs a prior filing (first filing per firm/doc type has none)
        nv, us, fl = int((df["novel"] == 1).sum()), int(df["useful"].sum()), int(df["fluff"].sum())
        m.update(novel=nv, repetitive=n - nv, novelty_ratio=nv / n, useful=us, fluff=fl,
                 fluff_ratio=fl / n, signal_to_noise=us / fl if fl else nan)
    return m
