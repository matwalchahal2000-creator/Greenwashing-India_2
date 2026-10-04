import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.extract import parse_name, pdf_to_text
from engine.textproc import classify, env_sentences, filing_metrics, load_keywords, score_sentences, split_sentences

st.set_page_config(page_title="Greenwash-India", layout="wide")
st.title("Environmental disclosure quality — Indian filings")
st.caption("Specificity · novelty · proportionality, adapted from Liu et al. (2026), Journal of Business Ethics")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Novelty rule", ["paper", "similarity"],
                    format_func=lambda m: "Paper: ≥ N words differ" if m == "paper" else "TF-IDF cosine < threshold")
    wd_thr = st.number_input("N (word-difference threshold)", 1, 10, 2)
    cos_thr = st.slider("Cosine threshold", 0.50, 0.99, 0.80, 0.01)
    kw_file = st.file_uploader("Custom keyword CSV (column: keyword)", type="csv")
params = dict(mode=mode, cos_thr=cos_thr, wd_thr=wd_thr)
keywords = load_keywords(kw_file)


def collect_files(source):
    if source == "Upload files":
        ups = st.file_uploader("Name files Firm_Year_DocType.pdf, e.g. TCS_2023_BRSR.pdf",
                               type=["pdf", "txt"], accept_multiple_files=True)
        return [(u.name, u.getvalue()) for u in ups]
    folder = st.text_input("Root folder (one subfolder per firm)")
    if folder and Path(folder).is_dir():
        paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in (".pdf", ".txt")]
        st.write(f"{len(paths)} files found")
        return [(p.name, p) for p in paths]
    return []


def run(files):
    docs = []
    for name, src in files:
        meta = parse_name(name)
        if meta:
            docs.append((*meta, name, src))
        else:
            st.warning(f"Skipped {name}: expected Firm_Year_DocType")
    docs.sort(key=lambda d: (d[0], d[2], d[1]))
    raw, prior, bar = {}, {}, st.progress(0.0)
    for i, (firm, year, dtype, name, src) in enumerate(docs):
        try:
            data = src if isinstance(src, bytes) else src.read_bytes()
            text = data.decode("utf-8", "ignore") if name.lower().endswith(".txt") else pdf_to_text(io.BytesIO(data))
        except Exception as e:
            st.warning(f"Skipped {name}: {e}")
            continue
        sents = env_sentences(split_sentences(text), keywords)
        raw[(firm, year, dtype)] = score_sentences(sents, prior.get((firm, dtype)))  # prior = previous year, same doc type
        prior[(firm, dtype)] = sents
        bar.progress((i + 1) / len(docs))
    return raw


def firm_tables(raw, firm):
    keys = [k for k in raw if k[0] == firm]
    sent = pd.concat([classify(raw[k], **params).assign(firm=k[0], year=k[1], doctype=k[2]) for k in keys],
                     ignore_index=True)
    met = pd.DataFrame([{"firm": k[0], "year": k[1], "doctype": k[2], **filing_metrics(classify(raw[k], **params))}
                        for k in keys])
    return sent, met


tab1, tab2, tab3 = st.tabs(["Score documents", "Trends & performance link", "Methodology"])

with tab1:
    source = st.radio("Source", ["Upload files", "Local folder"], horizontal=True)
    files = collect_files(source)
    if st.button("Run", type="primary", disabled=not files):
        st.session_state["raw"] = run(files)
    raw = st.session_state.get("raw")
    if raw:
        firms = sorted({k[0] for k in raw})
        tabs = {f: firm_tables(raw, f) for f in firms}
        metrics = pd.concat([m for _, m in tabs.values()], ignore_index=True)
        st.session_state["metrics"] = metrics
        st.dataframe(metrics)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f, (s, m) in tabs.items():
                z.writestr(f"{f}/{f}_sentences.csv", s.to_csv(index=False))
                z.writestr(f"{f}/{f}_filing_metrics.csv", m.to_csv(index=False))
        st.download_button("Download per-firm files (zip)", buf.getvalue(), "greenwash_india_output.zip")
        if source == "Local folder":
            out_dir = st.text_input("Output folder for per-firm CSVs")
            if st.button("Write to folder", disabled=not out_dir):
                for f, (s, m) in tabs.items():
                    d = Path(out_dir) / f
                    d.mkdir(parents=True, exist_ok=True)
                    s.to_csv(d / f"{f}_sentences.csv", index=False)
                    m.to_csv(d / f"{f}_filing_metrics.csv", index=False)
                st.success(f"Written to {out_dir}")
        pick = st.selectbox("Inspect sentences for", firms)
        st.dataframe(tabs[pick][0])

with tab2:
    m = st.session_state.get("metrics")
    if m is None:
        st.info("Run scoring first.")
    else:
        st.line_chart(m.groupby("year")[["specificity_ratio", "novelty_ratio", "fluff_ratio"]].mean())
        st.caption("Cross-firm means by year. Novelty/fluff need a prior filing of the same firm and document type.")
        perf = st.file_uploader("Performance CSV (columns: firm, year, plus numeric variables such as "
                                "emissions intensity or controversy count)", type="csv")
        if perf:
            p = pd.read_csv(perf)
            col = st.selectbox("Performance variable", [c for c in p.columns if c not in ("firm", "year")])
            y = st.selectbox("Text measure", ["fluff_ratio", "specificity_ratio", "novelty_ratio", "signal_to_noise"])
            j = m.merge(p, on=["firm", "year"]).dropna(subset=[col, y])
            st.scatter_chart(j, x=col, y=y)
            if len(j) > 2:
                st.write(f"Pearson r = {j[col].corr(j[y]):.3f}  (n = {len(j)})")
            st.caption("Descriptive only: a talk–walk gap needs controls, firm/industry-year effects and clustered SEs.")

with tab3:
    st.markdown("""
**Unit:** environmental sentences (keyword match, editable in `data/env_keywords.csv`).

**Specificity:** a sentence is *specific* if it has a number (years and 'Principle 6'-type references ignored), ₹/crore/lakh,
a unit (tCO₂e, kL, MW, %...), an Indian regulator/standard (CPCB, NGT, ISO 14001...), a state/city/site, or a named entity.

**Novelty:** each sentence is compared with the firm's nearest sentence in its previous filing of the same document type.
*Paper rule:* novel if ≥ N words differ. *Similarity rule:* novel if TF-IDF cosine < threshold.

**Proportionality:** *useful* = specific or novel; *fluff* = general and repetitive;
`signal_to_noise = useful / fluff` (a monotone transform of `fluff_ratio`, so report one of them).

**Not measured:** truthfulness, sincerity or actual environmental performance. Greenwashing requires linking these
text measures to performance (Tab 2). Strip BRSR question text before scoring; scanned PDFs need OCR.
""")
