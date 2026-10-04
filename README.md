# Greenwash-India

Streamlit app scoring environmental disclosure quality (specificity, novelty, proportionality) in Indian
company filings (BRSR, BRR, annual report/MD&A), adapted from Liu et al. (2026, *J. Business Ethics*).

## Input convention
`Firm_Year_DocType.pdf` (or `.txt`), one subfolder per firm, e.g. `TCS/TCS_2023_BRSR.pdf`.
Novelty compares each filing with the firm's previous year of the same DocType; the first year has no novelty.

## Run locally (direct folder access)
    pip install -r requirements.txt
    streamlit run app.py

## Deploy from GitHub
1. `git init && git add . && git commit -m "init"`, create a GitHub repo, `git push`.
2. Go to share.streamlit.io, sign in with GitHub, **Create app**, pick the repo/branch, main file `app.py`, Deploy.
3. On the cloud, use *Upload files* (size/memory limits apply; run large annual reports locally).

## Output
Per firm: `<Firm>/<Firm>_sentences.csv` and `<Firm>/<Firm>_filing_metrics.csv` (zip download, or written to a local folder).
Concatenate the metrics files for panel regressions.

## Extend
Edit `data/env_keywords.csv`; add states/regulators in `engine/textproc.py`; merge performance data in Tab 2.
