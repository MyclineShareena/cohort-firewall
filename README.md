# Cohort Firewall

Oncology dataset readiness checker. **Synthetic data only. Not for clinical decision-making.**

Rules decide the verdict (Ready / Conditional / Blocked / Insufficient Evidence);
Claude only parses the question and explains the deterministic check results.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.generate   # writes data/*.csv
.venv/bin/pytest -q                 # offline tests (live tests excluded)
.venv/bin/streamlit run app.py
```

## Live Claude
Create `.env` in the repo root (it is gitignored; never commit it):
```
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```
Shell environment variables take precedence over `.env`. The app re-reads `.env`
on every run, so adding a key switches it to live without a restart.
Live tests: `.venv/bin/pytest -m live`.

## Offline mode
Without a key (or if the API is unreachable), the six demo cases still run:
their hand-written requirements are used as a "cached parse (offline)" and the
explanation is built from the checks' own messages. Any other question shows
"Reading a new question needs live Claude. The six demo questions work offline."
and no verdict. The result data keeps the technical reason in `detail`
(`Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL`, or the API error type), with status
`setup_required` or `api_unavailable`.
