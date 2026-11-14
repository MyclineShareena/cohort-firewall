## Team
- MyclineShareena John Peter Kennedy
- Vinaya Rekha Avadhanam
- Nithya Sarvepalli

**Track:** Track 1 · Cohort Firewall
**Demo video:** https://www.youtube.com/watch?v=Lpp6ocGcg_o
**Data:** Synthetic data only (approved by the organizers for this project)

# Cohort Firewall

Oncology dataset readiness checker. **Synthetic data only. Not for clinical decision-making.**

Rules decide the verdict (Ready / Conditional / Blocked / Insufficient Evidence);
Claude only parses the question and explains the deterministic check results.

Plain-English guide to the project and every file: [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md).

## Setup
Tested with Python 3.14.
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.generate   # writes data/*.csv
.venv/bin/pytest -q                 # offline tests (live tests excluded)
.venv/bin/streamlit run app.py
```

## What to try (works offline)
Pick a demo case in the sidebar and click Run. Expected verdicts:

| Demo case | Verdict | Driving check(s) |
|---|---|---|
| clean.csv | Ready | none |
| messy.csv | Conditional | required_field_missingness, sample_size |
| broken.csv | Blocked | required_fields_present, outcome_derivable |
| leakage.csv | Blocked | label_leakage |
| epv_vs_riley.csv | Conditional | sample_size (passes EPV, fails Riley) |
| clean.csv + PD-L1 question | Insufficient Evidence | question_terms_mappable |

Then submit a human review and download the evidence record (JSON or Markdown).

## Supported questions
Death within a time horizon (binary, from vital_status and dates), survival
time (time-to-event), and descriptive questions with no outcome. Any other
outcome is reported as Insufficient Evidence rather than guessed.

## Live Claude
Create `.env` in the repo root (it is gitignored; never commit it):
```
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=...
```
Shell environment variables take precedence over `.env`. The app re-reads `.env`
on every run, so adding a key switches it to live without a restart.
Live tests: `.venv/bin/pytest -m live`.

Only a column-level profile is sent to Claude: column names, kinds, missing
fractions, category levels and each numeric column's min and max. No other row
values are sent.

## Offline mode
Without a key (or if the API is unreachable), the six demo cases still run:
their hand-written requirements are used as a "cached parse (offline)" and the
explanation is built from the checks' own messages. Any other question shows
"Reading a new question needs live Claude. The six demo questions work offline."
and no verdict. The result data keeps the technical reason in `detail`
(`Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL`, or the API error type), with status
`setup_required` or `api_unavailable`.

## Milestone 5 (bonus): Claude-alone vs rules engine
Measures how often Claude, judging dataset readiness on its own with no
checks or rules engine, agrees with the actual rules-engine verdict across 13
scenarios. See [experiments/README.md](experiments/README.md) to run it (needs
a live API key).

## Tests
`.venv/bin/pytest -q` passes offline: 278 passed, 1 skipped, 6 deselected
(live-API tests, run explicitly with `-m live`).
