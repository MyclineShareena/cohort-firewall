# Cohort Firewall

Oncology dataset readiness checker. **Synthetic data only. Not for clinical decision-making.**

Rules decide the verdict (Ready / Conditional / Blocked / Insufficient Evidence);
Claude only parses the question and explains the deterministic check results.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...  ANTHROPIC_MODEL=...
.venv/bin/python -m data.generate   # writes data/*.csv
.venv/bin/pytest -q
```
