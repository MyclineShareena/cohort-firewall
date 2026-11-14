# Project: Cohort Firewall — Oncology Dataset Readiness Checker

## Context
One-day Claude Build Day hackathon (theme: responsible healthcare AI),
Saturday Nov 14. Projects due 3:15 PM with a LIVE demo, so a working
prototype beats features. I'm a data scientist (Python, pandas,
Streamlit, FastAPI).

## Goal
Given an oncology dataset (CSV) and a proposed AI question in plain
English, return one verdict — Ready, Conditional, Blocked, or
Insufficient Evidence — with evidence and concrete next actions.

## Core design principle (non-negotiable)
Rules decide; Claude interprets and explains.
- Claude parses the question into structured requirements (JSON).
- Deterministic Python checks evaluate the dataset.
- A rules engine assigns the verdict. Claude NEVER assigns it.
- Claude explains using ONLY the check results passed to it.

## Hard constraints
- Synthetic data only. Never use or request real patient data.
- Assesses data readiness only. Never makes clinical decisions.
- Model from ANTHROPIC_MODEL env var, key from ANTHROPIC_API_KEY.
  Never hard-code either.

## Stack and environment
Python 3.14 (works; all tests pass), local .venv, pandas, numpy,
streamlit, anthropic SDK, pytest. Regenerate data with
`.venv/bin/python -m data.generate`.

## Grounding
- Checks are labeled by METRIC framework clusters (Schwabe et al.,
  npj Digit Med 2024): measurement process, timeliness,
  representativeness, informativeness, consistency.
- Sample size follows Riley et al., BMJ 2020 (doi 10.1136/bmj.m441);
  binary worked examples in Riley et al., Stat Med 2018 Part II
  (doi 10.1002/sim.7992).

## Structure
app.py, parser.py, checks.py, sample_size.py, rules.py, explainer.py,
prompts.py, data/generate.py, experiments/, tests/

## Requirements JSON (parser output)
target_population (column/operator/value filters), outcome (column,
type: binary|time_to_event), required_fields, time_horizon_months,
task_type (prediction|description|comparison),
n_predictor_parameters (k-level categorical = k-1 parameters),
unmappable_terms. Validate; on failure retry once, then return
Insufficient Evidence.

## Outcome derivation (already in checks.py: derive_binary_outcome)
Binary outcome derived from vital_status and dates. Patients alive with
follow-up shorter than the horizon have UNKNOWN outcome: exclude from
outcome counts and report how many were excluded.

## Checks
Each returns {name, metric_cluster, severity: hard|soft,
status: pass|fail|unknown|not_applicable, value, threshold, message}.
not_applicable is neutral for the verdict (treated like pass by rules.py)
but is NEVER shown or described as passed: UI shows it gray; the explainer
must say which checks were not applicable and why.
9. Consistency: question terms mappable (unknown when unmappable_terms is
   non-empty). unmappable_terms are never added to required_fields.
1. Consistency: required fields exist with expected types.
2. Informativeness: missingness, REQUIRED FIELDS ONLY
   (soft fail >20%, hard fail >40%).
3. Informativeness: outcome present and derivable (hard).
4. Timeliness: early censoring (soft fail >15% excluded).
5. Representativeness: subgroups (sex, age band, stage) above minimum
   counts.
6. Measurement process: label leakage, e.g. date_of_death when
   predicting survival (hard).
7. Sample size: Riley criteria (SOFT fail) plus EPV >= 10 shown for
   comparison.
8. Safety: columns resembling real identifiers (hard).

## Sample size (sample_size.py), binary outcome
- Criterion (i): n = P / ((S - 1) * ln(1 - R2cs / S)), S = 0.9,
  P = candidate predictor parameters.
- Criterion (iii): n = (1.96 / 0.05)^2 * phi * (1 - phi).
- Required n = max of criteria; report events needed (n * phi).
- Anticipated R2cs is an input; default = 15% of max R2cs, stated in
  output.
- Unit tests must reproduce a published worked example. Flag anything
  uncertain; I'll verify against the paper.

## Verdict rules (rules.py)
Any hard fail -> Blocked; else any unknown -> Insufficient Evidence;
else any soft fail -> Conditional; else Ready. Return verdict plus the
checks that drove it.
A verdict only ever describes the data. Missing API setup returns status
"setup_required", and an unreachable API returns "api_unavailable"; neither
has a verdict. The UI message for both is "Reading a new question needs live
Claude. The six demo questions work offline."; technical detail (env var
names, API error) lives only in the result's "detail" field and the README.
Offline demo mode: if the API is unavailable and file content + question
exactly match a demo case, use its hand-written requirements, labeled
"cached parse (offline)". Never for non-demo inputs.

## Demo files (Milestone 1 done, commit e9ab7ff)
- clean.csv: 2000 rows, 560 events, 60 excluded -> Ready
- messy.csv: 600 rows, 40 events, 50 excluded, stage 30% missing
  -> Conditional
- broken.csv: no vital_status -> Blocked
- leakage.csv: clean + date_of_death -> Blocked BY THE LEAKAGE CHECK
- epv_vs_riley.csv: 1000 rows, 100 events, 0 excluded, 10 parameters
  -> passes EPV, fails Riley, NO other failures -> Conditional
- Insufficient Evidence demo: clean.csv with a question about a field
  that doesn't exist (e.g. PD-L1 status).

## Pending fixes (do before Milestone 2)
1. Generator: make stage I and II death rates differ (~7% and ~15%),
   keeping all event and censoring counts exactly as listed above.
   Regenerate and recommit.
2. Plan tests that assert the DRIVING CHECK for every demo file, not
   just the verdict.

## UI (app.py)
CSV upload or demo-file picker, question box, optional anticipated R2
and predictor-parameter override, Run button. Show a color-coded
verdict card, checks table grouped by METRIC cluster, Riley vs EPV
comparison, Claude's explanation and next actions, and the note
"Synthetic data only. Not for clinical decision-making."

## Milestones (one at a time; stop and show me after each)
1. DONE: scaffold, generator, demo CSVs.
2. checks.py + sample_size.py + rules.py with tests.
3. parser.py + explainer.py with prompts.
4. Streamlit UI end to end on all demo files.
5. Bonus: experiment comparing Claude-alone verdicts vs rules engine
   on 10-15 scenarios with known verdicts.

## Working style
Propose a plan before each milestone. No features beyond this spec
unless asked. Run tests after every change; never leave the app
broken. Commit after each milestone. Ask when something is ambiguous.
