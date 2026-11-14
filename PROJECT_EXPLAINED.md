# Cohort Firewall, explained simply

## The big idea

Imagine a hospital has a big spreadsheet about cancer patients: age, cancer stage, blood test results, whether each patient is alive or has died, and so on. A researcher says: *"I want to train an AI that predicts which patients will die within 2 years."*

Before anyone builds that AI, someone should ask: **is this spreadsheet actually good enough for that job?** Are columns missing? Is too much data blank? Are there enough patients? Is anything in it "cheating"?

Cohort Firewall is the **safety inspector** that answers that question. You give it:
1. a spreadsheet (a CSV file), and
2. the question in plain English.

It gives back **one verdict**:

| Verdict | Meaning | Like a car inspection... |
|---|---|---|
| 🟢 **Ready** | No problems found | Passed |
| 🟡 **Conditional** | Usable, but with warnings | Passed, but your tyres are worn |
| 🔴 **Blocked** | A serious problem | Failed, brakes don't work |
| ⚪ **Insufficient Evidence** | Can't tell | Couldn't test, the part is missing |

It also says **why** it gave that verdict and **what you'd need to fix**.

## The most important rule: "Rules decide, Claude explains"

The project uses Claude (an AI), but **the AI is never allowed to decide the verdict**. Think of a football match:

- **Claude is the translator.** It turns your English question into a precise checklist the computer can follow.
- **The checks are the measuring tape.** Plain Python code measures the data. No AI is involved, so the answer is the same every time.
- **The rules engine is the referee.** It follows four fixed rules to pick the verdict.
- **Claude is also the commentator.** Afterwards it explains the result in plain words, using *only* what the checks found.

This matters because AI can make things up. A referee that follows fixed rules can't.

Two other ground rules:
- **Fake data only.** Every spreadsheet in the project is made up (synthetic). No real patients.
- **It judges data, not people.** It never says anything about a patient's treatment or health.

## What happens when you click "Run"

```
Your question ──► Claude turns it into a checklist (JSON)
                          │
Your spreadsheet ──► 9 checks measure the data
                          │
                   Rules engine picks the verdict
                          │
                   Claude explains it in plain English
                          │
                   Web page shows everything
                   + a human reviewer signs off
```

If there's no internet or API key, the 6 demo examples still work. They use a **pre-written checklist** instead of asking Claude (offline mode).

## How to run it

See [README.md](README.md). In short:

```bash
.venv/bin/streamlit run app.py      # start the web app
.venv/bin/pytest                    # run the offline tests
```

For live Claude, put `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` in a `.env` file (never commit it). Without them, the six demo cases still run offline.

---

# Every file, one by one

## Project notes and setup

**`CLAUDE.md`**: the project's instruction manual. It records the goal, the hard rules (fake data only, the AI never decides), what each check must do, the verdict rules and the milestones. Claude Code reads it at the start of every session.

**`README.md`**: the "how to run it" guide. It covers installing, starting the app, adding an API key in a `.env` file for live mode, and how offline mode works.

**`requirements.txt`**: the shopping list of Python libraries:
- `pandas`/`numpy` for spreadsheets and maths,
- `streamlit` for the web page,
- `anthropic` to talk to Claude,
- `pydantic` to check the shape of Claude's answers,
- `python-dotenv` to read the `.env` file,
- `pytest` for tests.

**`pytest.ini`**: test settings. Its key line, `-m "not live"`, means tests that would call the real Claude API (and cost money) are skipped unless you ask for them.

**`.gitignore`**: files git must never upload. The most important one is `.env`, which holds the secret API key.

**`experiments/.gitkeep`**: an empty placeholder folder for the bonus experiment, which compares Claude-alone verdicts with the rules engine.

---

## The data

**`data/generate.py`**: the **fake-patient factory**. Its main function, `make_cohort(...)`, makes up patients with:
- age, sex and cancer stage (I–IV),
- ECOG (how well the patient manages day to day),
- histology (cancer type),
- blood tests (albumin, LDH, hemoglobin, WBC),
- diagnosis date, last check-up date, and alive/dead.

It uses a fixed random "seed", so it makes exactly the same patients every time. It controls numbers precisely. For example, clean.csv has exactly 560 deaths within 24 months, and stage I patients die less often (~7%) than stage II (~15%). Run `.venv/bin/python -m data.generate` to rebuild all the CSVs.

**The five CSVs**: each is built to trigger one specific verdict, like test questions with known answers.

| File | What's special about it | Verdict |
|---|---|---|
| `clean.csv` | 2,000 good patients | Ready |
| `messy.csv` | Only 600 patients, and 30% of "stage" is blank | Conditional |
| `broken.csv` | The alive/dead column is deleted | Blocked |
| `leakage.csv` | Has a `date_of_death` column. That's cheating: the answer is hidden in the data | Blocked |
| `epv_vs_riley.csv` | 1,000 patients: enough by an old rule of thumb, not enough by the stricter modern formula | Conditional |

**`data/demo_cases.py`**: the **six demo scenarios**: the five files above plus one trick (clean.csv with a question about "PD-L1 status", a column that doesn't exist, which gives Insufficient Evidence). For each it stores:
- the exact question,
- the **hand-written checklist** (what Claude *should* produce, used in offline mode),
- the expected verdict and **which check should cause it**.

The tests use it as the answer key.

---

## The brain: four core files

### `checks.py`: the measuring tapes

This holds **9 checks**. Each returns the same shape of result:
- name, and which quality category it belongs to (the METRIC framework, Schwabe et al., npj Digital Medicine 2024),
- severity: **hard** (serious) or **soft** (warning),
- status: **pass / fail / unknown / not_applicable**,
- the measured value, the threshold, and a message.

The nine checks:
1. **`required_fields_present`**: do the needed columns exist, with sensible values? For example, is age a number? *(hard)*
2. **`question_terms_mappable`**: does everything in the question match a column? "PD-L1 status" doesn't, so the result is unknown.
3. **`required_field_missingness`**: how many blanks? Over 20% is a soft fail; over 40% is a hard fail.
4. **`outcome_derivable`**: can we work out who died within 24 months from the alive/dead column and the dates? *(hard)*
5. **`early_censoring`**: how many patients were followed for too short a time to know? Say someone was last seen alive at month 6. We can't know if they died by month 24, so they're excluded. More than 15% excluded is a warning.
6. **`subgroup_counts`**: does every group (men/women, each age band, each stage) have at least 30 patients? Otherwise the AI would know nothing about small groups.
7. **`label_leakage`**: is there a column that gives the answer away, like `date_of_death`? That's like taking an exam with the answer sheet attached. *(hard)*
8. **`sample_size`**: are there enough patients? It uses `sample_size.py`.
9. **`identifier_columns`**: does anything look like real personal info (names, social security numbers, phone numbers, emails, ZIP codes)? Patient IDs must look like `SYN-12345`. *(hard)*

Helper functions:
- **`derive_binary_outcome`**: decides for each patient whether they died within the time window (1), survived past it (0) or are unknown.
- **`apply_population`**: filters to the patients the question is about, e.g. "only stage III".
- **`run_checks`**: runs all 9 checks in order.

"**not_applicable**" means the check doesn't apply to this question. For example, leakage only matters when predicting death. It never counts against you, but it's never shown as "passed" either. It appears in grey.

### `sample_size.py`: "do we have enough patients?"

This uses a formula from Riley et al. (BMJ 2020), a well-known research paper. The idea: the more things you feed an AI (predictors), the more patients it needs, or it just memorises noise. It computes three criteria and takes the biggest:
- **(i)** the model mustn't be over-confident (shrinkage ≥ 0.9),
- **(ii)** the model's apparent score mustn't be inflated,
- **(iii)** the overall death rate must be estimated precisely (±5%).

It also computes the old rule of thumb, **EPV** ("10 deaths per predictor"), for comparison only. The demo case `epv_vs_riley` shows the difference: EPV says "fine" (100 deaths ÷ 10 = 10), but Riley says you need 1,205 patients and you have 1,000.

- **`ceil_count`** rounds "events needed" **up**, because you can't have half a patient, and it avoids a computer-maths glitch (7.000000001 shouldn't become 8).
- **R²cs** is a guess at how good the model will be. If you don't give one, it uses 15% of the maximum, and the page says so.

### `rules.py`: the referee

This is the shortest file and the most important. It checks four rules in order:
1. Any **hard fail**? → **Blocked**
2. Otherwise, any **unknown**? → **Insufficient Evidence**
3. Otherwise, any **soft fail**? → **Conditional**
4. Otherwise → **Ready**

It also returns which checks triggered the rule (the "driving checks"). No AI involved.

### `pipeline.py`: the conductor

**`assess(...)`** runs the whole show in order: ask Claude to parse → run the checks → apply the rules → ask Claude to explain. It also handles problems:
- **No API key** → status `setup_required`, **no verdict**. The verdict is only about the *data*, never "your setup is broken".
- **API down** → `api_unavailable`, no verdict.
- **Offline demo mode**: if the file's content (checked by a digital fingerprint, SHA-256) **and** the question exactly match a demo case, it uses that case's hand-written checklist, labelled "cached parse (offline)". This never happens for other inputs.

---

## The Claude part: three files

### `prompts.py`: the instructions for Claude

- **`PARSER_SYSTEM`** tells Claude how to turn a question into a checklist. It covers which columns to use, which patients, the outcome, the time horizon and which words don't match any column. It includes one worked example.
- **`EXPLAINER_SYSTEM`** tells Claude to explain the result using **only** the check results, and never to change the verdict.
- Two helper functions build the actual messages sent.

### `parser.py`: question → checklist

- **Loading settings:** `load_env` reads `.env` and never prints the key. `get_model` and `get_client` read the model name and key from environment variables (never written in the code).
- **The checklist's shape:** Pydantic models (`ParsedQuestion`, `Filter`, `Outcome`) define what Claude's answer must look like.
- **What Claude sees:** `profile_dataset` gives Claude a summary of the columns (names, types, possible values), **never the actual patient rows**.
- **Checking Claude's answer:** `validate_parsed` double-checks it. Do the columns really exist? Is the alive/dead column being used as a predictor (not allowed)?
- **Counting predictors:** `count_predictor_parameters` counts in Python, not by Claude. A category with 4 levels counts as 3.
- **Putting it together:** `parse_question` asks Claude. If the answer is invalid, it retries once with the error message, then gives up.

### `explainer.py`: checks → plain English

- **`explain`** asks Claude for a summary, a note for each driving check, a reason for each not-applicable check, and next actions.
- **`validate_explanation`** rejects bad answers. Examples: the summary names a different verdict, it says "all passed" when some weren't applicable, or it skips a driving check.
- **Fallback:** if Claude fails twice, or there's no API, **`fallback_explanation`** builds a plain explanation from the checks' own messages plus fixed advice (`FALLBACK_ACTIONS`).

---

## The audit trail: `evidence.py`

This is the paperwork. It records but never changes anything.
- **`evidence_needed`**: for each failing check, what would change the result. Example: "A cohort of at least 1,205 patients, or a model with fewer than 10 parameters."
- **`fields_used`**: which columns each check looked at.
- **`limitations`**: honest small print: fake data only, the thresholds are conventions, the R²cs guess.
- **Human review:**
  - `validate_review` / `make_review` handle the human sign-off: name, the checkbox, the decision and notes.
  - **`is_override`** spots when a human is *less careful* than the tool, e.g. approving a Blocked dataset. That's allowed, but they must write a justification, and it's flagged.
- **`build_record`, `record_json`, `record_markdown`**: create the downloadable evidence file containing everything above.

---

## The web page: `app.py`

Built with Streamlit, which turns Python into a web page.

- **Sidebar:** pick a demo case or upload a CSV, plus an optional description and advanced settings (R²cs, predictor count).
- **Main area:** the question box and the **Run** button.
- **Results, top to bottom:**
  1. **Badge**: Live (Claude) or Offline.
  2. **Verdict card**: big and colour-coded, with a score line ("9 checks: 7 pass, 2 soft fail...").
  3. **Why box**: only the checks that caused the verdict, in plain English (`headline()`), plus what evidence would change it.
  4. **Riley vs EPV cards**, when sample size is the problem.
  5. **Human review form**: "Human review required before this dataset is used" shows until it's submitted.
  6. **Explanation** and **Next actions**.
  7. **Checks table**, grouped by category. Passing checks are folded away.
  8. **Parsed requirements**: what the question was turned into.
  9. **Evidence record downloads** (JSON / Markdown), **Limitations**, and the note "Synthetic data only. Not for clinical decision-making."

The top half of the file holds small display helpers (`status_label`, `headline`, `scorecard`, `why_box_html`). The bottom half draws the page.

---

## The tests: `tests/`

Tests are little robots that check the code still works after every change. Run them with `.venv/bin/pytest`.

| File | What it proves |
|---|---|
| `conftest.py` | Setup so tests can import the project files |
| `fakes.py` | A **fake Claude** that gives pre-set answers, so tests don't need internet or money |
| `test_generate.py` | The fake data has exactly the promised numbers (560 deaths, etc.) |
| `test_checks.py` | Each of the 9 checks catches what it should |
| `test_sample_size.py` | The Riley formula reproduces the published worked example (623 / 662 / 221 → 662) |
| `test_rules.py` | The four verdict rules work in the right order |
| `test_parser.py` | Bad Claude answers are caught; retry works; predictor counting is right |
| `test_explainer.py` | Bad explanations are rejected; the fallback works |
| `test_pipeline.py` | The whole flow gives the right verdict **and the right driving check** for every demo |
| `test_offline_mode.py` | Offline mode works for the 6 demos and **only** for them |
| `test_app.py` | The web page itself, clicked by a robot: verdict card, Why box, review form, overrides, downloads |
| `test_evidence.py` | Evidence record, "evidence needed" text, overrides, rounding up |
| `test_live.py` | Real Claude calls. Skipped unless you run `.venv/bin/pytest -m live` |

---

## In one sentence

**Cohort Firewall is a safety inspector for medical AI training data. Plain code measures the data, fixed rules give the verdict, Claude only translates and explains, and a human signs off with everything recorded.**
