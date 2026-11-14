# Milestone 5: Claude-alone vs rules engine

Measures how often Claude, left to judge dataset readiness on its own (no
deterministic checks, no rules engine), agrees with the actual rules-engine
verdict. This is the baseline the project's "rules decide, Claude explains"
design argues against; it never runs inside the app.

- `scenarios.py`: 12 question/dataset pairs with a known rules-engine verdict
  (the six `DEMO_CASES` from `data/demo_cases.py` plus six more covering
  checks and task types the demo set doesn't exercise: early censoring, a
  small subgroup, an identifier-like column, a description task with no
  outcome, and a comparison task).
- `claude_alone.py`: gives Claude the same inputs the real parser sees
  (question, description, column profile; never patient rows) and asks it to
  return a verdict directly.
- `run_experiment.py`: for each scenario, runs `checks.run_checks` +
  `rules.assign_verdict` for the ground truth and `claude_alone.assess_alone`
  for Claude's own verdict, then writes a confusion matrix and per-scenario
  detail to `results.md`.

Run (needs `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`, same as the app):

```bash
.venv/bin/python -m experiments.run_experiment
```

This always calls the live API once per scenario; there is no offline mode
here. `tests/test_scenarios.py` checks the scenarios' ground truth offline;
`tests/test_claude_alone.py` checks the prompt/response handling offline with
a fake client. Neither test calls the real API.
