"""Milestone 5 (bonus): how often does Claude, judging readiness on its own, agree
with the rules-engine verdict?

Run from the project root:  .venv/bin/python -m experiments.run_experiment
Needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL (environment or .env). This always
calls the live API, once per scenario -- there is no offline/cached mode here.
Writes experiments/results.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import checks as C
from experiments.claude_alone import VERDICTS, assess_alone
from experiments.scenarios import SCENARIOS
from parser import ConfigError, get_client, get_model, profile_dataset
from rules import assign_verdict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


def ground_truth(df: pd.DataFrame, req: dict) -> dict:
    checks = C.run_checks(df, req)
    v = assign_verdict(checks)
    return {"verdict": v["verdict"], "driving": {c["name"] for c in v["driving_checks"]}}


def run(client, model: str) -> list[dict]:
    rows = []
    for sc in SCENARIOS:
        df = pd.read_csv(DATA_DIR / sc["file"])
        truth = ground_truth(df, sc["requirements"])
        alone = assess_alone(sc["question"], profile_dataset(df), sc.get("description"),
                             client=client, model=model)
        rows.append({
            "name": sc["name"], "file": sc["file"], "question": sc["question"],
            "truth_verdict": truth["verdict"], "truth_driving": sorted(truth["driving"]),
            "claude_status": alone["status"], "claude_verdict": alone["verdict"],
            "claude_concerns": alone["concerns"], "claude_reasoning": alone["reasoning"],
            "agree": alone["verdict"] == truth["verdict"],
        })
    return rows


def confusion_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts = {t: {p: 0 for p in VERDICTS} for t in VERDICTS}
    for r in rows:
        if r["claude_verdict"] in VERDICTS:
            counts[r["truth_verdict"]][r["claude_verdict"]] += 1
    return counts


def confusion_markdown(counts: dict[str, dict[str, int]]) -> str:
    header = "| rules engine \\ Claude alone | " + " | ".join(VERDICTS) + " |"
    sep = "|" + "---|" * (len(VERDICTS) + 1)
    lines = [header, sep]
    for t in VERDICTS:
        lines.append(f"| {t} | " + " | ".join(str(counts[t][p]) for p in VERDICTS) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict], path: Path) -> None:
    n = len(rows)
    n_ok = sum(1 for r in rows if r["claude_status"] == "ok")
    n_agree = sum(1 for r in rows if r["agree"])
    lines = [
        "# Milestone 5: Claude-alone vs rules engine",
        "",
        f"{n} scenarios. {n_ok} Claude calls returned a verdict; "
        f"{n_agree} of those agreed with the rules engine.",
        "",
        "## Confusion matrix",
        "",
        confusion_markdown(confusion_counts(rows)),
        "",
        "## Per-scenario detail",
        "",
    ]
    for r in rows:
        mark = "agree" if r["agree"] else "DISAGREE"
        lines += [
            f"### {r['name']} ({mark})",
            f"- File: {r['file']}",
            f"- Question: {r['question']}",
            f"- Rules engine: **{r['truth_verdict']}** "
            f"(driving: {', '.join(r['truth_driving']) or 'none'})",
            f"- Claude alone: **{r['claude_verdict']}** (status: {r['claude_status']})",
        ]
        if r["claude_concerns"]:
            lines.append(f"- Claude's concerns: {'; '.join(r['claude_concerns'])}")
        if r["claude_reasoning"]:
            lines.append(f"- Claude's reasoning: {r['claude_reasoning']}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    try:
        model = get_model()
        client = get_client()
    except ConfigError as e:
        print(f"Cannot run the live comparison: {e}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL (environment or .env) and re-run.",
              file=sys.stderr)
        sys.exit(1)

    rows = run(client, model)
    write_report(rows, RESULTS_PATH)
    n_ok = sum(1 for r in rows if r["claude_status"] == "ok")
    n_agree = sum(1 for r in rows if r["agree"])
    print(f"{len(rows)} scenarios, {n_agree}/{n_ok or len(rows)} agreed. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
