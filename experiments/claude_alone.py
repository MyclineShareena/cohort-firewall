"""Milestone 5: let Claude assign the readiness verdict itself, with none of the
deterministic checks or the rules engine run first. This is the baseline the
project's "rules decide, Claude explains" design is measured against -- it never
runs in the app itself.

Claude sees exactly what the real parser sees (question, optional description,
column profile; never patient rows) but here must also do the judgment that
checks.py and rules.py normally do deterministically.
"""
from __future__ import annotations

import json
from typing import Literal

from parser import ConfigError, _Strict, call_structured, get_client, get_model

VERDICTS = ("Ready", "Conditional", "Blocked", "Insufficient Evidence")

CLAUDE_ALONE_SYSTEM = """\
You are assessing whether an oncology dataset is ready for a proposed AI question. This \
is an experiment: unlike the real tool, no deterministic checks or rules engine run \
first. You must judge readiness yourself, from only the question, an optional one-line \
dataset description, and a column profile (name, kind, levels or range). You never see \
patient rows, and the dataset is synthetic.

Return one verdict:
- "Ready": no problems found.
- "Conditional": usable, but with caveats worth flagging.
- "Blocked": a serious problem makes the data unsuitable for this question.
- "Insufficient Evidence": you cannot tell from what you were given.

Also return concerns: a short list of the specific problems or unknowns driving your \
verdict (empty if none), and reasoning: 2-4 sentences explaining the verdict.

Consider things like: do the columns the question needs exist; how much of what's \
needed looks missing; can the outcome be derived from what's described; does any \
column look like it would leak the answer (e.g. a death date column when predicting \
death); would any subgroup the question implies be too small; do there look to be \
enough patients for the number of predictors the question implies; does anything in \
the column profile look like a real patient identifier. You do not have row-level \
data, so some of this must be inferred or flagged as unknown. This assesses data \
readiness only: give no clinical advice, diagnosis or treatment guidance.
"""


class ClaudeAloneVerdict(_Strict):
    verdict: Literal["Ready", "Conditional", "Blocked", "Insufficient Evidence"]
    concerns: list[str]
    reasoning: str


def claude_alone_user_message(question: str, profile: dict, description: str | None) -> str:
    return "\n\n".join([
        f"Dataset description: {description or 'none given'}",
        f"Column profile:\n{json.dumps(profile, indent=1, default=str)}",
        f"Question:\n{question}",
    ])


def assess_alone(question: str, profile: dict, description: str | None = None,
                 client=None, model: str | None = None) -> dict:
    """Returns {status, verdict, concerns, reasoning, error}.

    status: "ok" | "invalid_output" (Claude answered, but not a valid verdict)
            | "setup_required" | "api_unavailable"
    One attempt only: this is an analysis script, not a user-facing path, so it
    does not need the parser's retry-with-feedback behavior.
    """
    def fail(status, error):
        return {"status": status, "verdict": None, "concerns": None, "reasoning": None, "error": error}

    try:
        model = model or get_model()
        client = client or get_client()
    except ConfigError as e:
        return fail("setup_required", str(e))

    user = claude_alone_user_message(question, profile, description)
    parsed, err, api_failed = call_structured(client, model, CLAUDE_ALONE_SYSTEM, user,
                                              ClaudeAloneVerdict)
    if parsed is None:
        return fail("api_unavailable" if api_failed else "invalid_output", err)
    return {"status": "ok", "verdict": parsed["verdict"], "concerns": parsed["concerns"],
            "reasoning": parsed["reasoning"], "error": None}
