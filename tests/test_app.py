"""Headless UI tests (Streamlit AppTest). Forced offline: no .env, no key."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import parser as P
from data.demo_cases import DEMO_CASES

APP = str(Path(__file__).resolve().parents[1] / "app.py")
OFFLINE = "Offline: cached parse + rule-based explanation"


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(P, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def html_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def run_case(name):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="case").select(name).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_each_demo_case_runs_offline(offline, case):
    at = run_case(case["name"])
    text = html_text(at)
    assert OFFLINE in text
    assert f">{case['verdict']}</div>" in text  # verdict card
    for name in case["driving"]:
        assert name in text
    assert "Synthetic data only. Not for clinical decision-making." in "\n".join(
        c.value for c in at.caption)
    assert at.session_state.result["verdict"] == case["verdict"]


def test_selecting_case_fills_question(offline):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="case").select("pdl1_unmappable").run()
    assert "PD-L1" in at.text_area(key="question").value
    assert at.text_input(key="description").value == "Synthetic lung cancer cohort"


def test_edited_question_offline_is_setup_required(offline):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="question").input("Does albumin predict 12-month death?").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Set ANTHROPIC_API_KEY and ANTHROPIC_MODEL" in w.value for w in at.warning)
    assert "VERDICT" not in html_text(at)  # no verdict card


def test_not_applicable_shown_gray_not_passed(offline):
    # epv_vs_riley with a description question has no N/A; check a description task
    # via the pure helpers instead of the API.
    import app
    assert app.status_label({"status": "not_applicable", "severity": "hard"}) == "— not applicable"
    assert "pass" not in app.status_label({"status": "not_applicable", "severity": "soft"})


def test_mode_badge_kinds():
    import app
    base = {"status": "ok", "parse_source": "claude", "explanation": {"source": "claude"}}
    assert app.mode_badge(base)[1] == "Live: Claude"
    assert app.mode_badge({**base, "parse_source": app.CACHED_LABEL,
                           "explanation": {"source": "fallback"}})[1] == OFFLINE
    assert app.mode_badge({**base, "explanation": {"source": "fallback"}})[0] == "mixed"
    assert app.mode_badge({"status": "setup_required"}) is None


def test_sample_size_cards_under_verdict_when_driving(offline):
    at = run_case("epv_vs_riley")
    md = [m.value for m in at.markdown]
    verdict_i = next(i for i, m in enumerate(md) if "VERDICT (assigned" in m)
    riley_i = next(i for i, m in enumerate(md) if "RILEY ET AL." in m)
    epv_i = next(i for i, m in enumerate(md) if "EVENTS PER VARIABLE" in m)
    explain_i = next(i for i, m in enumerate(md) if "**sample_size**" in m)
    assert verdict_i < riley_i < explain_i and verdict_i < epv_i < explain_i
    assert "n = 1,000 vs ≥ 1,205" in md[riley_i] and "⚠️ Below" in md[riley_i]
    assert "EPV = 10.0 (≥ 10)" in md[epv_i]
    assert len(at.table) == 0  # no duplicate table at the bottom


def test_sample_size_table_at_bottom_when_not_driving(offline):
    at = run_case("clean")
    assert not any("RILEY ET AL." in m.value for m in at.markdown)
    values = " ".join(map(str, at.table[0].value["Value"].tolist()))
    assert "✅ meets" in values


def test_checks_tables_have_no_severity_column(offline):
    at = run_case("messy")
    tables = [m.value for m in at.markdown if "<table" in m.value]
    assert tables and all(">Message<" in t and "Severity" not in t for t in tables)
    assert any("⚠️ fail (soft)" in t and "Missing above 20%: stage 30%." in t for t in tables)
    assert not at.dataframe  # no truncating dataframe widgets


def _all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [x.value for x in at.info] + [x.value for x in at.warning]
    return "\n".join(parts)


def test_verdict_is_largest_element(offline):
    import re
    at = run_case("clean")
    sizes = [float(x) for x in re.findall(r"font-size:([\d.]+)rem", html_text(at))]
    card = next(m.value for m in at.markdown if "cf-verdict" in m.value)
    verdict_size = float(re.search(r'cf-verdict" style="font-size:([\d.]+)rem', card).group(1))
    assert verdict_size == max(sizes) and verdict_size > 2.75  # st.title is ~2.75rem
    assert "color:#1a7f37" in card  # Ready = green


@pytest.mark.parametrize("name,line", [
    ("messy", "9 checks: 7 pass, 2 soft fail, 0 hard fail, 0 unknown, 0 not applicable"),
    ("broken", "9 checks: 5 pass, 0 soft fail, 2 hard fail, 2 unknown, 0 not applicable"),
    ("clean", "9 checks: 9 pass, 0 soft fail, 0 hard fail, 0 unknown, 0 not applicable"),
])
def test_scorecard_line(offline, name, line):
    assert line in html_text(run_case(name))


def test_scorecard_counts_not_applicable():
    import app
    checks = [{"status": s, "severity": v} for s, v in
              [("pass", "hard"), ("fail", "soft"), ("fail", "hard"), ("unknown", "soft"),
               ("not_applicable", "hard"), ("not_applicable", "soft")]]
    assert app.scorecard(checks) == ("6 checks: 1 pass, 1 soft fail, 1 hard fail, 1 unknown, "
                                     "2 not applicable")


def test_why_box_lists_only_driving_checks(offline):
    at = run_case("broken")
    why = next(m.value for m in at.markdown if "cf-why" in m.value)
    assert "required_fields_present" in why and "outcome_derivable" in why
    assert "early_censoring" not in why and "sample_size" not in why  # unknown, not driving
    why_clean = next(m.value for m in run_case("clean").markdown if "cf-why" in m.value)
    assert "No check failed" in why_clean


def test_passing_checks_collapsed(offline):
    at = run_case("messy")
    exp = [e for e in at.expander if "passing checks" in e.label]
    assert len(exp) == 1 and exp[0].label == "7 passing checks"
    inside = "\n".join(m.value for m in exp[0].markdown)
    assert "required_fields_present" in inside and "✅ pass" in inside
    outside = [m.value for m in at.markdown if "<table" in m.value and m.value not in inside]
    assert outside and not any("✅ pass" in t for t in outside)


@pytest.mark.parametrize("case", DEMO_CASES, ids=lambda c: c["name"])
def test_offline_note_and_no_env_vars_in_main_view(offline, case):
    at = run_case(case["name"])
    text = _all_text(at)
    assert "Running offline: explanation built from the checks' own messages." in text
    assert "ANTHROPIC" not in text and ".env" not in text
