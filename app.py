"""Cohort Firewall: Streamlit UI.  Run:  .venv/bin/streamlit run app.py

Rules decide the verdict; Claude only parses the question and explains. Without an
API key the demo cases still run in offline demo mode (cached parse + rule-based
explanation). Adding ANTHROPIC_API_KEY / ANTHROPIC_MODEL to .env switches to live on
the next Run: the environment is re-read on every call.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import evidence as E
from data.demo_cases import DEMO_CASES
from pipeline import CACHED_LABEL, DATA_DIR, assess

SAFETY_NOTE = "Synthetic data only. Not for clinical decision-making."
OFFLINE_NOTE = "Running offline: explanation built from the checks' own messages."

CASE_LABELS = {
    "clean": "clean.csv",
    "messy": "messy.csv",
    "broken": "broken.csv",
    "leakage": "leakage.csv",
    "epv_vs_riley": "epv_vs_riley.csv",
    "pdl1_unmappable": "clean.csv + PD-L1 question",
}
CASES = {c["name"]: c for c in DEMO_CASES}

VERDICT_STYLE = {  # (accent colour, one-line meaning)
    "Ready": ("#1a7f37", "No failing checks for this question."),
    "Conditional": ("#b7791f", "Usable with documented caveats: soft checks failed."),
    "Blocked": ("#cf222e", "A hard check failed; do not proceed with this question."),
    "Insufficient Evidence": ("#57606a", "Readiness could not be determined."),
}
CLUSTER_ORDER = ["consistency", "informativeness", "timeliness", "representativeness",
                 "measurement process", "safety"]
STATUS_LABEL = {"pass": "✅ pass", "unknown": "❔ unknown", "not_applicable": "— not applicable"}

BADGE_STYLE = {
    "live": ("#0969da", "Live: Claude"),
    "offline": ("#8250df", "Offline: cached parse + rule-based explanation"),
    "mixed": ("#b7791f", "Live: Claude parse + rule-based explanation"),
}


# ------------------------------------------------------------------ pure helpers (tested)

def mode_badge(result: dict) -> tuple[str, str] | None:
    """(kind, label) describing where the parse and explanation came from."""
    if result.get("status") != "ok":
        return None
    if result["parse_source"] == CACHED_LABEL:
        return "offline", BADGE_STYLE["offline"][1]
    if result["explanation"]["source"] == "claude":
        return "live", BADGE_STYLE["live"][1]
    return "mixed", BADGE_STYLE["mixed"][1]


def status_label(check: dict) -> str:
    if check["status"] == "fail":
        return f"❌ fail ({check['severity']})" if check["severity"] == "hard" else "⚠️ fail (soft)"
    return STATUS_LABEL[check["status"]]


def _list(items, limit=4) -> str:
    items = list(items)
    return ", ".join(map(str, items[:limit])) + (", ..." if len(items) > limit else "")


def headline(check: dict) -> str:
    """Short plain-English summary for the Why box. Display only: the verdict logic
    and the full technical message (shown in the checks table) are unchanged."""
    name, status, v = check["name"], check["status"], check.get("value")
    if status in {"pass", "not_applicable"}:
        return check["message"]
    if name == "sample_size" and status == "fail":
        return (f"Too few patients for {v['n_params']} predictors: Riley needs "
                f"{v['n_required']:,}, this dataset has {v['n_available']:,}.")
    if name == "required_field_missingness" and status == "fail":
        over = {c: f for c, f in v["by_column"].items() if f > 0.20}
        cols = _list(f"{c} {f:.0%}" for c, f in sorted(over.items(), key=lambda x: -x[1]))
        limit = "over 40%" if check["severity"] == "hard" else "limit 20%"
        return f"Too much missing data in required fields: {cols} ({limit})."
    if name == "required_fields_present" and status == "fail":
        if v and v["missing"]:
            return f"Required column(s) missing from the file: {_list(v['missing'])}."
        return "Some required columns contain values of the wrong type."
    if name == "outcome_derivable":
        if status == "unknown":
            return "The outcome to assess could not be determined from the question."
        if v is None:
            return "The outcome can't be worked out: the columns needed to derive it are missing."
        return "The outcome has no variation: every patient has the same result."
    if name == "question_terms_mappable":
        return f"The question asks about {_list(v)}, which no column in the dataset records."
    if name == "question_parsed":
        return "The question could not be turned into checkable requirements."
    if name == "early_censoring" and status == "fail":
        return (f"{v:.0%} of patients were followed too briefly to know their outcome "
                "(limit 15%).")
    if name == "subgroup_counts" and status == "fail":
        small = [f"{k} ({n})" for k, n in v.items() if n < check["threshold"]]
        return f"Some subgroups have fewer than {check['threshold']} patients: {_list(small)}."
    if name == "label_leakage":
        return f"{_list(v)} reveals the outcome and would leak it into the model."
    if name == "identifier_columns":
        return f"Columns that look like real patient identifiers: {_list(v, 3)}."
    if status == "unknown":
        return f"Could not be assessed: {check['message']}"
    return check["message"]


def scorecard(checks: list[dict]) -> str:
    """'N checks: X pass, Y soft fail, Z hard fail, W unknown, V not applicable'."""
    def n(pred):
        return sum(1 for c in checks if pred(c))
    return (f"{len(checks)} checks: {n(lambda c: c['status'] == 'pass')} pass, "
            f"{n(lambda c: c['status'] == 'fail' and c['severity'] == 'soft')} soft fail, "
            f"{n(lambda c: c['status'] == 'fail' and c['severity'] == 'hard')} hard fail, "
            f"{n(lambda c: c['status'] == 'unknown')} unknown, "
            f"{n(lambda c: c['status'] == 'not_applicable')} not applicable")


def riley_epv_table(check: dict) -> pd.DataFrame | None:
    v = check.get("value")
    if check["status"] == "not_applicable" or not v or "n_required" not in v:
        return None
    rows = [
        ("Riley (i) shrinkage S ≥ 0.9", f"n ≥ {v['criterion_i']:,}"),
        ("Riley (ii) Nagelkerke optimism ≤ 0.05", f"n ≥ {v['criterion_ii']:,}"),
        ("Riley (iii) overall risk ± 0.05", f"n ≥ {v['criterion_iii']:,}"),
        ("Riley required (max)", f"n ≥ {v['n_required']:,} ({v['events_required']:,} events), "
                                 f"criterion {v['binding_criterion']}"),
        ("Available", f"n = {v['n_available']:,} ({v['events']:,} events)"),
        ("Riley result", "✅ meets" if v["n_available"] >= v["n_required"] else "⚠️ below"),
        ("EPV (rule of thumb, comparison only)",
         f"{v['epv']:.1f} events per parameter — {'✅ ≥' if v['epv_pass'] else '⚠️ <'} "
         f"{v['epv_threshold']}"),
    ]
    return pd.DataFrame(rows, columns=["Criterion", "Value"])


# ------------------------------------------------------------------ rendering

def _pill(color: str, text: str) -> str:
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
            f'border:1px solid {color};color:{color};font-size:0.85rem;font-weight:600">'
            f'{html.escape(text)}</span>')


def render_badge(result: dict) -> None:
    badge = mode_badge(result)
    if badge:
        color = BADGE_STYLE[badge[0]][0]
        st.markdown(_pill(color, badge[1]), unsafe_allow_html=True)


def render_verdict_card(result: dict) -> None:
    verdict = result["verdict"]
    color, meaning = VERDICT_STYLE[verdict]
    st.markdown(
        f'<div style="border-left:10px solid {color};border-radius:10px;padding:16px 22px;'
        f'margin:8px 0 6px 0;background:{color}1a">'
        f'<div style="font-size:0.8rem;opacity:0.8">VERDICT (assigned by the rules engine)</div>'
        f'<div class="cf-verdict" style="font-size:3.4rem;line-height:1.1;font-weight:800;'
        f'color:{color}">{html.escape(verdict)}</div>'
        f'<div>{html.escape(meaning)}</div>'
        f'<div class="cf-scorecard" style="font-size:0.9rem;margin-top:8px;opacity:0.85">'
        f'{html.escape(scorecard(result["checks"]))}</div></div>',
        unsafe_allow_html=True)


def _needed_html(check: dict, req: dict | None) -> str:
    need = E.evidence_needed(check, req)
    if not need:
        return ""
    return (f'<div class="cf-needed" style="font-size:0.88rem;opacity:0.85;margin-top:2px">'
            f'<i>Evidence needed to reconsider:</i> {html.escape(need)}</div>')


def why_box_html(result: dict) -> str:
    """Only the checks that drove the verdict."""
    color = VERDICT_STYLE[result["verdict"]][0]
    driving = result["driving_checks"]
    if driving:
        items = "".join(
            f'<li><b>{html.escape(status_label(c))}</b> {html.escape(headline(c))} '
            f'<code style="opacity:0.7">{html.escape(c["name"])}</code>'
            f'{_needed_html(c, result["requirements"])}</li>' for c in driving)
        body = f'<ul style="margin:4px 0 0 0;padding-left:20px">{items}</ul>'
    else:
        body = "<div>No check failed or was inconclusive.</div>"
    return (f'<div class="cf-why" style="border:1px solid {color};border-radius:8px;'
            f'padding:10px 16px;margin:6px 0 10px 0">'
            f'<div style="font-weight:700">Why</div>{body}</div>')


def render_explanation(result: dict) -> None:
    ex = result["explanation"]
    st.subheader("Explanation")
    if ex["source"] == "fallback":
        # Offline: the Why box and checks table already show the findings.
        st.caption(OFFLINE_NOTE)
        st.write(ex["summary"])
    else:
        _render_claude_explanation(ex)
    _render_next_actions(ex)


def _render_claude_explanation(ex: dict) -> None:
    st.write(ex["summary"])
    for d in ex["driving_checks"]:
        st.markdown(f"- **{d['name']}**: {d['explanation']}")
    if ex["not_applicable"]:
        st.markdown("**Not applicable** (not evaluated, not passed):")
        for n in ex["not_applicable"]:
            st.markdown(f'<div style="color:#6e7781">— {html.escape(n["name"])}: '
                        f'{html.escape(n["reason"])}</div>', unsafe_allow_html=True)


def _render_next_actions(ex: dict) -> None:
    st.subheader("Next actions")
    if ex["next_actions"]:
        for a in ex["next_actions"]:
            tag = f"**{a['check_name']}**: " if a["check_name"] else ""
            st.markdown(f"- {tag}{a['action']}")
    else:
        st.write("No failing checks to address.")


_CELL = "padding:6px 8px;border-bottom:1px solid rgba(128,128,128,0.25);vertical-align:top"


def checks_table_html(checks: list[dict], req: dict | None = None) -> str:
    """Full-width table, message column wraps; not-applicable rows gray. Failing and
    unknown checks also say what evidence would let them be reconsidered."""
    rows = []
    for c in checks:
        gray = "color:#8c959f;" if c["status"] == "not_applicable" else ""
        rows.append(
            f'<tr style="{gray}"><td style="{_CELL};white-space:nowrap">{html.escape(status_label(c))}</td>'
            f'<td style="{_CELL}"><code>{html.escape(c["name"])}</code></td>'
            f'<td style="{_CELL};overflow-wrap:anywhere">{html.escape(c["message"])}'
            f'{_needed_html(c, req)}</td></tr>')
    head = "".join(f'<th style="{_CELL};text-align:left">{h}</th>'
                   for h in ("Status", "Check", "Message"))
    return ('<table style="width:100%;border-collapse:collapse;table-layout:fixed;'
            'font-size:0.92rem;margin-bottom:12px">'
            '<colgroup><col style="width:9.5rem"><col style="width:14rem"><col></colgroup>'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>")


def render_checks(result: dict) -> None:
    st.subheader("Checks by METRIC cluster")
    open_checks = [c for c in result["checks"] if c["status"] != "pass"]
    passing = [c for c in result["checks"] if c["status"] == "pass"]
    if open_checks:
        _render_cluster_tables(open_checks, result["requirements"])
    else:
        st.write("Every check passed.")
    if passing:
        with st.expander(f"{len(passing)} passing checks"):
            _render_cluster_tables(passing)


def _render_cluster_tables(checks: list[dict], req: dict | None = None) -> None:
    for cluster in CLUSTER_ORDER:
        group = [c for c in checks if c["metric_cluster"] == cluster]
        if group:
            st.markdown(f"**{cluster.capitalize()}**")
            st.markdown(checks_table_html(group, req), unsafe_allow_html=True)


def _card(title: str, headline: str, lines: list[str], color: str, footer: str) -> str:
    body = "".join(f"<div>{html.escape(x)}</div>" for x in lines)
    return (f'<div style="border:1px solid {color};border-top:6px solid {color};border-radius:8px;'
            f'padding:12px 16px;height:100%;background:{color}12">'
            f'<div style="font-size:0.8rem;opacity:0.8">{html.escape(title)}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:{color};margin:2px 0 6px">'
            f'{html.escape(headline)}</div>{body}'
            f'<div style="font-size:0.8rem;opacity:0.75;margin-top:8px">{html.escape(footer)}</div>'
            "</div>")


def sample_size_cards_html(check: dict) -> tuple[str, str] | None:
    """(Riley card, EPV card) HTML, or None when sample size was not computed."""
    v = check.get("value")
    if check["status"] == "not_applicable" or not v or "n_required" not in v:
        return None
    ok = v["n_available"] >= v["n_required"]
    riley = _card(
        "RILEY ET AL. (BMJ 2020): decides the sample-size check",
        f"{'✅ Meets' if ok else '⚠️ Below'}: n = {v['n_available']:,} vs ≥ {v['n_required']:,}",
        [f"Events: {v['events']:,} available vs {v['events_required']:,} needed",
         f"(i) shrinkage S ≥ 0.9: n ≥ {v['criterion_i']:,}",
         f"(ii) Nagelkerke optimism ≤ 0.05: n ≥ {v['criterion_ii']:,}",
         f"(iii) overall risk ± 0.05: n ≥ {v['criterion_iii']:,}",
         f"Binding criterion: ({v['binding_criterion']}), {v['n_params']} parameters"],
        VERDICT_STYLE["Ready"][0] if ok else VERDICT_STYLE["Conditional"][0],
        v["r2cs_note"])
    epv = _card(
        "EVENTS PER VARIABLE: rule of thumb, comparison only",
        f"{'✅' if v['epv_pass'] else '⚠️'} EPV = {v['epv']:.1f} "
        f"({'≥' if v['epv_pass'] else '<'} {v['epv_threshold']})",
        [f"{v['events']:,} events / {v['n_params']} parameters",
         "Shown for comparison; it does not affect the verdict."],
        "#57606a", "EPV ignores outcome prevalence and expected model fit; Riley accounts for both.")
    return riley, epv


def render_sample_size_cards(check: dict) -> None:
    cards = sample_size_cards_html(check)
    if cards is None:
        return
    st.markdown("**Sample size: Riley vs EPV**")
    left, right = st.columns(2)
    left.markdown(cards[0], unsafe_allow_html=True)
    right.markdown(cards[1], unsafe_allow_html=True)


def render_sample_size(result: dict) -> None:
    ss = next((c for c in result["checks"] if c["name"] == "sample_size"), None)
    if ss is None:
        return
    st.subheader("Sample size: Riley vs EPV")
    table = riley_epv_table(ss)
    if table is None:
        st.markdown(f'<div style="color:#6e7781">{html.escape(ss["message"])}</div>',
                    unsafe_allow_html=True)
        return
    st.table(table)
    st.caption(ss["value"]["r2cs_note"])


def render_review(result: dict) -> None:
    """Human review, recorded alongside the verdict. It never changes the verdict."""
    st.subheader("Human review")
    review = st.session_state.get("review")
    if review:
        st.success(f"Review recorded: {review['decision']}, by {review['reviewer']} at "
                   f"{review['submitted_at']}. The tool's verdict ({result['verdict']}) is "
                   "unchanged; the review is recorded alongside it.")
        if review["notes"]:
            st.caption(f"Notes: {review['notes']}")
        return
    st.warning(E.REVIEW_PENDING)
    with st.form("review_form"):
        reviewer = st.text_input("Reviewer name", key="review_name")
        reviewed = st.checkbox("I have reviewed the evidence", key="review_checked")
        decision = st.radio("Decision", E.DECISIONS, index=None, key="review_decision")
        notes = st.text_area("Notes", key="review_notes", height=80)
        submitted = st.form_submit_button("Submit review")
    if submitted:
        errors = E.validate_review(reviewer, reviewed, decision)
        if errors:
            for e in errors:
                st.error(e)
            return
        st.session_state.review = E.make_review(reviewer, reviewed, decision, notes,
                                                result["verdict"])
        st.rerun()


def current_record(result: dict) -> dict:
    return E.build_record(result, st.session_state.get("review"),
                          st.session_state.get("dataset_info"))


def render_evidence_record(result: dict) -> None:
    st.subheader("Evidence record")
    record = current_record(result)
    pending = record["human_review"]["status"] == "pending"
    st.caption("Question, parsed requirements, verdict, every check, the decision rule, "
               "limitations and the review decision."
               + (" Human review is still pending." if pending else ""))
    left, right = st.columns(2)
    left.download_button("Download JSON", E.record_json(record),
                         file_name="cohort_firewall_evidence.json", mime="application/json")
    right.download_button("Download Markdown", E.record_markdown(record),
                          file_name="cohort_firewall_evidence.md", mime="text/markdown")


def render_limitations(result: dict) -> None:
    st.subheader("Limitations")
    st.markdown("\n".join(f"- {x}" for x in E.limitations(result)))


def render_result(result: dict) -> None:
    render_badge(result)
    if result["status"] != "ok":
        st.warning(result["message"])  # technical detail stays in result["detail"]
        return
    render_verdict_card(result)
    st.markdown(why_box_html(result), unsafe_allow_html=True)
    ss_driving = next((c for c in result["driving_checks"] if c["name"] == "sample_size"), None)
    if ss_driving:
        render_sample_size_cards(ss_driving)
    render_review(result)
    for note in result["notes"]:
        st.info(note)
    render_explanation(result)
    render_checks(result)
    if not ss_driving:
        render_sample_size(result)
    with st.expander("Parsed requirements"):
        st.caption(f"Parse source: {result['parse_source']}")
        st.json(result["requirements"] or {"error": result["parse"]["error"]})
    render_evidence_record(result)
    render_limitations(result)
    st.caption(SAFETY_NOTE)


# ------------------------------------------------------------------ page

@st.cache_data
def load_demo_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _select_case() -> None:
    case = CASES[st.session_state.case]
    st.session_state.question = case["question"]
    st.session_state.description = case["description"]
    st.session_state.pop("result", None)
    st.session_state.pop("review", None)


def main() -> None:
    st.set_page_config(page_title="Cohort Firewall", page_icon="🛡️", layout="wide")
    st.title("Cohort Firewall")
    st.caption("Oncology dataset readiness checker. Rules decide the verdict; Claude only "
               "parses the question and explains. " + SAFETY_NOTE)

    if "case" not in st.session_state:
        st.session_state.case = DEMO_CASES[0]["name"]
        _select_case()

    with st.sidebar:
        st.header("Input")
        source = st.radio("Dataset", ["Demo case", "Upload CSV"], key="source")
        df = None
        if source == "Demo case":
            st.selectbox("Demo case", list(CASES), key="case", on_change=_select_case,
                         format_func=CASE_LABELS.get)
            df = load_demo_csv(CASES[st.session_state.case]["file"])
        else:
            up = st.file_uploader("Synthetic CSV", type="csv")
            if up is not None:
                df = pd.read_csv(up)
        st.text_input("Dataset description (optional)", key="description")
        with st.expander("Advanced"):
            r2 = st.number_input("Anticipated R²cs (default: 15% of max)", min_value=0.0,
                                 max_value=1.0, value=None, step=0.01, format="%.3f", key="r2")
            p_override = st.number_input("Predictor-parameter override", min_value=1,
                                         value=None, step=1, key="p_override")
        if df is not None:
            st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

    st.text_area("Proposed AI question", key="question", height=90)
    run = st.button("Run", type="primary",
                    disabled=df is None or not st.session_state.question.strip())
    if run:
        st.session_state.pop("review", None)  # a new result needs a new review
        st.session_state.dataset_info = {
            "name": CASES[st.session_state.case]["file"] if source == "Demo case"
            else getattr(up, "name", "uploaded CSV"),
            "rows": len(df), "columns": list(df.columns)}
        with st.spinner("Assessing..."):
            st.session_state.result = assess(
                df, st.session_state.question, st.session_state.description.strip() or None,
                r2=r2, p_override=int(p_override) if p_override else None)
    if "result" in st.session_state:
        render_result(st.session_state.result)


main()
