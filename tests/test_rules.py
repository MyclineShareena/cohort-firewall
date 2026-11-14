from rules import assign_verdict


def c(name, status, severity="soft"):
    return {"name": name, "metric_cluster": "x", "severity": severity, "status": status,
            "value": None, "threshold": None, "message": ""}


def test_ready():
    r = assign_verdict([c("a", "pass"), c("b", "pass", "hard")])
    assert r["verdict"] == "Ready" and r["driving_checks"] == []


def test_precedence_hard_over_unknown_over_soft():
    checks = [c("soft", "fail"), c("unk", "unknown"), c("hard", "fail", "hard")]
    r = assign_verdict(checks)
    assert r["verdict"] == "Blocked"
    assert [x["name"] for x in r["driving_checks"]] == ["hard"]

    r = assign_verdict(checks[:2])
    assert r["verdict"] == "Insufficient Evidence"
    assert [x["name"] for x in r["driving_checks"]] == ["unk"]

    r = assign_verdict(checks[:1])
    assert r["verdict"] == "Conditional"
    assert [x["name"] for x in r["driving_checks"]] == ["soft"]


def test_empty_is_insufficient():
    assert assign_verdict([])["verdict"] == "Insufficient Evidence"


def test_not_applicable_is_neutral():
    r = assign_verdict([c("a", "pass"), c("na", "not_applicable", "hard")])
    assert r["verdict"] == "Ready" and r["driving_checks"] == []
    r = assign_verdict([c("na", "not_applicable"), c("s", "fail")])
    assert [x["name"] for x in r["driving_checks"]] == ["s"]


def test_rule_text_returned_and_used_by_evidence_record():
    import evidence as E
    from rules import RULE_TEXT
    cases = {"Blocked": [c("h", "fail", "hard")], "Insufficient Evidence": [c("u", "unknown")],
             "Conditional": [c("s", "fail")], "Ready": [c("p", "pass")]}
    for verdict, checks in cases.items():
        r = assign_verdict(checks)
        assert r["verdict"] == verdict and r["rule"] == RULE_TEXT[verdict]
    assert E.DECISION_RULES is RULE_TEXT  # one source of truth
