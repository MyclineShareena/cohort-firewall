"""Milestone 5 scenarios: question + dataset -> known rules-engine verdict.

Extends the six DEMO_CASES with scenarios that drive checks and task types the
demo set never exercises (early_censoring, subgroup_counts, identifier_columns,
a "description" task with no outcome, a "comparison" task). "requirements" is
fed straight to checks.run_checks (no Claude parsing involved): it is the
ground truth the rules engine evaluates. tests/test_scenarios.py asserts each
scenario's declared "verdict"/"driving" actually comes out of
checks.run_checks + rules.assign_verdict, so these stay true if checks.py
changes.
"""
from __future__ import annotations

from data.demo_cases import DEMO_CASES, DESCRIPTION, FIELDS_10, FIELDS_11, LUNG, QUESTION

DEATH_24M = {"column": "vital_status", "type": "binary"}


def _req(required_fields=(), predictor_columns=(), outcome=None, task_type="prediction",
        time_horizon_months=None, n_predictor_parameters=None, target_population=(),
        unmappable_terms=(), population_from_description=()):
    return {
        "target_population": list(target_population),
        "outcome": outcome,
        "required_fields": list(required_fields),
        "predictor_columns": list(predictor_columns),
        "time_horizon_months": time_horizon_months,
        "task_type": task_type,
        "n_predictor_parameters": n_predictor_parameters,
        "unmappable_terms": list(unmappable_terms),
        "population_from_description": list(population_from_description),
    }


NEW_SCENARIOS = [
    {
        "name": "clean_long_horizon_censoring",
        "file": "clean.csv",
        "description": DESCRIPTION,
        "question": ("Predict death within 36 months of diagnosis in lung cancer patients "
                     "using age, sex, stage, ECOG, histology, albumin and LDH."),
        "requirements": _req(FIELDS_11, FIELDS_11, DEATH_24M, "prediction", 36, 11,
                             population_from_description=LUNG),
        "verdict": "Conditional",
        "driving": {"early_censoring"},
    },
    {
        "name": "clean_stage_ii_small_subgroup",
        "file": "clean.csv",
        "description": DESCRIPTION,
        "question": ("Among stage II patients, predict death within 24 months of diagnosis "
                     "using age, sex and ECOG."),
        "requirements": _req(["age", "sex", "ecog"], ["age", "sex", "ecog"], DEATH_24M,
                             "prediction", 24, 3,
                             target_population=[{"column": "stage", "operator": "==", "value": "II"}]),
        "verdict": "Conditional",
        "driving": {"subgroup_counts", "sample_size"},
    },
    {
        "name": "identifiers_blocked",
        "file": "identifiers.csv",
        "description": DESCRIPTION,
        "question": QUESTION,
        "requirements": _req(FIELDS_11, FIELDS_11, DEATH_24M, "prediction", 24, 11,
                             population_from_description=LUNG),
        "verdict": "Blocked",
        "driving": {"identifier_columns"},
    },
    {
        "name": "description_no_outcome",
        "file": "clean.csv",
        "description": DESCRIPTION,
        "question": "Describe the baseline age, sex, stage and ECOG distribution of the cohort.",
        "requirements": _req(["age", "sex", "stage", "ecog"], [], None, "description", None, None,
                             population_from_description=LUNG),
        "verdict": "Insufficient Evidence",
        "driving": {"outcome_derivable"},
    },
    {
        "name": "comparison_stage_mortality",
        "file": "clean.csv",
        "description": DESCRIPTION,
        "question": "Compare 24-month mortality between stage III and stage IV patients.",
        "requirements": _req(["stage"], [], DEATH_24M, "comparison", 24, None,
                             population_from_description=LUNG),
        "verdict": "Ready",
        "driving": set(),
    },
    {
        "name": "messy_unmappable_biomarker",
        "file": "messy.csv",
        "description": DESCRIPTION,
        "question": ("Predict death within 24 months of diagnosis in lung cancer patients using "
                     "age, sex, stage, ECOG, histology, albumin, LDH and tumor mutational burden."),
        "requirements": _req(FIELDS_10, FIELDS_10, DEATH_24M, "prediction", 24, 10,
                             unmappable_terms=["tumor mutational burden"],
                             population_from_description=LUNG),
        "verdict": "Insufficient Evidence",
        "driving": {"question_terms_mappable"},
    },
]

SCENARIOS = DEMO_CASES + NEW_SCENARIOS
