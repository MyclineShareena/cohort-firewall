"""Demo scenarios: question, hand-written requirements JSON, expected outcome.

The requirements here are the ground truth the parser (Milestone 3) should
produce. `driving` is the exact set of check names that must drive the verdict.
n_predictor_parameters is what Python computes from predictor_columns
(numeric = 1, categorical with k levels = k - 1); tests assert they agree.
"""
from __future__ import annotations

# age 1 + sex 1 + stage 3 + ecog 1 + histology 3 + albumin 1 = 10 parameters
FIELDS_10 = ["age", "sex", "stage", "ecog", "histology", "albumin"]
FIELDS_11 = FIELDS_10 + ["ldh"]

DESCRIPTION = "Synthetic lung cancer cohort"

QUESTION = ("Predict death within 24 months of diagnosis in lung cancer patients "
            "using age, sex, stage, ECOG, histology, albumin and LDH.")


def _req(fields, n_params, unmappable=(), from_description=()):
    return {
        "target_population": [],
        "outcome": {"column": "vital_status", "type": "binary"},
        "required_fields": list(fields),
        "predictor_columns": list(fields),
        "time_horizon_months": 24,
        "task_type": "prediction",
        "n_predictor_parameters": n_params,
        "unmappable_terms": list(unmappable),
        "population_from_description": list(from_description),
    }


LUNG = ["lung cancer patients"]  # satisfied by DESCRIPTION, not by a column


DEMO_CASES = [
    {"name": "clean", "file": "clean.csv", "question": QUESTION, "description": DESCRIPTION,
     "requirements": _req(FIELDS_11, 11, from_description=LUNG),
     "verdict": "Ready", "driving": set()},
    {"name": "messy", "file": "messy.csv", "question": QUESTION, "description": DESCRIPTION,
     "requirements": _req(FIELDS_11, 11, from_description=LUNG),
     "verdict": "Conditional", "driving": {"required_field_missingness", "sample_size"}},
    {"name": "broken", "file": "broken.csv", "question": QUESTION, "description": DESCRIPTION,
     "requirements": _req(FIELDS_11, 11, from_description=LUNG),
     "verdict": "Blocked", "driving": {"required_fields_present", "outcome_derivable"}},
    {"name": "leakage", "file": "leakage.csv", "question": QUESTION, "description": DESCRIPTION,
     "requirements": _req(FIELDS_11, 11, from_description=LUNG),
     "verdict": "Blocked", "driving": {"label_leakage"}},
    {"name": "epv_vs_riley", "file": "epv_vs_riley.csv", "description": DESCRIPTION,
     "question": ("Predict death within 24 months of diagnosis in lung cancer patients "
                  "using age, sex, stage, ECOG, histology and albumin."),
     "requirements": _req(FIELDS_10, 10, from_description=LUNG),
     "verdict": "Conditional", "driving": {"sample_size"}},
    {"name": "pdl1_unmappable", "file": "clean.csv", "description": DESCRIPTION,
     "question": ("Predict death within 24 months using age, sex, stage, ECOG, histology, "
                  "albumin and PD-L1 status."),
     "requirements": _req(FIELDS_10, 10, unmappable=["PD-L1 status"]),
     "verdict": "Insufficient Evidence", "driving": {"question_terms_mappable"}},
]
