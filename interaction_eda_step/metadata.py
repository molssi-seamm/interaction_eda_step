# -*- coding: utf-8 -*-

"""Metadata describing the results of the Interaction EDA step.

The results are the common EDA terms (see :mod:`interaction_eda_step.terms`) of
the whole cluster, their 2-body part (the sum over the pairs), their exact
≥3-body part (cluster - Σ pairs) and the truncated 3-body estimate from the
triples that were computed. All energies are in kJ/mol.
"""

from .terms import COMMON_TERMS

metadata = {}

PARTS = {
    "": ("cluster", "of the whole cluster"),
    " 2-body": ("2-body", "summed over all pairs of fragments"),
    " ≥3-body": ("≥3-body", "exact many-body part: cluster minus the sum of pairs"),
    " 3-body (triples)": (
        "3-body (triples)",
        "3-body increments summed over the triples computed; truncated, not the"
        " ≥3-body total",
    ),
}

metadata["results"] = {}
for _term in COMMON_TERMS:
    for _suffix, (_part, _text) in PARTS.items():
        _key = f"EDA {_term}{_suffix}"
        metadata["results"][_key] = {
            "description": f"The EDA {_term} interaction energy, {_text}",
            "dimensionality": "scalar",
            "type": "float",
            "units": "kJ/mol",
            "format": ".3f",
        }
        if _suffix in ("", " ≥3-body"):
            metadata["results"][_key][
                "property"
            ] = f"EDA {_term}{_suffix}#Interaction EDA#{{model}}"

metadata["results"].update(
    {
        "EDA method": {
            "description": "The EDA scheme, code and settings",
            "dimensionality": "scalar",
            "type": "string",
        },
        "EDA pairs": {
            "description": "The EDA terms of each pair: {'i+j': {term: kJ/mol}}",
            "dimensionality": "{n_pairs}",
            "type": "json",
            "units": "kJ/mol",
        },
        "EDA triples": {
            "description": "The 3-body increments of each triple computed",
            "dimensionality": "{n_triples}",
            "type": "json",
            "units": "kJ/mol",
        },
        "number of fragments": {
            "description": "The number of fragments",
            "dimensionality": "scalar",
            "type": "integer",
        },
    }
)
