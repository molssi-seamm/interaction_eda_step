#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `interaction_eda_step` package."""

import pytest  # noqa: F401
import interaction_eda_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = interaction_eda_step.InteractionEda()
    assert (
        str(type(result))
        == "<class 'interaction_eda_step.interaction_eda.InteractionEda'>"
    )


def _values(**changes):
    P = interaction_eda_step.InteractionEdaParameters()
    values = P.current_values()
    values.update({k.replace("_", " "): v for k, v in changes.items()})
    return P, values


def test_rules_gpu4pyscf_defaults():
    P, values = _values()
    applies = P.applicable(values)
    for key in ("vv10", "vv10 in response", "dispersion", "low fp64 gpus"):
        assert applies[key], key
    assert not applies["fragment column"]
    assert not applies["triples cutoff"]
    assert P.choices("ferf response", values) is None
    assert P.implied(values) == {}
    assert P.problems(values) == []


def test_rules_psi4():
    P, values = _values(backend="Psi4", functional="wB97X-V")
    applies = P.applicable(values)
    for key in ("vv10", "vv10 in response", "dispersion", "low fp64 gpus"):
        assert not applies[key], key
        assert P.not_applicable_reason(key, values)
    assert P.problems(values) == []
    suggestions = P.functional_suggestions(values)
    assert "wB97X-V" in suggestions and "B3LYP" in suggestions
    assert "wB97M-V" not in suggestions and "B97M-V" not in suggestions

    # Meta-GGAs are refused with Psi4, whatever the FERF response
    for ferf in ("coupled", "uncoupled"):
        P, values = _values(backend="Psi4", functional="wB97M-V", ferf_response=ferf)
        assert any("meta-GGA" in p for p in P.problems(values))

    # ... but fine with GPU4PySCF, and a variable is only known at run time
    P, values = _values(functional="wB97M-V")
    assert P.problems(values) == []
    assert "wB97M-V" in P.functional_suggestions(values)
    P, values = _values(backend="Psi4", functional="$xc")
    assert P.problems(values) == []


def test_backend_settings():
    node = interaction_eda_step.InteractionEda()
    node._id = (1,)
    P = node.parameters.current_values()
    s = node.backend_settings(P)
    assert s["nlc"] is None and s["response_nlc"] is True
    assert s["disp"] is None and s["allow_low_fp64"] is False

    P["vv10"] = "off (whole functional)"
    s = node.backend_settings(P)
    assert s["nlc"] is False and s["response_nlc"] is True

    # Psi4: the GPU, VV10 and dispersion settings are left to the backend
    P = node.parameters.current_values()
    P.update(
        {
            "backend": "Psi4",
            "functional": "wB97X-V",
            "dispersion": "d4",
            "vv10": "off (whole functional)",
        }
    )
    s = node.backend_settings(P)
    for key in ("nlc", "response_nlc", "disp", "allow_low_fp64"):
        assert key not in s, key
    from interaction_eda_step.backends.psi4 import Psi4

    backend = Psi4()
    full = backend.settings(s)
    assert full["response_nlc"] is False and full["nlc"] is None
    assert backend.problems(full) == []
    text = " ".join(node.description_text(P).split())
    assert "no VV10 in the FERF response" in text


def test_n_cores():
    """SEAMM's ncores option caps the cores the backend uses."""
    node = interaction_eda_step.InteractionEda()
    node.global_options = {}
    assert node.n_cores(11) == 11
    node.global_options = {"ncores": "available"}
    assert node.n_cores(11) == 11
    node.global_options = {"ncores": "5"}
    assert node.n_cores(11) == 5
    assert node.n_cores(4) == 4
    node.global_options = {"ncores": 0}
    assert node.n_cores(4) == 1
