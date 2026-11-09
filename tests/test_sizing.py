"""Sizing a Psi4 EDA task: the memory asked for holds the peak, and the EDA
engine's share of Psi4's memory holds its in-core integrals."""

import importlib.resources
import re

import pytest

from interaction_eda_step import sizing

DRIVER = importlib.resources.files("interaction_eda_step") / "data" / "psi4_eda.py"


def engine_fraction():
    text = DRIVER.read_text()
    return float(re.search(r"^ENGINE_JK_FRACTION = ([0-9.]+)", text, re.M).group(1))


# (nbf, naux): Li+-EC, EC-EC, the stage-2 Li+ shells (2026-11-09) and shellC
SIZES = [(284, 579), (552, 1092), (1085, 2163), (1121, 2235), (1388, 2763)]


@pytest.mark.parametrize("nbf, naux", SIZES)
def test_memory_holds_the_engine_integrals_and_the_peak(nbf, naux):
    cores = sizing.psi4_cores(nbf)
    job = sizing.psi4_memory(nbf, naux, True, cores=cores)
    integrals = sizing.SCREENED * sizing.df_bytes(nbf, naux)
    psi4 = 0.8 * job
    # the engine's share keeps its integrals in core, with room to spare
    assert engine_fraction() * psi4 > 1.2 * integrals
    # the peak (about 1.9 times the integrals) fits in what is asked for
    assert sizing.PEAK_OVER_JK * integrals < job <= sizing.PSI4_MAX_GB * 1e9


def test_shell_a_as_measured():
    """Shell A (1121 nbf) peaked at 98 GB; Psi4 needed 47.6 GiB in core."""
    job = sizing.psi4_memory(1121, 2235, True, cores=32)
    assert job >= 98e9 * 1.2
    assert engine_fraction() * 0.8 * job > 47.6 * 2**30


def test_times_include_the_fixed_part():
    # small pairs measured 240-270 s on 8 cores (ChemAI 5680)
    assert 220 < sizing.psi4_seconds(116, 8) < 300
    # na6_min, 7 fragments: 1362 s on 8 cores (ChemAI 5679)
    assert 1200 < sizing.psi4_seconds(383, 8, 7) < 1600
