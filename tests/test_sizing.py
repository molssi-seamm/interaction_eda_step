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


def test_times_include_the_task_overhead():
    """Every estimate has the task's overhead on top of the calculation."""
    overhead = sizing.TASK_OVERHEAD_SECONDS
    assert overhead >= 41  # the longest launch + collection measured (5681)
    for nbf in (50, 116, 300, 1121):
        assert sizing.psi4_seconds(nbf, 8) > overhead
        assert sizing.gpu_seconds(nbf) > overhead
    # tiny calculations are mostly overhead
    assert sizing.psi4_seconds(20, 8) < overhead + 20


def test_times_cover_measured_runs():
    """Estimates cover runs measured with the threads they asked for."""
    # Li+ PF6- pair, 300 nbf, 8 threads: Psi4 336 s, task 356 s (TC 8586041)
    assert 356 < sizing.psi4_seconds(300, 8) < 2 * 356
    # na6_min, 7 fragments, 383 nbf: 1362 s on 8 threads (ChemAI 5679)
    assert 1362 < sizing.psi4_seconds(383, 8, 7) < 1.5 * 1362
    # shell A, 5 fragments, 1121 nbf: 7712 s on 16 threads (TC 8552392)
    assert 7712 < sizing.psi4_seconds(1121, 16, 5) < 1.1 * 7712
    # small pairs no longer carry the 1-thread floor (was 258 s at 116 nbf)
    assert sizing.psi4_seconds(116, 8) < 180
