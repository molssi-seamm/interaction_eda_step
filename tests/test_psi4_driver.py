# -*- coding: utf-8 -*-

"""The Psi4 ALMO-EDA driver (data/psi4_eda.py): the integral algorithm, and a
density-fitted run of the water dimer, which runs only where Psi4 is installed
(the seamm-psi4 Conda environment)."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import interaction_eda_step

DRIVER = Path(interaction_eda_step.__file__).parent / "data" / "psi4_eda.py"
DATA = Path(__file__).parent / "data"
# The HF water dimer (S22) with exact integrals (GPU4PySCF 1.9.1 and Psi4 agree to
# 1e-4 kJ/mol), kJ/mol
EXACT = {
    "total": -15.396,
    "frozen": -6.092,
    "classical electrostatic": -35.548,
    "pauli": 55.949,
    "polarization": -4.672,
    "charge transfer": -4.632,
}


def _driver():
    spec = importlib.util.spec_from_file_location("_psi4_eda", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jk_algorithm():
    """Integral-direct, which was 4.5x faster than in-core PK; MEM_DF with an
    auxiliary basis."""
    driver = _driver()
    assert driver.jk_algorithm([], {}) == "DIRECT"
    assert driver.jk_algorithm([], {"auxbasis": None}) == "DIRECT"
    assert driver.jk_algorithm([], {"auxbasis": "def2-universal-jkfit"}) == "MEM_DF"


def _psi4_python():
    for path in (
        Path("~/miniconda3/envs/seamm-psi4/bin/python").expanduser(),
        Path("~/miniforge3/envs/seamm-psi4/bin/python").expanduser(),
    ):
        if path.exists():
            return path
    return None


@pytest.mark.skipif(_psi4_python() is None, reason="no seamm-psi4 environment")
def test_density_fitted_water_dimer(tmp_path):
    """With an auxiliary basis every Fock build is density fitted; the fitting
    basis must be built explicitly (Psi4 makes it only for SCF_TYPE DF), or the
    run stops with 'point group has not been set'. The terms stay within
    0.03 kJ/mol of the exact ones."""
    shutil.copy(DATA / "psi4_df_water_dimer.json", tmp_path / "in.json")
    done = subprocess.run(
        [
            str(_psi4_python()),
            "-u",
            str(DRIVER),
            "in.json",
            "-o",
            "out.jsonl",
            "--logdir",
            "logs",
            "--threads",
            "2",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PSI_SCRATCH": str(tmp_path), "OMP_NUM_THREADS": "2", "PATH": "/usr/bin"},
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    (record,) = [json.loads(line) for line in (tmp_path / "out.jsonl").open()]
    assert "error" not in record, record.get("error")
    terms = record["eda_kJ_mol"]
    for term, value in EXACT.items():
        assert terms[term] == pytest.approx(value, abs=0.03), term


def test_backend_memory():
    """A backend that takes memory_gb gets 80 % of the allocation unless set; a
    backend without the key is left alone."""
    from interaction_eda_step.interaction_eda import InteractionEda

    ce = {"MEM_PER_NODE": 64 * 10**9}
    got = InteractionEda.backend_memory({"memory_gb": None, "xc": "hf"}, ce)
    assert got["memory_gb"] == 51.2 and got["xc"] == "hf"
    assert InteractionEda.backend_memory({"memory_gb": 8}, ce)["memory_gb"] == 8
    assert "memory_gb" not in InteractionEda.backend_memory({"xc": "hf"}, ce)
    assert InteractionEda.backend_memory({"memory_gb": None}, {})["memory_gb"] is None
