#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The GPU4PySCF backend without a GPU: its input, its driver's helpers, and
reading results written in the driver's format."""

import importlib.resources
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from interaction_eda_step.backends import BACKENDS, Calculation, GPU4PySCF
from interaction_eda_step.fragments import Fragment
from interaction_eda_step import terms

DATA = Path(__file__).parent / "data"


def li_ec_fragments():
    li = Fragment(0, "Li+", [0], ["Li"], np.zeros((1, 3)), charge=1)
    water = Fragment(
        1,
        "H2O",
        [1, 2, 3],
        ["O", "H", "H"],
        np.array([[2.0, 0, 0], [2.6, 0.76, 0], [2.6, -0.76, 0]]),
    )
    return [li, water]


def load_driver():
    path = importlib.resources.files("interaction_eda_step") / "data"
    path = path / "gpu4pyscf_eda.py"
    spec = importlib.util.spec_from_file_location("gpu4pyscf_eda", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports nothing from GPU4PySCF at load
    return module


def test_registry():
    assert BACKENDS["GPU4PySCF"] is GPU4PySCF
    backend = GPU4PySCF()
    assert backend.scheme in terms.SCHEMES and backend.device == "gpu"


def test_input_files():
    backend = GPU4PySCF()
    frags = li_ec_fragments()
    files = backend.files([Calculation((0, 1), frags)], {"ferf": "uncoupled"})
    assert set(files) == {"input.json", "gpu4pyscf_eda.py"}
    data = json.loads(files["input.json"])
    s = data["settings"]
    # the validated defaults, with the one change
    assert (s["xc"], s["basis"], s["auxbasis"]) == ("wb97m-v", "def2-tzvpd", None)
    assert s["svd_threshold"] == 1.0e-4 and s["ferf"] == "uncoupled"
    (calc,) = data["calculations"]
    assert calc["id"] == "0+1"
    assert [f["charge"] for f in calc["fragments"]] == [1, 0]
    assert [f["spin"] for f in calc["fragments"]] == [0, 0]
    assert calc["fragments"][1]["coordinates"][1] == [2.6, 0.76, 0.0]
    assert backend.command()[0] == "{code}"


def test_method_string():
    text = GPU4PySCF().method_string({})
    assert text.startswith("scheme=ALMO-EDA2;backend=GPU4PySCF;xc=wb97m-v")
    assert "auxbasis=none" in text and "ferf=coupled" in text
    assert text.endswith(";grids=150,974/99,590")
    text = GPU4PySCF().method_string({"xc_grid": [99, 590], "nlc_grid": [50, 194]})
    assert text.endswith(";grids=99,590/50,194")


def test_grids_reach_the_driver():
    driver = load_driver()
    backend = GPU4PySCF()
    calcs = [Calculation((0, 1), li_ec_fragments())]
    s = json.loads(backend.files(calcs, {})["input.json"])["settings"]
    assert driver.grid_options(s) == {"xc_grid": (150, 974), "nlc_grid": (99, 590)}
    # an input from before the setting existed keeps GPU4PySCF's defaults
    assert driver.grid_options({"xc": "wb97m-v"}) == {}


def test_driver_helpers(tmp_path):
    driver = load_driver()
    out = tmp_path / "results.jsonl"
    out.write_text(
        json.dumps({"id": "0+1", "eda_kJ_mol": {"total": -1.0}})
        + "\n"
        + json.dumps({"id": "0+2", "error": "boom"})
        + "\nnot json\n"
    )
    assert driver.done_ids(str(out)) == {"0+1"}
    assert driver._plain({"a": np.float64(1.5), "b": [np.int64(2)]}) == {
        "a": 1.5,
        "b": [2.0],
    }


def test_read_results(tmp_path):
    """Shell A's records rewritten in the driver's format."""
    lines, calcs = [], []
    for line in (DATA / "shellA_dp000_svd1e-4.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = tuple(int(x) for x in r["fragments"].split("+"))
        calcs.append(Calculation(key, []))
        if key == (2, 3):
            lines.append(json.dumps({"id": "2+3", "error": "Traceback ..."}))
            continue
        lines.append(json.dumps({"id": r["fragments"], "eda_kJ_mol": r["eda_kJ_mol"]}))
    calcs.append(Calculation((0, 1, 2), []))  # never run
    (tmp_path / "results.jsonl").write_text("\n".join(lines) + "\n")

    results = GPU4PySCF().read_results(tmp_path, calcs)
    assert results[(2, 3)].energies is None and "Traceback" in results[(2, 3)].error
    assert results[(0, 1, 2)].error.startswith("no result")
    assert results[(0, 1)].energies["total"] == pytest.approx(-206.436, abs=1e-3)
    assert results[(0, 1, 2, 3, 4)].energies["total"] == pytest.approx(
        -496.400, abs=1e-3
    )


def test_rerun_replaces_failure(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"id": "0+1", "error": "out of memory"})
        + "\n"
        + json.dumps({"id": "0+1", "eda_kJ_mol": {"total": -2.0}})
        + "\n"
    )
    results = GPU4PySCF().read_results(tmp_path, [Calculation((0, 1), [])])
    assert results[(0, 1)].energies == {"total": -2.0}


def test_svd_warning():
    from interaction_eda_step.interaction_eda import svd_warning

    frags = li_ec_fragments()
    assert svd_warning({"svd_threshold": 1.0e-4}, frags) is None
    assert svd_warning({"svd_threshold": 1.0e-5}, frags) is None
    text = svd_warning({"svd_threshold": 1.0e-3}, frags)
    assert "0:Li+" in text and "1e-4" in text
    neutral = [frags[1], Fragment(2, "H2O", [4], ["O"], np.zeros((1, 3)))]
    assert svd_warning({"svd_threshold": 1.0e-3}, neutral) is None


def test_low_fp64_names():
    driver = load_driver()
    for name in (
        "Tesla T4",
        "NVIDIA L40S",
        "NVIDIA L4",
        "NVIDIA A10",
        "NVIDIA RTX A6000",
    ):
        assert driver.low_fp64(name), name
    for name in (
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A30",
        "NVIDIA H200",
        "NVIDIA B200",
        "Tesla V100-PCIE-16GB",
    ):
        assert not driver.low_fp64(name), name


def test_gpu_refusal():
    """Low-FP64 GPUs are refused unless allowed; V100 and the data-centre GPUs run."""
    driver = load_driver()
    text = driver.gpu_refusal("NVIDIA L40S", {})
    assert "nothing was run" in text and "'allow'" in text
    assert driver.gpu_refusal("NVIDIA L40S", {"allow_low_fp64": True}) is None
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A30", "Tesla V100-PCIE-16GB"):
        assert driver.gpu_refusal(name, {}) is None


def test_refusal_setting_reaches_the_driver():
    backend = GPU4PySCF()
    calcs = [Calculation((0, 1), li_ec_fragments())]
    s = json.loads(backend.files(calcs, {})["input.json"])["settings"]
    assert s["allow_low_fp64"] is False
    s = json.loads(backend.files(calcs, {"allow_low_fp64": True})["input.json"])
    assert s["settings"]["allow_low_fp64"] is True
    assert "fp64" not in backend.method_string({"allow_low_fp64": True})


# ---- Psi4 (ALMO-EDA2 on CPUs), not yet offered by the step ----------------
def load_psi4_driver():
    path = importlib.resources.files("interaction_eda_step") / "data" / "psi4_eda.py"
    spec = importlib.util.spec_from_file_location("psi4_eda", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports nothing from Psi4 at load
    return module


def test_psi4_backend_input_and_command():
    from interaction_eda_step.backends.psi4 import Psi4

    backend = Psi4()
    files = backend.files([Calculation((0, 1), li_ec_fragments())], {})
    assert set(files) == {"input.json", "psi4_eda.py"}
    s = json.loads(files["input.json"])["settings"]
    assert (s["xc"], s["ferf"], s["response_nlc"]) == ("wb97x-v", "coupled", False)
    assert backend.command()[:2] == ["{code}", "psi4_eda.py"]
    assert backend.command()[-2:] == ["--threads", "{NTASKS}"]
    assert (
        backend.configure({"code": "psi4 -n 4", "installation": "conda"})["code"]
        == "python"
    )
    assert "backend=Psi4" in backend.method_string({})


def test_psi4_backend_problems():
    from interaction_eda_step.backends.psi4 import Psi4

    backend = Psi4()
    assert backend.problems({}) == []
    for ferf in ("coupled", "uncoupled"):
        (text,) = backend.problems({"xc": "wb97m-v", "ferf": ferf})
        assert "meta-GGA" in text
    (text,) = backend.problems({"xc": "wb97x-v", "response_nlc": True})
    assert "VV10 response kernel" in text
    assert backend.problems({"xc": "hf"}) == []


def test_psi4_driver_numerics():
    """The driver's linear algebra, without Psi4."""
    d = load_psi4_driver()
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    A = A - A.T
    U = d.expm_antisymmetric(A)
    assert np.allclose(U @ U.T, np.eye(6))
    w, V = np.linalg.eig(A)
    assert np.allclose(U, (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real)

    M = rng.standard_normal((5, 5))
    M = M + M.T
    assert np.allclose(d.unpack_tril(d.pack_tril(M), 5), M)

    S = np.eye(5) + 0.1 * (M @ M.T) / 10
    F = M
    e, C = d.generalized_eigh(F, S)
    assert np.allclose(F @ C, S @ C @ np.diag(e)) and np.all(np.diff(e) >= 0)

    P = rng.standard_normal((8, 8))
    SPD = P @ P.T + 8 * np.eye(8)
    b = rng.standard_normal(8)
    x = d.conjugate_gradient(lambda v: SPD @ v, b, np.zeros(8), lambda r: r, 1e-12)
    assert np.allclose(SPD @ x, b)
    B = rng.standard_normal((8, 3))
    X = d.block_conjugate_gradient(lambda V: SPD @ V, B, lambda R: R, 1e-12)
    assert np.allclose(SPD @ X, B)

    diis = d.DIIS()
    # On a linear problem x -> x - (A x - c), DIIS converges to the solution
    c = rng.standard_normal(4)
    A4 = np.eye(4) + 0.3 * np.diag([1, 2, 3, 4])
    x = np.zeros(4)
    for _ in range(10):
        r = A4 @ x - c
        x = diis.update(x - 0.5 * r, r)
    assert np.allclose(A4 @ x, c, atol=1e-8)
