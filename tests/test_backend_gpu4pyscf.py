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
