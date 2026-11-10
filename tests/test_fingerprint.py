#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The tasks' fingerprints: only what determines the results."""

import json
from pathlib import Path

import numpy as np

from interaction_eda_step.backends import Calculation, GPU4PySCF
from interaction_eda_step.backends.base import fingerprint
from interaction_eda_step.backends.psi4 import Psi4
from interaction_eda_step.fragments import Fragment

DATA = Path(__file__).parent / "data"


def li_water():
    li = Fragment(0, "Li+", [0], ["Li"], np.zeros((1, 3)), charge=1)
    water = Fragment(
        1,
        "H2O",
        [1, 2, 3],
        ["O", "H", "H"],
        np.array([[2.0, 0, 0], [2.6, 0.76, 0], [2.6, -0.76, 0]]),
    )
    return [li, water]


def psi4_input(settings=None, fragments=None):
    calc = Calculation((0, 1), fragments or li_water())
    return Psi4().files([calc], settings or {})["input.json"]


def test_same_input_same_fingerprint():
    text = psi4_input()
    assert Psi4().fingerprint(text) == Psi4().fingerprint(json.loads(text))
    assert Psi4().fingerprint(text).startswith("eda-sha256:")


def test_run_only_settings_and_driver_do_not_count():
    """Memory, threads and the driver's text leave the fingerprint unchanged."""
    base = Psi4().fingerprint(psi4_input())
    assert Psi4().fingerprint(psi4_input({"memory_gb": 100})) == base
    data = json.loads(psi4_input())
    data["settings"]["threads"] = 32
    assert Psi4().fingerprint(data) == base
    # The driver is a task file, but not part of the fingerprint
    files = Psi4().files([Calculation((0, 1), li_water())], {})
    assert set(files) == {"input.json", "psi4_eda.py"}
    assert Psi4().fingerprint(files["input.json"]) == base


def test_what_determines_the_results_counts():
    base = Psi4().fingerprint(psi4_input())
    assert Psi4().fingerprint(psi4_input({"xc": "b3lyp"})) != base
    assert Psi4().fingerprint(psi4_input({"basis": "def2-svp"})) != base
    moved = li_water()
    moved[1].coordinates = moved[1].coordinates + 0.01
    assert Psi4().fingerprint(psi4_input(fragments=moved)) != base
    # The backend and its results version
    data = json.loads(psi4_input())
    assert fingerprint(data, "Psi4", Psi4.results_version + 1) != base
    assert fingerprint(data, "GPU4PySCF", Psi4.results_version) != base
    assert GPU4PySCF().fingerprint(data) != base


def test_finished_tasks_of_an_older_version_match():
    """A task written by 2026.11.9.5 (ChemAI 5681, pair 0.0+1) and the same
    calculation built now have the same fingerprint, though the driver differs."""
    text = (DATA / "li_pf6_pair_2026.11.9.5_input.json").read_text()
    old = json.loads(text)
    calc = old["calculations"][0]
    fragments = []
    atom = 0
    for f in calc["fragments"]:
        index, name = f["label"].split(":", 1)
        n = len(f["symbols"])
        fragments.append(
            Fragment(
                int(index),
                name,
                list(range(atom, atom + n)),
                list(f["symbols"]),
                np.array(f["coordinates"]),
                charge=f["charge"],
                multiplicity=f["spin"] + 1,
            )
        )
        atom += n
    settings = {k: v for k, v in old["settings"].items() if v is not None}
    now = Psi4().files([Calculation((0, 1), fragments)], settings)["input.json"]
    assert json.loads(now) == old
    assert Psi4().fingerprint(now) == Psi4().fingerprint(text)
