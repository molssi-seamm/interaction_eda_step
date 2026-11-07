#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The many-body bookkeeping and terms, against stored GPU4PySCF results.

Shell A at d = 0: Li+ + DMC + 2 EC + DMC, wB97M-V/def2-TZVPD, ALMO-EDA2
(GPU4PySCF 1.9.1, ChemAI), the cluster and its 10 pairs. The expected values
are science's reference table (kJ/mol).
"""

import itertools
import json
from pathlib import Path

import pytest

from interaction_eda_step import manybody, terms

DATA = Path(__file__).parent / "data"

# term: (sum of pairs, cluster, >=3-body exact)
REFERENCE = {
    "electrostatic": (-612.816, -580.425, 32.392),
    "classical electrostatic": (-493.805, -493.805, 0.000),
    "pauli": (344.359, 306.287, -38.072),
    "dispersion": (-69.865, -65.267, 4.598),
    "frozen": (-338.323, -339.405, -1.083),
    "polarization": (-283.132, -123.505, 159.627),
    "charge transfer": (-45.484, -33.489, 11.995),
    "total": (-666.939, -496.400, 170.539),
}


def shell_a():
    results = {}
    for line in (DATA / "shellA_dp000_svd1e-4.jsonl").read_text().splitlines():
        record = json.loads(line)
        key = tuple(int(x) for x in record["fragments"].split("+"))
        results[key] = record["eda_kJ_mol"]
    return results


def test_shell_a_reference():
    results = shell_a()
    mb = manybody.analyze(results, 5, terms.ALMO_EDA2.terms)
    assert mb.all_pairs and len(mb.pairs) == 10
    for term, (pairs, cluster, ge3) in REFERENCE.items():
        assert mb.sum_pairs[term] == pytest.approx(pairs, abs=1.5e-3)
        assert mb.cluster[term] == pytest.approx(cluster, abs=1.5e-3)
        assert mb.ge3_exact[term] == pytest.approx(ge3, abs=1.5e-3)
    # classical electrostatics is pairwise additive
    assert abs(mb.ge3_exact["classical electrostatic"]) < 1.0e-6
    assert mb.three_body == {} and mb.sum_three_body is None
    assert mb.n_triples_total == 10


def test_identities():
    for key, energies in shell_a().items():
        assert terms.ALMO_EDA2.check(energies) == [], key
    bad = dict(shell_a()[(0, 1)])
    bad["total"] += 1.0
    assert [t for t, *_ in terms.ALMO_EDA2.check(bad)] == ["total"]


def test_common_terms():
    e = shell_a()[(0, 1)]
    common = terms.ALMO_EDA2.to_common(e)
    assert common["induction"] == pytest.approx(
        e["polarization"] + e["charge transfer"]
    )
    # the SAPT-comparable pair: classical electrostatics and modified Pauli
    assert common["electrostatics"] == e["classical electrostatic"]
    assert common["Pauli"] == pytest.approx(
        e["frozen"] - e["classical electrostatic"] - e["dispersion"]
    )
    assert common["frozen"] == pytest.approx(
        common["electrostatics"] + common["Pauli"] + common["dispersion"]
    )
    assert set(common) == set(terms.COMMON_TERMS)


def test_common_many_body():
    """Shell A's common >=3-body parts: electrostatics exactly 0, and the frozen
    term's nonadditivity all in Pauli (science's numbers, 2026-11-07)."""
    mb = manybody.analyze(shell_a(), 5, terms.ALMO_EDA2.terms)
    ge3 = terms.ALMO_EDA2.to_common(mb.ge3_exact)
    assert ge3["electrostatics"] == pytest.approx(0.0, abs=1.0e-6)
    # frozen - classical (0) - dispersion = -1.083 - 4.598
    assert ge3["Pauli"] == pytest.approx(-5.681, abs=1.5e-3)
    # = native electrostatic + pauli, 32.392 - 38.072
    native = mb.ge3_exact["electrostatic"] + mb.ge3_exact["pauli"]
    assert ge3["Pauli"] == pytest.approx(native, abs=1.0e-6)
    for name in ("frozen", "dispersion", "polarization", "charge transfer", "total"):
        assert ge3[name] == pytest.approx(REFERENCE[name][2], abs=1.5e-3)


def test_three_body_inclusion_exclusion():
    """Synthetic triples: E(ijk) = sum of its pairs + a known 3-body part."""
    results = shell_a()
    three = {}
    for k, key in enumerate(itertools.combinations(range(5), 3)):
        if k % 2:
            continue  # a truncated set of triples
        three[key] = 0.1 * (k + 1)
        pairs = [results[p] for p in itertools.combinations(key, 2)]
        results[key] = {
            t: sum(p[t] for p in pairs) + three[key] for t in terms.ALMO_EDA2.terms
        }
    mb = manybody.analyze(results, 5, terms.ALMO_EDA2.terms)
    assert set(mb.three_body) == set(three)
    for key, value in three.items():
        assert mb.three_body[key]["total"] == pytest.approx(value)
    assert mb.sum_three_body["pauli"] == pytest.approx(sum(three.values()))
    # the exact >=3-body comes from the cluster, unaffected by the triples
    assert mb.ge3_exact["total"] == pytest.approx(170.539, abs=1.5e-3)


def test_missing_pair():
    results = shell_a()
    del results[(2, 3)]
    mb = manybody.analyze(results, 5)
    assert mb.sum_pairs is None and mb.ge3_exact is None
    assert mb.cluster is not None


def test_dimer():
    mb = manybody.analyze({(0, 1): {"total": -1.0}}, 2)
    assert mb.cluster == {"total": -1.0}
    assert mb.sum_pairs == {"total": -1.0}
    assert mb.ge3_exact is None


def test_bad_keys():
    with pytest.raises(manybody.ManyBodyError):
        manybody.analyze({(1, 0): {"total": 0.0}}, 2)
    with pytest.raises(manybody.ManyBodyError):
        manybody.analyze({(0, 5): {"total": 0.0}}, 3)


def test_fragment_sets():
    sets = manybody.fragment_sets(4, ("cluster", "pairs", "triples"), [(2, 1, 0)])
    assert sets[0] == (0, 1, 2, 3)
    assert len(sets) == 1 + 6 + 1 and sets[-1] == (0, 1, 2)
    # a dimer's cluster is its pair: computed once
    assert manybody.fragment_sets(2) == [(0, 1)]
