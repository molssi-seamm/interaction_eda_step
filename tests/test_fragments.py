#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fragments from a SEAMM configuration."""

from pathlib import Path

import molsystem
import pytest

from interaction_eda_step import fragments as F

DATA = Path(__file__).parent / "data"


@pytest.fixture()
def configuration():
    db = molsystem.SystemDB(filename="file:seamm_db?mode=memory&cache=shared")
    system = db.create_system()
    yield system.create_configuration()
    db.close()


def first_sdf_record(path):
    text = path.read_text()
    return text.split("$$$$")[0] + "$$$$\n"


def test_shell_a_molecules(configuration):
    configuration.from_sdf_text(first_sdf_record(DATA / "shellA_dp000.sdf"))
    frags = F.from_configuration(configuration)
    assert len(frags) == 5
    assert [f.charge for f in frags] == [1, 0, 0, 0, 0]
    assert frags[0].symbols == ["Li"] and frags[0].name == "Li+"
    assert sorted(f.name for f in frags[1:]) == ["DMC", "DMC", "EC", "EC"]
    assert sum(len(f.atoms) for f in frags) == 45
    # the reference calculations used the same fragment order
    assert [len(f.atoms) for f in frags] == [1, 12, 10, 10, 12]
    # every fragment's coordinates are the configuration's
    xyz = configuration.atoms.get_coordinates(fractionals=False)
    for f in frags:
        for k, a in enumerate(f.atoms):
            assert list(f.coordinates[k]) == pytest.approx(list(xyz[a]))


def test_triples_within(configuration):
    configuration.from_sdf_text(first_sdf_record(DATA / "shellA_dp000.sdf"))
    frags = F.from_configuration(configuration)
    assert len(F.triples_within(frags, None)) == 10
    assert F.triples_within(frags, 0.1) == []
    near = F.triples_within(frags, 3.0)
    assert 0 < len(near) < 10
    for key in near:
        for i, j in [(key[0], key[1]), (key[0], key[2]), (key[1], key[2])]:
            assert F.min_distance(frags[i], frags[j]) <= 3.0


def test_one_molecule_is_an_error(configuration):
    configuration.from_sdf_text(
        "water\n  test\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.1173 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000   -0.7572   -0.4692 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0\n  1  3  1  0\nM  END\n$$$$\n"
    )
    with pytest.raises(F.FragmentError, match="at least two"):
        F.from_configuration(configuration)
