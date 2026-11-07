#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The GPU4PySCF installer's decisions, without conda or a GPU."""

from interaction_eda_step import installer


def test_build_choice():
    choose = installer.choose_build
    build, why = choose(system="Darwin", driver=None)
    assert build is None and "Linux" in why
    assert choose(system="Linux", driver=(12, 4))[0] == "cuda12x"
    assert choose(system="Linux", driver=(13, 0))[0] == "cuda12x"
    build, why = choose(system="Linux", driver=None)
    assert build == "cuda12x" and "login node" in why
    build, why = choose(system="Linux", driver=(11, 8))
    assert build is None and "CUDA 12" in why
    assert choose("cuda13x", system="Linux", driver=None)[0] == "cuda13x"
    assert choose("auto", system="Linux", driver=(12, 9))[0] == "cuda12x"
    assert choose("cu128", system="Linux", driver=None)[0] is None
    # forcing never installs on macOS
    assert choose("cuda12x", system="Darwin", driver=None)[0] is None


def test_builds():
    for build, reqs in installer.BUILDS.items():
        assert reqs[0].startswith(f"gpu4pyscf-{build}>=1.9")
        assert any(r.startswith("cupy-") and r.endswith("[ctk]") for r in reqs)
        assert "pyscf-dispersion" in reqs


def test_parse():
    out = 'noise\nPROBE{"gpus": ["NVIDIA A30"]}\nmore'
    assert installer.parse(out, "PROBE") == {"gpus": ["NVIDIA A30"]}
    assert installer.parse("VERSION1.9.1\n", "VERSION") == "1.9.1"
    assert installer.parse('LINKED["/a/libcusolver.so"]', "LINKED") == [
        "/a/libcusolver.so"
    ]
    assert installer.parse(None, "PROBE") is None
    assert installer.parse("nothing", "PROBE") is None


def test_assess():
    good = {
        "cupy": "14.2.0",
        "gpus": ["NVIDIA A30"],
        "gpu_ok": True,
        "gpu4pyscf": "1.9.1",
        "pyscf": "2.15.0",
        "almo_eda2": True,
    }
    ok, lines = installer.assess(good)
    assert ok and any("A30" in line for line in lines)
    ok, lines = installer.assess({**good, "gpus": ["Tesla T4"]})
    assert ok and any("poor double precision" in line for line in lines)
    ok, lines = installer.assess({**good, "almo_eda2": False})
    assert not ok
    ok, _ = installer.assess({**good, "gpu_ok": False})
    assert not ok
    # a login node: no GPU and GPU4PySCF does not import, which is expected
    login = {"gpu_error": "CUDARuntimeError", "import_error": "ImportError: x"}
    ok, lines = installer.assess(login, have_driver=False)
    assert ok and "as expected" in lines[0]
    assert not installer.assess(login, have_driver=True)[0]
    assert not installer.assess(None)[0]


def test_exists_by_path(tmp_path):
    """An environment given by its path is found by its conda-meta."""
    inst = installer.Installer.__new__(
        installer.Installer
    )  # no seamm-manager set-up needed
    env = tmp_path / "env"
    assert not inst._exists(str(env))
    (env / "conda-meta").mkdir(parents=True)
    assert inst._exists(str(env))
