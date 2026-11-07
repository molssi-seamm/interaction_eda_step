#!/usr/bin/env python
"""Write the GPU validation inputs with the plug-in's own code.

Shell A's fragments come from its SDF through the step's fragment code (molecules
and charges from the configuration); the Li+-EC wall points and the S22 water
dimer from their extxyz keys. The inputs are written with the GPU4PySCF
backend's ``files()``, so the driver runs exactly what the step would send.

    python make_inputs.py OUTDIR

writes OUTDIR/input_coupled.json, OUTDIR/input_uncoupled.json and the driver.
Calculation ids are "<case>:<fragments>", e.g. "shellA:0+1".
"""

import json
from pathlib import Path
import shlex
import sys

import numpy as np

import molsystem
from interaction_eda_step import fragments as F
from interaction_eda_step import manybody
from interaction_eda_step.backends import Calculation, GPU4PySCF

DATA = Path(__file__).parents[2] / "tests" / "data"


def extxyz_frames(path):
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        info = dict(t.split("=", 1) for t in shlex.split(lines[i + 1]) if "=" in t)
        rows = [line.split() for line in lines[i + 2 : i + 2 + n]]
        yield info, [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])
        i += 2 + n


def fragments_from_keys(info, symbols, xyz):
    charges = [int(c) for c in info["fragment_charges"].split(",")]
    out = []
    for index, (spec, charge) in enumerate(zip(info["fragments"].split(";"), charges)):
        name, start, end = spec.split(":")
        atoms = list(range(int(start), int(end)))
        out.append(
            F.Fragment(index, name, atoms, [symbols[a] for a in atoms], xyz[atoms], charge)
        )
    return out


def shell_a_fragments():
    db = molsystem.SystemDB(filename="file:val_db?mode=memory&cache=shared")
    configuration = db.create_system().create_configuration()
    text = (DATA / "shellA_dp000.sdf").read_text().split("$$$$")[0] + "$$$$\n"
    configuration.from_sdf_text(text)
    frags = F.from_configuration(configuration)
    db.close()
    return frags


def calculations(case, frags, sets):
    out = []
    for key in manybody.fragment_sets(len(frags), sets):
        calc = Calculation(key, [frags[i] for i in key])
        out.append((f"{case}:{calc.id}", calc))
    return out


def write(outdir, name, calcs, settings):
    backend = GPU4PySCF()
    files = backend.files([c for _, c in calcs], settings)
    data = json.loads(files["input.json"])
    for entry, (cid, _) in zip(data["calculations"], calcs):
        entry["id"] = cid
    (outdir / name).write_text(json.dumps(data, indent=1))
    (outdir / "gpu4pyscf_eda.py").write_text(files["gpu4pyscf_eda.py"])
    print(f"{name}: {len(calcs)} calculations, settings {data['settings']}")


def main(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shell_a = shell_a_fragments()
    li_ec = [
        (info["name"], fragments_from_keys(info, s, x))
        for info, s, x in extxyz_frames(DATA / "li_ec_wall3.extxyz")
    ]
    water = [
        (info["name"], fragments_from_keys(info, s, x))
        for info, s, x in extxyz_frames(DATA / "water_dimer.extxyz")
    ]

    coupled = calculations("shellA", shell_a, ("cluster", "pairs"))
    for name, frags in li_ec + water:
        coupled += calculations(name, frags, ("cluster",))
    write(outdir, "input_coupled.json", coupled, {})

    uncoupled = calculations("shellA", shell_a, ("cluster",))
    for name, frags in li_ec:
        uncoupled += calculations(name, frags, ("cluster",))
    write(outdir, "input_uncoupled.json", uncoupled, {"ferf": "uncoupled"})


def anion(outdir):
    """PF6- -- EC and Li+ -- PF6- at three FERF SVD thresholds (science, 11-07)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames = [
        (info["name"], fragments_from_keys(info, s, x))
        for info, s, x in extxyz_frames(DATA / "anion_validation.extxyz")
    ]
    for threshold in ("1e-3", "1e-4", "1e-5"):
        calcs = []
        for name, frags in frames:
            calcs += calculations(name, frags, ("cluster",))
        write(
            outdir,
            f"input_anion_svd{threshold}.json",
            calcs,
            {"svd_threshold": float(threshold)},
        )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "anion":
        anion(sys.argv[1])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "validation_run")
