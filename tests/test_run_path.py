#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The step's whole run path, with a stand-in for GPU4PySCF.

A flowchart reads shell A (Li+ + DMC + 2 EC + DMC, from an SDF with Li+'s
charge) and runs the Interaction EDA step. The "Python" of the GPU4PySCF
environment is a script that answers each calculation of ``input.json`` with
the stored GPU4PySCF results for that set of fragments, so the fragments the
step makes, the files it writes, the many-body tables and the stored results
are all checked against the real numbers.
"""

import json
from pathlib import Path
import shlex
import sys

import pytest

seamm_testing = pytest.importorskip("seamm_exec.testing")

DATA = Path(__file__).parent / "data"
SOURCE = Path(__file__).parents[1]

FAKE = '''#!{python}
"""Stand-in for the GPU4PySCF environment's Python."""
import json, os, sys
reference = {{}}
for line in open({reference!r}):
    r = json.loads(line)
    reference[r["fragments"]] = r["eda_kJ_mol"]
args = sys.argv[1:]
assert args[0] == "gpu4pyscf_eda.py", args
data = json.load(open(args[1]))
assert data["settings"]["xc"] == "wb97m-v", data["settings"]
assert data["settings"]["auxbasis"] is None, data["settings"]
# like the real driver: one log per calculation, in --logdir (default "logs")
logdir = args[args.index("--logdir") + 1] if "--logdir" in args else "logs"
os.makedirs(logdir, exist_ok=True)
with open(args[args.index("-o") + 1], "a") as out:
    for calc in data["calculations"]:
        with open(os.path.join(logdir, calc["id"] + ".log"), "w") as log:
            log.write("fake log\\n")
        charges = [f["charge"] for f in calc["fragments"]]
        assert charges == [1 if f["label"].startswith("0:") else 0
                           for f in calc["fragments"]], calc["id"]
        record = {{"id": calc["id"], "gpu": "fake", "wall_s": 1.0}}
        if calc["id"] in reference:
            record["eda_kJ_mol"] = reference[calc["id"]]
        else:
            record["error"] = "no stored result"
        out.write(json.dumps(record) + "\\n")
'''

SPEC = """
title: Interaction EDA of shell A
steps:
- Read Structure:
    file: {sdf}
- Interaction EDA: {{}}
"""


def test_shell_a(tmp_path):
    sdf = tmp_path / "shellA.sdf"
    text = (DATA / "shellA_dp000.sdf").read_text()
    sdf.write_text(text.split("$$$$")[0] + "$$$$\n")

    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text(
        FAKE.format(
            python=sys.executable,
            reference=str(DATA / "shellA_dp000_svd1e-4.jsonl"),
        )
    )
    fake.chmod(0o755)

    job = seamm_testing.run_spec(
        tmp_path,
        SPEC.format(sdf=sdf),
        inis={"gpu4pyscf": {"installation": "local", "code": str(fake)}},
        source=SOURCE,
    )
    steps = [p for p in job.iterdir() if (p / "eda.json").exists()]
    assert len(steps) == 1, list(job.iterdir())
    step = steps[0]

    result = json.loads((step / "eda.json").read_text())
    assert [f["charge"] for f in result["fragments"]] == [1, 0, 0, 0, 0]
    assert len(result["calculations"]) == 11
    assert all(c["error"] is None for c in result["calculations"].values())
    assert result["ge3body_exact"]["total"] == pytest.approx(170.539, abs=1.5e-3)
    assert result["ge3body_exact"]["polarization"] == pytest.approx(159.627, abs=1.5e-3)
    assert result["sum_pairs"]["pauli"] == pytest.approx(344.359, abs=1.5e-3)

    frames = (step / "eda.extxyz").read_text()
    info = dict(token.split("=", 1) for token in shlex.split(frames.splitlines()[1]))
    assert float(info["EDA_ALMOEDA2_total"]) == pytest.approx(-496.400, abs=1.5e-3)
    assert float(info["EDA_ALMOEDA2_total_ge3body_exact"]) == pytest.approx(
        170.539, abs=1.5e-3
    )
    induction = float(info["EDA_ALMOEDA2_polarization"]) + float(
        info["EDA_ALMOEDA2_charge_transfer"]
    )
    assert float(info["EDA_ALMOEDA2_induction"]) == pytest.approx(induction)
    assert "ferf=coupled" in info["EDA_method"]
    assert "BSSE" in info["EDA_conventions"]
    assert frames.count("Properties=") == 11

    assert (step / "0+1.log").exists() and (step / "0+1+2+3+4.log").exists()

    out = (step / "step.out").read_text()
    assert "≥3-body (exact)" in out and "ALMO-EDA2 interaction energies" in out


REFUSING = '''#!{python}
"""Stand-in for the GPU4PySCF environment's Python on a low-FP64 GPU."""
import sys
print("REFUSED: The NVIDIA L40S has little double-precision (FP64) throughput, "
      "so the EDA would be very slow: nothing was run.", file=sys.stderr)
sys.exit(3)
'''


def test_low_fp64_gpu_refused(tmp_path):
    sdf = tmp_path / "shellA.sdf"
    text = (DATA / "shellA_dp000.sdf").read_text()
    sdf.write_text(text.split("$$$$")[0] + "$$$$\n")
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text(REFUSING.format(python=sys.executable))
    fake.chmod(0o755)
    with pytest.raises(AssertionError) as e:
        seamm_testing.run_spec(
            tmp_path,
            SPEC.format(sdf=sdf),
            inis={"gpu4pyscf": {"installation": "local", "code": str(fake)}},
            source=SOURCE,
        )
    assert "little double-precision (FP64)" in str(e.value)
    assert "All of the EDA calculations failed" not in str(e.value)
