#!/usr/bin/env python
"""Compare the driver's results with the stored GPU4PySCF results, term by term.

    python compare.py results_coupled.jsonl results_uncoupled.jsonl [--gate 0.01]

Every term of every calculation, and shell A's 2-body sum and exact >=3-body
part, must be within the gate (kJ/mol) of the stored results (ChemAI job 14834,
science's shell A file, the ARC uncoupled runs).
"""

import argparse
import json
from pathlib import Path

from interaction_eda_step import manybody, terms

DATA = Path(__file__).parents[2] / "tests" / "data"


def load(path, by="id"):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r[by] if by in r else r["name"]] = r
    return out


def references(uncoupled=False):
    """{"<case>:<fragments>": {term: kJ/mol}}."""
    ref = {}
    if uncoupled:
        files = [("li_ec_wall3_uncoupled_ref.jsonl", None)]
        files.append(("shellA_dp000_uncoupled_ref.jsonl", "shellA"))
    else:
        files = [
            ("li_ec_wall3_ref.jsonl", None),
            ("water_dimer_ref.jsonl", None),
            ("shellA_dp000_svd1e-4.jsonl", "shellA"),
        ]
    for name, case in files:
        for line in (DATA / name).read_text().splitlines():
            r = json.loads(line)
            if case is None:
                key = f"{r['name']}:0+1"
            else:
                key = f"{case}:{r['fragments']}"
            ref[key] = r["eda_kJ_mol"]
    return ref


def compare(results_path, uncoupled, gate):
    results = load(results_path)
    ref = references(uncoupled)
    worst, failures, rows = 0.0, [], []
    for key, expected in sorted(ref.items()):
        r = results.get(key)
        if r is None or "eda_kJ_mol" not in r:
            failures.append(f"{key}: no result ({(r or {}).get('error', 'missing')})")
            continue
        got = r["eda_kJ_mol"]
        diff = max(abs(got[t] - expected[t]) for t in expected)
        worst = max(worst, diff)
        rows.append(
            f"  {key:28s} total {got['total']:10.3f} (ref {expected['total']:10.3f})"
            f"  max|diff| {diff:.2e}  {r.get('wall_s')} s {r.get('gpu', '')}"
        )
        if diff > gate:
            failures.append(f"{key}: max |diff| {diff:.4f} kJ/mol > {gate}")

    # shell A many-body parts from the new results
    shell = {
        tuple(int(x) for x in k.split(":")[1].split("+")): v["eda_kJ_mol"]
        for k, v in results.items()
        if k.startswith("shellA:") and "eda_kJ_mol" in v
    }
    shell_ref = {
        tuple(int(x) for x in k.split(":")[1].split("+")): v
        for k, v in ref.items()
        if k.startswith("shellA:")
    }
    if len(shell) == len(shell_ref) == 11:
        new = manybody.analyze(shell, 5, terms.ALMO_EDA2.terms)
        old = manybody.analyze(shell_ref, 5, terms.ALMO_EDA2.terms)
        for part in ("sum_pairs", "ge3_exact"):
            a, b = getattr(new, part), getattr(old, part)
            diff = max(abs(a[t] - b[t]) for t in a)
            worst = max(worst, diff)
            rows.append(f"  shellA {part:21s} max|diff| {diff:.2e}")
            if diff > gate:
                failures.append(f"shellA {part}: max |diff| {diff:.4f} > {gate}")
    for key, r in results.items():
        bad = terms.ALMO_EDA2.check(r.get("eda_kJ_mol", {}), tolerance=1e-6)
        if bad:
            failures.append(f"{key}: identities fail {bad}")
    return worst, failures, rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("coupled")
    p.add_argument("uncoupled", nargs="?")
    p.add_argument("--gate", type=float, default=0.01)
    a = p.parse_args()
    ok = True
    for path, uncoupled in ((a.coupled, False), (a.uncoupled, True)):
        if not path:
            continue
        worst, failures, rows = compare(path, uncoupled, a.gate)
        label = "uncoupled" if uncoupled else "coupled"
        print(f"{label}: {path}")
        print("\n".join(rows))
        print(f"  worst |diff| {worst:.2e} kJ/mol; gate {a.gate}")
        for f in failures:
            print("  FAIL", f)
        ok = ok and not failures
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
