#!/usr/bin/env python
"""Second-generation ALMO-EDA with GPU4PySCF, for SEAMM's Interaction EDA step.

Runs in the GPU4PySCF environment, which needs nothing from SEAMM:

    python gpu4pyscf_eda.py input.json [-o results.jsonl]

``input.json``::

    {"settings": {"xc": "wb97m-v", "basis": "def2-tzvpd", "auxbasis": null,
                  "nlc": null, "disp": null, "svd_threshold": null,
                  "ferf": "coupled", "response_nlc": true, "verbose": 3},
     "calculations": [{"id": "0+1",
                       "fragments": [{"label": "0:Li+", "symbols": ["Li"],
                                      "coordinates": [[0.0, 0.0, 0.0]],
                                      "charge": 1, "spin": 0}, ...]}, ...]}

``nlc: false`` removes VV10 from the whole functional; ``response_nlc: false``
removes it only from the FERF response kernel (the SCF and SCF-MI keep it).
Coordinates are in Å. Each calculation appends one JSON line to the results file:
the ALMO-EDA terms in kJ/mol, the energies in hartree, the wall time, the peak GPU
memory and the versions -- or the error. Calculations already in the results file
are skipped, so a job that ran out of time can simply be run again.

The GPU needs good double-precision throughput (A100, A30, H100/H200, B200).
"""

import argparse
import functools
import json
import os
import re
import sys
import threading
import time
import traceback

# GPUs made for single precision: FP64 at 1/32-1/64 of FP32 (whole words, so
# that "A10" does not match "A100").
LOW_FP64 = re.compile(r"\b(T4|L4|L40S?|A10|A16|A40|RTX|GEFORCE)\b", re.IGNORECASE)


def low_fp64(gpu):
    """Whether the GPU, by name, has little double-precision throughput."""
    return LOW_FP64.search(gpu) is not None


class PeakGPUMemory:
    """Sample the device's used memory (all processes on it) in a thread."""

    def __init__(self, cp, interval=0.1):
        self.cp = cp
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.peak = max(self.peak, total - free)
            time.sleep(self.interval)

    def __enter__(self):
        self.base = self.cp.cuda.runtime.memGetInfo()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self.thread.join()


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path) as fd:
            for line in fd:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if "eda_kJ_mol" in record:
                    ids.add(record["id"])
    return ids


def versions():
    out = {"python": sys.version.split()[0]}
    for name in ("gpu4pyscf", "pyscf", "cupy"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "?")
        except Exception:  # pragma: no cover
            out[name] = None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input")
    p.add_argument("-o", "--output", default="results.jsonl")
    p.add_argument("--logdir", default="logs")
    args = p.parse_args(argv)

    with open(args.input) as fd:
        data = json.load(fd)
    settings = data.get("settings", {})
    calculations = data["calculations"]

    import cupy as cp
    import pyscf
    import gpu4pyscf.properties.eda as eda_module

    ferf_options = {}
    if settings.get("svd_threshold") is not None:
        ferf_options["virtual_singular_value_threshold"] = float(
            settings["svd_threshold"]
        )
    if settings.get("ferf", "coupled") == "uncoupled":
        ferf_options["uncoupled_ferf"] = True
    if ferf_options:
        eda_module._get_eda_polarization_FERF_subspace_projector = functools.partial(
            eda_module._get_eda_polarization_FERF_subspace_projector, **ferf_options
        )

    if not settings.get("response_nlc", True):
        # VV10 off in the FERF response kernel only; the SCF and SCF-MI keep the
        # full functional (the Psi4-parity mode). eda.py looks gen_vind up at
        # call time, so rebinding the module attribute is enough.
        import gpu4pyscf.properties.polarizability as polarizability

        _gen_vind = polarizability.gen_vind

        def gen_vind_no_nlc(mf, mo_coeff, mo_occ, with_nlc=True):
            return _gen_vind(mf, mo_coeff, mo_occ, with_nlc=False)

        polarizability.gen_vind = gen_vind_no_nlc

    nlc = settings.get("nlc")
    if isinstance(nlc, str) and nlc.lower() in ("false", "no", "none", "off", "0", ""):
        nlc = False
    xc = settings.get("xc", "wb97m-v")
    basis = settings.get("basis", "def2-tzvpd")
    auxbasis = settings.get("auxbasis")
    disp = settings.get("disp")
    verbose = int(settings.get("verbose", 3))

    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"]
    gpu = gpu.decode() if isinstance(gpu, bytes) else str(gpu)
    info = versions()
    print(f"GPU: {gpu}; {info}", flush=True)
    if low_fp64(gpu):
        print(
            f"WARNING: the {gpu} has little double-precision (FP64) throughput, so "
            "this will be very slow. Use an A100, A30, H100/H200 or B200.",
            flush=True,
        )
    os.makedirs(args.logdir, exist_ok=True)
    skip = done_ids(args.output)

    n_failed = 0
    with open(args.output, "a") as out:
        for calc in calculations:
            cid = calc["id"]
            if cid in skip:
                print(f"{cid}: already done", flush=True)
                continue
            log = os.path.join(args.logdir, f"{cid}.log")
            with open(log, "w") as logfd:
                record = {"id": cid, "gpu": gpu, "versions": info}
                t0 = time.time()
                try:
                    mols = []
                    for frag in calc["fragments"]:
                        geom = "\n".join(
                            "{} {:.10f} {:.10f} {:.10f}".format(s, *xyz)
                            for s, xyz in zip(frag["symbols"], frag["coordinates"])
                        )
                        m = pyscf.M(
                            atom=geom,
                            basis=basis,
                            charge=int(frag.get("charge", 0)),
                            spin=int(frag.get("spin", 0)),
                            verbose=0,
                        )
                        m.stdout = logfd  # one log for all fragments
                        m.verbose = verbose
                        mols.append(m)
                    cp.get_default_memory_pool().free_all_blocks()
                    with PeakGPUMemory(cp) as mem:
                        eda, dft = eda_module.eval_ALMO_EDA_2_energies(
                            mols,
                            xc=xc,
                            auxbasis=auxbasis,
                            verbose=verbose,
                            nlc=nlc,
                            disp=disp,
                        )
                    cp.cuda.runtime.deviceSynchronize()
                    record["eda_kJ_mol"] = {
                        k: float(v) for k, v in eda.items() if k != "unit"
                    }
                    record["energies_hartree"] = _plain(dft.get("energy"))
                    record["nao"] = int(sum(m.nao for m in mols))
                    record["peak_gpu_mem_GB"] = round(mem.peak / 1e9, 3)
                except Exception:
                    n_failed += 1
                    record["error"] = traceback.format_exc()
                    print(f"{cid}: FAILED\n{record['error']}", flush=True)
                record["wall_s"] = round(time.time() - t0, 2)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if "eda_kJ_mol" in record:
                e = record["eda_kJ_mol"]
                print(
                    f"{cid}: total {e['total']:.3f} kJ/mol ({record['wall_s']} s, "
                    f"{record['peak_gpu_mem_GB']} GB)",
                    flush=True,
                )
    return 1 if n_failed else 0


def _plain(value):
    """JSON-safe copy of numbers, lists and dicts (numpy/cupy scalars too)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    sys.exit(main())
