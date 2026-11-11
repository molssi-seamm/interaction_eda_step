# -*- coding: utf-8 -*-

"""How big an EDA calculation is, and what to ask the queue for.

Each calculation (a set of fragments) runs as its own task, so the step must say
before it runs how many cores, how much memory and how much time the task needs.
The size is the number of basis functions (and, with density fitting, auxiliary
functions), counted from the per-element table in ``data/basis_functions.json``
(from basis_set_exchange). The cost and memory models are fits to measured runs:
see the developer notes in ``docs``.
"""

import functools
import importlib.resources
import json
import math

#: The auxiliary basis used to size density-fitted runs when none is named.
DEFAULT_AUX = "def2-universal-jkfit"

#: Time per task outside the calculation itself, added to every estimate: the
#: batch job starting and staging the environments, launching the program, and
#: writing and collecting the results. Measured on TinkerCliffs (ChemAI
#: 5680/5681, 425 tasks, 2026-11-10): launch and collection 7 s median, 41 s at
#: most; a bundle's start before its first task 6 s median, 14 s at the 90th
#: percentile.
TASK_OVERHEAD_SECONDS = 60.0


@functools.lru_cache(maxsize=1)
def _table():
    path = importlib.resources.files("interaction_eda_step") / "data"
    return json.loads((path / "basis_functions.json").read_text())["basis"]


def n_functions(atomic_numbers, basis):
    """The number of (spherical) basis functions of `basis` on these atoms, or
    None if the basis or an element is not in the table."""
    table = _table().get(str(basis).lower())
    if table is None:
        return None
    total = 0
    for z in atomic_numbers:
        n = table.get(str(int(z)))
        if n is None:
            return None
        total += n
    return total


def atomic_numbers(symbols):
    from seamm_mbe.elements import atomic_number

    return [atomic_number(s) for s in symbols]


def size(symbols, basis, auxbasis=None):
    """(nbf, naux) of a set of atoms: the basis functions, and the auxiliary
    functions of `auxbasis` (or of the default fitting basis, for sizing)."""
    z = atomic_numbers(symbols)
    return n_functions(z, basis), n_functions(z, auxbasis or DEFAULT_AUX)


# ---------------------------------------------------------------------------
# Psi4 on CPUs (density fitting, wB97X-V/def2-TZVPD), TinkerCliffs AMD nodes.
# Wall time on 16 threads at measured sizes (2026-11-08/09), interpolated on a
# log-log scale and extrapolated with the last slope (about nbf**2.8); the
# thread scaling is in PSI4_SPEEDUP. Memory: the density-fitted integrals
# (8 bytes * naux * nbf(nbf+1)/2, twice for a range-separated functional) plus a
# fixed part.
# ---------------------------------------------------------------------------
#: Wall time of a pair on 16 threads, by nbf; the 1121-nbf point is shell A (5
#: fragments, 7712 s) divided by fragment_factor(5).
PSI4_T16 = ((116, 66.0), (284, 400.0), (552, 956.0), (1121, 5008.0))
#: The largest memory to ask for: above it the EDA engine reads its integrals
#: from disk (DISK_DF) rather than the task waiting for a bigger node.
PSI4_MAX_GB = 240.0
#: Speed-up relative to 16 threads.
PSI4_SPEEDUP = {4: 0.41, 8: 0.85, 16: 1.0, 32: 1.6, 64: 2.4}
#: Memory besides the JK: about 4 GB plus 0.5 GB per thread (EC-EC: 6.3 GB on
#: 16 threads, 21 GB on 32, 2026-11-09).
PSI4_FIXED_GB = 4.0
PSI4_PER_THREAD_GB = 0.5
#: At least this much per thread: solvent-solvent pairs on 16 threads were killed
#: by the cgroup at 17-19 GB (ChemAI 5681, 2026-11-10).
PSI4_MIN_GB_PER_THREAD = 2.0
#: Psi4's screening keeps about 3/4 of the function pairs (shell A, 1121 nbf:
#: 47.6 GiB needed against 62.8 for all pairs, 2026-11-09).
SCREENED = 0.75
#: The peak memory of a Psi4 EDA is about 1.9 times its in-core integrals (shell
#: A, 1121 nbf: 98 GB at 51 GB, 2026-11-09): the integrals, plus the grids, the
#: matrices and the threads' buffers. Asked for with 30 % to spare. The driver
#: gives the integrals 75 % of Psi4's memory (65 % of this), so they fit in core.
PEAK_OVER_JK = 1.9
PEAK_HEADROOM = 1.3


def df_bytes(nbf, naux, range_separated=True):
    """The in-core MEM_DF tensors of the EDA engine: 8 * naux * nbf**2 bytes for
    all function pairs, three of them for a range-separated functional."""
    return 8 * naux * nbf * nbf * (3 if range_separated else 1)


def psi4_cores(nbf, cores="automatic"):
    """Threads for a Psi4 EDA of `nbf` basis functions."""
    if str(cores) != "automatic":
        return max(1, int(cores))
    if nbf <= 400:
        return 8
    if nbf <= 800:
        return 16
    return 32


def psi4_memory(nbf, naux, density_fitting=True, memory="automatic", cores=16):
    """Bytes of memory for a Psi4 EDA task (what the queue is asked for).

    Enough for the peak, about 1.9 times the screened in-core integrals, with
    30 % to spare, and at least 2 GB per thread. Psi4 gets 65 % of it, and the
    EDA engine's in-core JK 75 % of that, so the integrals stay in core (MEM_DF)
    rather than being read from disk. The SCFs use DiskDFJK, which needs less.
    Shell A (1121 nbf) peaked at 98 GB with 160 GB for Psi4; this gives 127 GB."""
    if str(memory) != "automatic":
        return int(memory)
    jk = SCREENED * df_bytes(nbf, naux) / 1e9 if density_fitting and naux else 0.0
    gb = 1.25 * (PSI4_FIXED_GB + PSI4_PER_THREAD_GB * cores + jk)
    if jk:
        gb = max(gb, PEAK_HEADROOM * PEAK_OVER_JK * jk + 2.0)
    gb = max(gb, PSI4_MIN_GB_PER_THREAD * cores)
    return int(math.ceil(min(gb, PSI4_MAX_GB)) * 1e9)


def _loglog(points, x):
    """Interpolate (x, y) points on a log-log scale; extrapolate with the end
    slopes."""
    points = sorted(points)
    if x <= points[0][0]:
        (x0, y0), (x1, y1) = points[0], points[1]
    elif x >= points[-1][0]:
        (x0, y0), (x1, y1) = points[-2], points[-1]
    else:
        i = max(k for k in range(len(points)) if points[k][0] <= x)
        (x0, y0), (x1, y1) = points[i], points[i + 1]
    slope = math.log(y1 / y0) / math.log(x1 / x0)
    return y0 * (x / x0) ** slope


def fragment_factor(n_fragments):
    """More fragments, more SCF-MI and FERF work at the same size: na6_min (7
    fragments, 383 nbf) took 1362 s on 8 threads (ChemAI 5679), about twice a
    pair; 0.18 per extra fragment covers it with the task overhead."""
    return 1.0 + 0.18 * max(0, int(n_fragments) - 2)


def psi4_seconds(nbf, cores, n_fragments=2):
    """Estimated wall time of a Psi4 EDA task, with the task's overhead."""
    t = _loglog(PSI4_T16, nbf) * fragment_factor(n_fragments)
    t /= _loglog(list(PSI4_SPEEDUP.items()), max(1, int(cores)))
    return TASK_OVERHEAD_SECONDS + t


# ---------------------------------------------------------------------------
# GPU4PySCF (one GPU). V100 times (2026-11-07): 284 -> 87 s, 552 -> 200 s,
# 1121 -> 1538 s, about nbf**2.4; host memory is small.
# ---------------------------------------------------------------------------
GPU_T = (87.0, 284)
GPU_EXPONENT = 2.4
GPU_CORES = 4
GPU_MEMORY_GB = 32


def gpu_seconds(nbf, n_fragments=2):
    """Estimated wall time of a GPU4PySCF EDA task, with the task's overhead."""
    t, n = GPU_T
    t *= (nbf / n) ** GPU_EXPONENT * fragment_factor(n_fragments)
    return TASK_OVERHEAD_SECONDS + t
