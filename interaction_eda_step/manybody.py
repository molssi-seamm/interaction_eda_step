# -*- coding: utf-8 -*-

"""Many-body bookkeeping of EDA interaction energies, term by term.

Every EDA number is an *interaction* energy of a set of fragments, relative to
the same fragments alone at the same geometry, so single fragments contribute
nothing. For the fragments i, j, k, ... of a cluster:

* the 2-body part of a term is the sum over the pairs, E(ij);
* **≥3-body (exact)** = E(cluster) - Σ pairs E(ij). It needs the cluster and
  every pair, and it is exact;
* the 3-body increment of a triple is, by inclusion--exclusion,
  ΔE(ijk) = E(ijk) - E(ij) - E(ik) - E(jk). Summed over the triples that were
  computed (e.g. those within a cutoff) it is the **3-body (triples,
  truncated)** estimate. It is a truncated sum and must not be read as the
  ≥3-body total; with all triples, the difference between the two is the
  ≥4-body part.

Fragment sets are tuples of 0-based fragment indices in ascending order.
"""

from dataclasses import dataclass, field
import itertools


class ManyBodyError(ValueError):
    """Results needed for a many-body term are missing."""


@dataclass
class ManyBody:
    """The many-body analysis of one structure.

    Attributes
    ----------
    n_fragments : int
    terms : (str, ...)
        The terms analyzed, in order.
    cluster : {str: float} or None
        The whole cluster's terms.
    pairs : {(int, int): {str: float}}
    triples : {(int, int, int): {str: float}}
        The triples' own interaction energies.
    sum_pairs : {str: float} or None
        Σ over all pairs; None unless every pair is present.
    ge3_exact : {str: float} or None
        cluster - Σ pairs; None unless the cluster and every pair are present.
    three_body : {(int, int, int): {str: float}}
        The 3-body increment of each triple whose pairs are present.
    sum_three_body : {str: float} or None
        Σ of the 3-body increments over the triples computed (truncated).
    n_triples_total : int
        How many triples the cluster has, to show how truncated the sum is.
    """

    n_fragments: int
    terms: tuple
    cluster: dict = None
    pairs: dict = field(default_factory=dict)
    triples: dict = field(default_factory=dict)
    sum_pairs: dict = None
    ge3_exact: dict = None
    three_body: dict = field(default_factory=dict)
    sum_three_body: dict = None
    n_triples_total: int = 0

    @property
    def all_pairs(self):
        return len(self.pairs) == self.n_fragments * (self.n_fragments - 1) // 2


def _sum(dicts, terms):
    return {t: sum(d[t] for d in dicts) for t in terms}


def _diff(a, b, terms):
    return {t: a[t] - b[t] for t in terms}


def analyze(results, n_fragments, terms=None):
    """Do the many-body bookkeeping.

    Parameters
    ----------
    results : {tuple(int, ...): {str: float}}
        The terms (kJ/mol) of each computed set of fragments, keyed by the
        ascending fragment indices. The whole cluster's key is
        ``tuple(range(n_fragments))``.
    n_fragments : int
        The number of fragments in the cluster.
    terms : [str] or None
        The terms to analyze; default those common to all results, in the
        order of the first result.

    Returns
    -------
    ManyBody
    """
    if n_fragments < 2:
        raise ManyBodyError("An interaction needs at least two fragments.")
    for key in results:
        if (
            len(key) < 2
            or list(key) != sorted(set(key))
            or key[0] < 0
            or key[-1] >= n_fragments
        ):
            raise ManyBodyError(f"Bad fragment set {key!r} for {n_fragments} fragments")
    if terms is None:
        first = next(iter(results.values()), {})
        terms = [t for t in first if all(t in r for r in results.values())]
    terms = tuple(terms)

    everything = tuple(range(n_fragments))
    mb = ManyBody(n_fragments=n_fragments, terms=terms)
    mb.n_triples_total = n_fragments * (n_fragments - 1) * (n_fragments - 2) // 6
    for key, energies in results.items():
        energies = {t: float(energies[t]) for t in terms}
        if key == everything:
            mb.cluster = energies
        if len(key) == 2:
            mb.pairs[key] = energies
        elif len(key) == 3:
            mb.triples[key] = energies

    if mb.all_pairs:
        mb.sum_pairs = _sum(mb.pairs.values(), terms)
        if mb.cluster is not None and n_fragments > 2:
            mb.ge3_exact = _diff(mb.cluster, mb.sum_pairs, terms)

    for key, energies in sorted(mb.triples.items()):
        sub = [pair for pair in itertools.combinations(key, 2)]
        if all(pair in mb.pairs for pair in sub):
            mb.three_body[key] = _diff(
                energies, _sum([mb.pairs[p] for p in sub], terms), terms
            )
    if mb.three_body:
        mb.sum_three_body = _sum(mb.three_body.values(), terms)
    return mb


def fragment_sets(n_fragments, sets=("cluster", "pairs"), triples=None):
    """The sets of fragments to compute.

    Parameters
    ----------
    n_fragments : int
    sets : [str]
        Any of "cluster", "pairs" and "triples".
    triples : [(int, int, int)] or None
        The triples to use when "triples" is asked for; default all of them.

    Returns
    -------
    [tuple(int, ...)]
        Each set once, the cluster first, then pairs and triples in order.
    """
    out = []
    if "cluster" in sets:
        out.append(tuple(range(n_fragments)))
    if "pairs" in sets:
        out.extend(itertools.combinations(range(n_fragments), 2))
    if "triples" in sets:
        if triples is None:
            triples = itertools.combinations(range(n_fragments), 3)
        out.extend(tuple(sorted(t)) for t in triples)
    seen, unique = set(), []
    for key in out:
        if key not in seen and len(key) >= 2:
            seen.add(key)
            unique.append(key)
    return unique
