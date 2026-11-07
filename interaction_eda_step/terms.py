# -*- coding: utf-8 -*-

"""The energy terms of each EDA scheme, and how they map onto common terms.

Each scheme keeps its own ("native") terms, with the names the code uses. A few
*common* terms let results from different schemes be put side by side, but only
where the physics really is the same:

* SAPT's induction contains both polarization and charge transfer, so it maps to
  ALMO's polarization **plus** charge transfer ("induction" here). Only schemes
  that split induction apart (ALMO-EDA) fill "polarization" and "charge
  transfer"; SAPT never does. Treating SAPT induction as polarization is the
  lumping that hides charge-transfer problems, e.g. for Li⁺.
* "frozen" is ALMO's frozen term (electrostatics + Pauli + dispersion). SAPT has
  no such term and leaves it empty.

All energies are in kJ/mol.
"""

from dataclasses import dataclass, field

COMMON_TERMS = (
    "electrostatics",
    "Pauli",
    "dispersion",
    "frozen",
    "polarization",
    "charge transfer",
    "induction",
    "total",
)


@dataclass(frozen=True)
class Scheme:
    """An EDA scheme: its native terms, identities, and common-term map.

    Attributes
    ----------
    name : str
        e.g. "ALMO-EDA2".
    terms : (str, ...)
        The native terms, in the code's names and print order.
    common : {str: (str, ...)}
        Each common term as the sum of native terms.
    identities : ((str, (str, ...)), ...)
        Native term = sum of other native terms, which hold exactly.
    pairwise_additive : (str, ...)
        Native terms that are exactly pairwise additive (no ≥3-body part).
    max_fragments : int or None
        The most fragments one calculation can have; None for no limit.
    notes : (str, ...)
        What the terms mean, for the output and the extxyz labels.
    """

    name: str
    terms: tuple
    common: dict = field(default_factory=dict)
    identities: tuple = ()
    pairwise_additive: tuple = ()
    max_fragments: int = None
    notes: tuple = ()

    def to_common(self, energies):
        """The common terms of one result.

        Parameters
        ----------
        energies : {str: float}
            Native terms, kJ/mol.

        Returns
        -------
        {str: float}
            The common terms this scheme defines.
        """
        out = {}
        for name, parts in self.common.items():
            if all(p in energies for p in parts):
                out[name] = sum(energies[p] for p in parts)
        return out

    def check(self, energies, tolerance=1.0e-6):
        """The identities that do not hold, as (term, value, sum of parts).

        Parameters
        ----------
        energies : {str: float}
            Native terms, kJ/mol.
        tolerance : float
            kJ/mol.

        Returns
        -------
        [(str, float, float)]
            Empty when everything is consistent.
        """
        bad = []
        for name, parts in self.identities:
            if name in energies and all(p in energies for p in parts):
                total = sum(energies[p] for p in parts)
                if abs(energies[name] - total) > tolerance:
                    bad.append((name, energies[name], total))
        return bad


ALMO_EDA2 = Scheme(
    name="ALMO-EDA2",
    terms=(
        "electrostatic",
        "classical electrostatic",
        "pauli",
        "dispersion",
        "frozen",
        "polarization",
        "charge transfer",
        "total",
    ),
    common={
        "electrostatics": ("electrostatic",),
        "Pauli": ("pauli",),
        "dispersion": ("dispersion",),
        "frozen": ("frozen",),
        "polarization": ("polarization",),
        "charge transfer": ("charge transfer",),
        "induction": ("polarization", "charge transfer"),
        "total": ("total",),
    },
    identities=(
        ("frozen", ("electrostatic", "pauli", "dispersion")),
        ("total", ("frozen", "polarization", "charge transfer")),
    ),
    pairwise_additive=("classical electrostatic",),
    notes=(
        "Second-generation ALMO-EDA (Horn and Head-Gordon): frozen = electrostatic"
        " + Pauli + dispersion; polarization from fragment electric-response"
        " functions (FERF); charge transfer is the rest of the SCF relaxation.",
        "The fragments are at their geometry in the cluster: there is no"
        " preparation (distortion) term.",
        "Charge transfer carries the basis-set superposition error; polarization"
        " does not.",
        "Dispersion is defined against a dispersion-free partner functional (HF)."
        " With a -D3/-D4 functional it also contains the empirical dispersion,"
        " including the D4 three-body (ATM) term, which then appears in the"
        " ≥3-body dispersion.",
        "Classical electrostatic is the interaction of the frozen fragment charge"
        " densities; it is pairwise additive. 'induction' = polarization + charge"
        " transfer, comparable to SAPT induction.",
    ),
)

SCHEMES = {s.name: s for s in (ALMO_EDA2,)}
