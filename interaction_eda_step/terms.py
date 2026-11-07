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
* SAPT's electrostatics is the Coulomb interaction of the isolated monomers'
  densities, which is ALMO's *classical* electrostatic term (Q-Chem's CLS ELEC).
  ALMO's own "electrostatic" and "pauli" come from an orthogonal decomposition of
  the frozen wavefunction (Horn, Mao and Head-Gordon 2016; Q-Chem's ELEC and
  PAULI) and are not comparable with SAPT: for PF6⁻–EC each differs from SAPT by
  36 kJ/mol. So the common "electrostatics" is the classical term and the common
  "Pauli" is frozen − classical electrostatic − dispersion (Q-Chem's MOD PAULI),
  within 2 kJ/mol of SAPT exchange. frozen = electrostatics + Pauli + dispersion
  holds in both sets, and the common ≥3-body electrostatics is exactly 0.

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
        Each common term as the sum of native terms; a name starting with "-"
        is subtracted.
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
            signed = [(-1, p[1:]) if p.startswith("-") else (1, p) for p in parts]
            if all(p in energies for _, p in signed):
                out[name] = sum(sign * energies[p] for sign, p in signed)
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
        "electrostatics": ("classical electrostatic",),
        "Pauli": ("frozen", "-classical electrostatic", "-dispersion"),
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
        "Two electrostatic terms: 'classical electrostatic' (Q-Chem CLS ELEC) is the"
        " Coulomb interaction of the isolated fragments' densities, comparable to"
        " SAPT electrostatics and pairwise additive; 'electrostatic' and 'pauli'"
        " (Q-Chem ELEC, PAULI) come from the orthogonal decomposition of the frozen"
        " wavefunction (Horn, Mao and Head-Gordon 2016) and are NOT comparable to"
        " SAPT.",
        "The common terms use the SAPT-comparable pair: electrostatics = classical"
        " electrostatic; Pauli = frozen - classical electrostatic - dispersion"
        " (Q-Chem MOD PAULI). frozen = electrostatics + Pauli + dispersion holds"
        " in both sets; the common >=3-body electrostatics is exactly 0.",
        "'induction' = polarization + charge transfer, comparable to SAPT"
        " induction.",
    ),
)

SCHEMES = {s.name: s for s in (ALMO_EDA2,)}
