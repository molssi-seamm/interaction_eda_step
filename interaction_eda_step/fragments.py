# -*- coding: utf-8 -*-

"""The fragments of a structure for EDA.

By default each molecule of the configuration is a fragment. The molecules
come from the bonds, and their charges from the atoms' formal charges or, for
known molecules (water, carbonates, Li⁺, BF₄⁻, PF₆⁻, monatomic ions), from
:mod:`seamm_mbe`'s catalog, which also checks that every fragment has a
sensible number of electrons. Alternatively an integer atom column (e.g.
"fragment") assigns the atoms to fragments, with charges summed from the
formal charges.
"""

from dataclasses import dataclass
import itertools

import numpy as np


class FragmentError(ValueError):
    """The fragments cannot be made from the structure."""


@dataclass
class Fragment:
    """One fragment.

    Attributes
    ----------
    index : int
        0-based.
    name : str
        e.g. "Li+", "EC", or the formula.
    atoms : [int]
        0-based atom indices in the configuration, ascending.
    symbols : [str]
    coordinates : numpy.ndarray
        (n, 3) Cartesian, Å.
    charge : int
    multiplicity : int
    """

    index: int
    name: str
    atoms: list
    symbols: list
    coordinates: np.ndarray
    charge: int = 0
    multiplicity: int = 1

    @property
    def n_electrons(self):
        from seamm_mbe.elements import atomic_number

        return sum(atomic_number(s) for s in self.symbols) - self.charge

    def label(self):
        """A short label, e.g. '0:Li+'."""
        return f"{self.index}:{self.name}"


def from_configuration(configuration, column=None, charges=None):
    """The fragments of a configuration.

    Parameters
    ----------
    configuration : molsystem._Configuration
        A cluster (periodicity 0).
    column : str or None
        An integer atom column assigning the atoms to fragments; None to use
        the molecules.
    charges : {str: int} or None
        Charges by molecule type (molecules) or by fragment number as a string
        (column), overriding the formal charges and the catalog.

    Returns
    -------
    [Fragment]
    """
    if configuration.periodicity != 0:
        raise FragmentError(
            "Interaction EDA needs a cluster (a non-periodic structure); this "
            f"configuration has periodicity {configuration.periodicity}."
        )
    if configuration.n_atoms == 0:
        raise FragmentError("The configuration has no atoms.")
    if column:
        return _from_column(configuration, column, charges or {})
    return _from_molecules(configuration, charges or {})


def _from_molecules(configuration, charges):
    from seamm_mbe import System
    from seamm_mbe.system import StructureError

    try:
        system = System.from_configuration(configuration, charges=charges)
    except StructureError as e:
        raise FragmentError(str(e)) from e
    fragments = []
    for molecule in system.molecules:
        mtype = system.types[molecule.type]
        atoms = [int(a) for a in molecule.atoms]
        fragments.append(
            Fragment(
                index=molecule.index,
                name=mtype.name,
                atoms=atoms,
                symbols=[system.symbols[a] for a in atoms],
                coordinates=np.array(molecule.coordinates, dtype=float),
                charge=int(mtype.charge or 0),
                multiplicity=int(mtype.multiplicity or 1),
            )
        )
    _check(fragments)
    return fragments


def _from_column(configuration, column, charges):
    atoms = configuration.atoms
    if column not in atoms:
        raise FragmentError(
            f"The atoms have no column '{column}' to define the fragments."
        )
    ids = [int(x) for x in atoms.get_column_data(column)]
    symbols = atoms.symbols
    xyz = np.array(atoms.get_coordinates(fractionals=False), dtype=float)
    formal = (
        [int(q or 0) for q in atoms.get_column_data("formal_charge")]
        if "formal_charge" in atoms
        else [0] * len(symbols)
    )
    fragments = []
    for index, fid in enumerate(sorted(set(ids))):
        members = [k for k, x in enumerate(ids) if x == fid]
        charge = int(charges.get(str(fid), sum(formal[k] for k in members)))
        fragments.append(
            Fragment(
                index=index,
                name=_formula([symbols[k] for k in members]),
                atoms=members,
                symbols=[symbols[k] for k in members],
                coordinates=xyz[members],
                charge=charge,
            )
        )
    _check(fragments)
    return fragments


def _formula(symbols):
    from seamm_mbe.elements import hill_formula

    return hill_formula(symbols)


def _check(fragments):
    if len(fragments) < 2:
        raise FragmentError(
            f"The structure has {len(fragments)} fragment(s); an interaction "
            "needs at least two. Are the bonds right?"
        )
    for f in fragments:
        if f.multiplicity != 1 or f.n_electrons % 2 != 0:
            raise FragmentError(
                f"Fragment {f.label()} has {f.n_electrons} electrons and "
                f"multiplicity {f.multiplicity}; only closed-shell fragments "
                "are supported."
            )


def min_distance(a, b):
    """The shortest atom-atom distance between two fragments, Å."""
    d = a.coordinates[:, None, :] - b.coordinates[None, :, :]
    return float(np.sqrt((d * d).sum(axis=-1)).min())


def triples_within(fragments, cutoff):
    """The triples whose three pairs are all within ``cutoff`` (Å, shortest
    atom-atom distance); all triples when ``cutoff`` is None."""
    n = len(fragments)
    out = []
    for key in itertools.combinations(range(n), 3):
        if cutoff is None or all(
            min_distance(fragments[i], fragments[j]) <= cutoff
            for i, j in itertools.combinations(key, 2)
        ):
            out.append(key)
    return out
