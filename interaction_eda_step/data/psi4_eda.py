#!/usr/bin/env python
"""Second-generation ALMO-EDA with Psi4, for SEAMM's Interaction EDA step.

Runs in the Psi4 environment, which needs nothing from SEAMM:

    python psi4_eda.py input.json [-o results.jsonl] [--logdir logs]

``input.json`` is the same as for ``gpu4pyscf_eda.py``::

    {"settings": {"xc": "wb97x-v", "basis": "def2-tzvpd", "auxbasis": null,
                  "nlc": null, "disp": null, "svd_threshold": 1e-4,
                  "ferf": "coupled", "response_nlc": false, "threads": 8,
                  "memory_gb": 8},
     "calculations": [{"id": "0+1", "fragments": [{"label": "0:Li+",
                       "symbols": ["Li"], "coordinates": [[0.0, 0.0, 0.0]],
                       "charge": 1, "spin": 0}, ...]}, ...]}

The method is GPU4PySCF 1.9.1's ``gpu4pyscf/properties/eda.py`` (the reference),
step for step, with Psi4 doing the integrals, the Coulomb and exchange builds and the
exchange-correlation potential and kernel, and NumPy the rest:

1. an SCF of each fragment in its own basis;
2. the frozen state: the fragments' occupied orbitals, block-diagonal in the
   full basis and orthonormalized, and its energy;
3. the orthogonal decomposition: the frozen orbitals rotated among the occupied
   orbitals of different fragments to minimize the sum of the fragment energies;
   the electrostatic energy of the resulting fragment densities, the kinetic-energy
   pressure, and the dispersion energy as the change of the exchange-correlation
   energy on forming the frozen state, less the same for Hartree-Fock exchange;
4. the classical electrostatic energy of the isolated fragments' densities;
5. each fragment's FERF space: its occupied orbitals and the virtual responses to
   uniform fields and field gradients (dipole + quadrupole), truncated by singular
   value;
6. SCF-MI (Stoll's equations) of the fragments within their FERF spaces: the
   polarization energy;
7. the full SCF: the charge-transfer energy.

Psi4 1.11 has the exchange-correlation response kernel (``compute_Vx``) only for
LDA and GGA functionals without VV10. So the coupled FERF response needs a
non-meta functional, with VV10 left out of the kernel (``response_nlc: false``);
the SCF, SCF-MI and frozen energies always keep the full functional. Meta-GGAs
(e.g. wB97M-V) need ``ferf: uncoupled``, which moves energy from polarization to
charge transfer (+1.4 kJ/mol for Li+-EC at equilibrium) and is not comparable to
coupled results. Closed-shell fragments only; ``nlc: false`` and ``disp`` are not
supported yet.

Each calculation appends one JSON line to the results file, in the same form as
``gpu4pyscf_eda.py``: the terms in kJ/mol, the energies in hartree, the wall time
and the versions -- or the error. Calculations already done are skipped.
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

# CODATA 2018, as PySCF uses: 1e-3 * HARTREE2J * AVOGADRO
HARTREE_TO_KJMOL = 4.3597447222071e-18 * 6.02214076e23 * 1e-3

#: The Psi4 settings that match GPU4PySCF's eval_ALMO_EDA_2_energies defaults.
GRIDS = {
    "dft_radial_points": 99,
    "dft_spherical_points": 590,
    "dft_vv10_radial_points": 50,
    "dft_vv10_spherical_points": 194,
}
CONV_TOL = 1.0e-10  # energy, for every SCF here (GPU4PySCF conv_tol)
CONV_TOL_CPSCF = 1.0e-8  # residual of the response equations
MAX_CYCLE = 100
psi4 = None  # imported in main(), so the helpers can be tested without Psi4


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def inv_sqrt(M):
    w, U = np.linalg.eigh(M)
    if w.min() <= 1.0e-12:
        raise RuntimeError(
            f"Matrix not positive definite (smallest eigenvalue {w.min()})"
        )
    return (U * w**-0.5) @ U.T


def generalized_eigh(F, S):
    """Eigenvalues and vectors of F C = S C e, ascending."""
    X = inv_sqrt(S)
    e, U = np.linalg.eigh(X @ F @ X)
    return e, X @ U


def pack_tril(M):
    return M[np.tril_indices(M.shape[0])]


def unpack_tril(v, n):
    M = np.zeros((n, n))
    M[np.tril_indices(n)] = v
    return M + np.tril(M, -1).T


class DIIS:
    """Pulay DIIS on vectors with their error vectors (as pyscf.lib.diis.DIIS)."""

    def __init__(self, space=8):
        self.space = space
        self.x = []
        self.e = []

    def update(self, x, xerr):
        self.x.append(x.copy())
        self.e.append(xerr.copy())
        if len(self.x) > self.space:
            self.x.pop(0)
            self.e.pop(0)
        n = len(self.x)
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i + 1):
                B[i, j] = B[j, i] = np.dot(self.e[i], self.e[j])
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
        return sum(ci * xi for ci, xi in zip(c, self.x))


def conjugate_gradient(matvec, b, x0, precond, tol, max_iter=500):
    """Solve A x = b for symmetric positive definite A (one right-hand side)."""
    x = x0.copy()
    r = b - matvec(x)
    z = precond(r)
    p = z.copy()
    rz = np.dot(r, z)
    for _ in range(max_iter):
        if np.linalg.norm(r) < tol:
            return x
        Ap = matvec(p)
        a = rz / np.dot(p, Ap)
        x += a * p
        r -= a * Ap
        z = precond(r)
        rz_new = np.dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if np.linalg.norm(r) < tol:
        return x
    raise RuntimeError(
        f"Conjugate gradient not converged (|r| = {np.linalg.norm(r):.1e})"
    )


def block_conjugate_gradient(matvec, B, precond, tol, max_iter=200):
    """Solve A X = B column by column, with one matvec on all active columns."""
    X = np.zeros_like(B)
    R = B - matvec(X) if np.any(X) else B.copy()
    Z = precond(R)
    P = Z.copy()
    rz = np.einsum("ij,ij->j", R, Z)
    active = np.linalg.norm(R, axis=0) >= tol
    for _ in range(max_iter):
        if not active.any():
            return X
        AP = np.zeros_like(P)
        AP[:, active] = matvec(P[:, active])
        pap = np.einsum("ij,ij->j", P, AP)
        a = np.where(active, rz / np.where(active, pap, 1.0), 0.0)
        X += P * a
        R -= AP * a
        Z = precond(R)
        rz_new = np.einsum("ij,ij->j", R, Z)
        beta = np.where(active, rz_new / np.where(active, rz, 1.0), 0.0)
        P = Z + P * beta
        rz = rz_new
        active = np.linalg.norm(R, axis=0) >= tol
    if active.any():
        raise RuntimeError(
            f"Response equations not converged (max |r| = "
            f"{np.linalg.norm(R, axis=0).max():.1e})"
        )
    return X


# ---------------------------------------------------------------------------
# Psi4 pieces
# ---------------------------------------------------------------------------
def jk_algorithm(fragments, settings):
    """PK (in core) when the integrals fit comfortably in memory, else DIRECT.

    Both are exact. PK keeps the integrals in memory, so the many Fock builds of
    the EDA are fast; it must never go out of core, because its disk files
    collide with the SCFs' (psi4.core.clean() removes them), so anything that
    might not fit is integral-direct. With an auxiliary basis: MEM_DF.
    """
    if settings.get("auxbasis"):
        return "MEM_DF"
    basis = psi4.core.BasisSet.build(
        molecule(fragments), "ORBITAL", settings.get("basis", "def2-tzvpd")
    )
    n = basis.nbf()
    pk_bytes = 8.0 * (n * (n + 1) / 2) ** 2 / 2  # one triangle of pairs
    # The full system's engine and Psi4's own SCF each hold one copy
    return "PK" if 3 * pk_bytes < psi4.get_memory() else "DIRECT"


def method_name(xc):
    return "hf" if xc is None or xc.lower() in ("hf", "scf") else xc.lower()


def build_superfunctional(xc, with_vv10=True):
    sup, _disp = psi4.driver.dft.build_superfunctional(method_name(xc), True)
    if not with_vv10 and sup.needs_vv10():
        sup.set_do_vv10(False)
    return sup


def molecule(fragments, which=None):
    """A Psi4 molecule of the fragments (indices ``which``, default all).

    Never moved or reoriented and without symmetry, so the AO order and the
    coordinates are those of the input, fragment after fragment.
    """
    lines = []
    for i, frag in enumerate(fragments):
        if which is not None and i not in which:
            continue
        lines.append("--")
        lines.append(f"{int(frag.get('charge', 0))} {int(frag.get('spin', 0)) + 1}")
        for s, xyz in zip(frag["symbols"], frag["coordinates"]):
            lines.append("{} {:.10f} {:.10f} {:.10f}".format(s, *xyz))
    lines = lines[1:]
    lines += ["units angstrom", "no_com", "no_reorient", "symmetry c1"]
    mol = psi4.core.Molecule.from_string("\n".join(lines))
    mol.update_geometry()
    return mol


class Engine:
    """Energy and Fock matrix of a density in one basis with one functional.

    ``compute(Ca)`` takes orthonormal occupied (alpha) orbitals; the density is
    D = 2 Ca Ca^T. Exact for HF, hybrids, range-separated hybrids and VV10: it
    reproduces Psi4's own SCF energy at a converged density to 1e-13 hartree.
    """

    def __init__(self, basis, sup, jk_type):
        self.basis = basis
        self.sup = sup
        self.nbf = basis.nbf()
        self.jk = psi4.core.JK.build(basis, jk_type=jk_type)
        self.jk.set_memory(int(psi4.get_memory() * 0.5 / 8))
        self.lrc = sup.is_x_lrc()
        if self.lrc:
            self.jk.set_do_wK(True)
            self.jk.set_omega(sup.x_omega())
        self.jk.initialize()
        self.alpha = sup.x_alpha() if sup.is_x_hybrid() else 0.0
        self.beta = sup.x_beta() if self.lrc else 0.0
        self.V = None
        if sup.needs_xc():
            self.V = psi4.core.VBase.build(basis, sup, "RV")
            self.V.initialize()

    def jk_build(self, left, right=None):
        """J, K (and wK) for the densities left[k] right[k]^T."""
        self.jk.C_clear()
        for k, L in enumerate(left):
            self.jk.C_left_add(psi4.core.Matrix.from_array(np.ascontiguousarray(L)))
            if right is not None:
                self.jk.C_right_add(
                    psi4.core.Matrix.from_array(np.ascontiguousarray(right[k]))
                )
        self.jk.compute()
        J = [np.asarray(m).copy() for m in self.jk.J()]
        K = [np.asarray(m).copy() for m in self.jk.K()]
        wK = [np.asarray(m).copy() for m in self.jk.wK()] if self.lrc else None
        return J, K, wK

    def two_electron(self, Ca):
        """(G, E_J, E_X, E_XC) for D = 2 Ca Ca^T; G is the alpha Fock part."""
        J, K, wK = self.jk_build([Ca])
        J, K = J[0], K[0]
        Da = Ca @ Ca.T
        G = 2 * J
        E_J = 2 * np.vdot(Da, J)
        E_X = 0.0
        if self.alpha:
            G -= self.alpha * K
            E_X -= self.alpha * np.vdot(Da, K)
        if self.lrc:
            G -= self.beta * wK[0]
            E_X -= self.beta * np.vdot(Da, wK[0])
        E_XC = 0.0
        if self.V is not None:
            Vm = psi4.core.Matrix(self.nbf, self.nbf)
            self.V.set_D([psi4.core.Matrix.from_array(Da)])
            self.V.compute_V([Vm])
            q = self.V.quadrature_values()
            E_XC = q["FUNCTIONAL"] + q.get("VV10", 0.0)
            G += np.asarray(Vm)
        return G, E_J, E_X, E_XC, K

    def energy_and_fock(self, Ca, H, e_nuc):
        G, E_J, E_X, E_XC, _ = self.two_electron(Ca)
        Da = Ca @ Ca.T
        return 2 * np.vdot(Da, H) + E_J + E_X + E_XC + e_nuc, H + G

    def xc_energies(self, Ca):
        """(exchange + XC energy of the functional, HF exchange energy) of 2 Ca Ca^T."""
        _, _, E_X, E_XC, K = self.two_electron(Ca)
        return E_X + E_XC, -np.vdot(Ca @ Ca.T, K)

    def response(self, Cv, Co, X, Vx):
        """C_v^T v[dm1] C_o for amplitudes X (nfield, nvir, nocc).

        dm1 = 2 (C_v x C_o^T + C_o x^T C_v^T), the change of the total density;
        v = J - (alpha K + beta wK) / 2 + f_xc, as PySCF's gen_response.
        ``Vx`` is a VBase for the kernel (None for HF), set to the fragment density.
        """
        left = [Cv @ x for x in X]
        J, K, wK = self.jk_build(left, [Co] * len(left))
        fxc = [None] * len(X)
        if Vx is not None:
            # One pass over the grid for all the fields
            dm1s, rets = [], []
            for L in left:
                D = L @ Co.T
                dm1s.append(psi4.core.Matrix.from_array(2 * (D + D.T)))
                rets.append(psi4.core.Matrix(self.nbf, self.nbf))
            Vx.compute_Vx(dm1s, rets)
            fxc = [np.asarray(r) for r in rets]
        out = []
        for k in range(len(X)):
            v = 4 * J[k]
            if self.alpha:
                v -= 0.5 * self.alpha * 2 * (K[k] + K[k].T)
            if self.lrc:
                v -= 0.5 * self.beta * 2 * (wK[k] + wK[k].T)
            if fxc[k] is not None:
                v += fxc[k]
            out.append(Cv.T @ v @ Co)
        return np.array(out)


# ---------------------------------------------------------------------------
# The ALMO-EDA2 steps
# ---------------------------------------------------------------------------
class Fragment:
    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.label = data.get("label", str(index))
        if int(data.get("spin", 0)) != 0:
            raise NotImplementedError(
                f"Fragment {self.label} is open-shell; only closed shells are "
                "supported."
            )


def run_fragment(fragments, i, xc, log):
    """SCF of fragment i in its own basis. Returns (energy, wfn)."""
    mol = molecule(fragments, which=[i])
    psi4.core.clean()
    e, wfn = psi4.energy(method_name(xc), molecule=mol, return_wfn=True)
    log(
        f"Fragment {i} energy = {e:.10f} hartree, {wfn.basisset().nbf()} basis "
        f"functions, {wfn.nmo()} orbitals, {wfn.nalpha()} occupied"
    )
    return e, wfn


def ferf_space(wfn, xc, settings, jk_type, log):
    """G for one fragment: its occupied orbitals and the FERF virtual responses."""
    basis = wfn.basisset()
    C = np.asarray(wfn.Ca()).copy()
    eps = np.asarray(wfn.epsilon_a()).copy()
    nocc = wfn.nalpha()
    Co, Cv = C[:, :nocc], C[:, nocc:]
    mol = wfn.molecule()
    # Centre of mass with mass numbers in single precision, as PySCF's
    # atom_mass_list() and eda.py do.
    mass = np.array([round(mol.mass(a)) for a in range(mol.natom())], dtype=np.float32)
    xyz = np.asarray(mol.geometry())  # bohr
    com = (mass @ xyz) / mass.sum()
    mints = psi4.core.MintsHelper(basis)
    m = [-np.asarray(x) for x in mints.ao_multipoles(2, list(map(float, com)))]
    r = m[:3]  # x, y, z
    xx, xy, xz, yy, yz, zz = m[3:9]
    tr = xx + yy + zz
    xx, yy, zz = 1.5 * (xx - tr / 3), 1.5 * (yy - tr / 3), 1.5 * (zz - tr / 3)
    xy, xz, yz = 1.5 * xy, 1.5 * xz, 1.5 * yz
    quad = [
        (2.0 / np.sqrt(3.0)) * xy,
        (2.0 / np.sqrt(3.0)) * yz,
        zz,
        (2.0 / np.sqrt(3.0)) * xz,
        (1.0 / np.sqrt(3.0)) * (xx - yy),
    ]
    h = np.array([-2 * Cv.T @ M @ Co for M in r + quad])  # (8, nvir, nocc)
    e_ai = eps[nocc:, None] - eps[None, :nocc]
    if settings.get("ferf", "coupled") == "uncoupled":
        kappa = -h / e_ai
    else:
        sup = build_superfunctional(xc, with_vv10=False)
        if sup.is_meta():
            raise RuntimeError(
                f"Psi4 has no exchange-correlation response kernel for meta-GGA "
                f"functionals such as {xc}, so the coupled FERF response cannot be "
                "computed; use the uncoupled response."
            )
        full_sup = build_superfunctional(xc)
        if full_sup.needs_vv10() and settings.get("response_nlc", True):
            raise RuntimeError(
                "Psi4 has no VV10 response kernel. Set 'VV10 in the FERF response' "
                "to off (the SCF and SCF-MI keep VV10; the polarization/charge "
                "transfer split moves by about 0.001 kJ/mol)."
            )
        eng = Engine(basis, sup, jk_type)
        Vx = None
        if sup.needs_xc():
            Vx = psi4.core.VBase.build(basis, sup, "RV")
            Vx.initialize()
            Vx.set_D([psi4.core.Matrix.from_array(Co @ Co.T)])
        nf, nv, no = h.shape

        def matvec(Xcols):
            X = Xcols.T.reshape(-1, nv, no)
            AX = e_ai * X + eng.response(Cv, Co, X, Vx)
            return AX.reshape(len(X), -1).T

        def precond(R):
            return R / e_ai.reshape(-1, 1)

        B = (-h).reshape(nf, -1).T
        kappa = block_conjugate_gradient(matvec, B, precond, CONV_TOL_CPSCF)
        kappa = kappa.T.reshape(nf, nv, no)
    threshold = float(settings.get("svd_threshold") or 1.0e-4)
    space = [Co]
    for k in kappa:
        U, s, _ = np.linalg.svd(k, full_matrices=False)
        space.append(Cv @ U[:, s > threshold])
    U, s, _ = np.linalg.svd(np.hstack(space), full_matrices=False)
    G = U[:, s > threshold]
    log(f"FERF space: {G.shape[1]} of {sum(x.shape[1] for x in space)} vectors kept")
    return G


def orthogonal_decomposition(eng, S, H_frag, e_nuc_frag, Cocc, nocc, log):
    """Rotate the orthonormal frozen orbitals among fragments (occ-occ) to minimize
    the sum of the fragment energies. Returns (orbitals, E_initial, E_final)."""
    n = len(nocc)
    off = np.cumsum([0] + list(nocc))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sizes = [nocc[i] * nocc[j] for i, j in pairs]
    poff = np.cumsum([0] + sizes)

    def fragment_fock_energy(C):
        Fs, Es = [], []
        for i in range(n):
            Ci = C[:, off[i] : off[i + 1]]
            E, F = eng.energy_and_fock(Ci, H_frag[i], e_nuc_frag[i])
            Fs.append(F)
            Es.append(E)
        return Fs, Es

    C = Cocc
    Fs, Es = fragment_fock_energy(C)
    E = E0 = sum(Es)
    log(f"Orthogonal decomposition: cycle 0 energy {E:.12f}")
    for cycle in range(MAX_CYCLE):
        blocks = [C[:, off[i] : off[i + 1]] for i in range(n)]
        g = np.zeros(poff[-1])
        for p, (i, j) in enumerate(pairs):
            g[poff[p] : poff[p + 1]] = (
                2 * blocks[i].T @ (Fs[j] - Fs[i]) @ blocks[j]
            ).ravel()

        def hess(x):
            y = np.zeros_like(x)
            for p, (i, j) in enumerate(pairs):
                xij = x[poff[p] : poff[p + 1]].reshape(nocc[i], nocc[j])
                for q, (k, l) in enumerate(pairs):
                    ykl = np.zeros((nocc[k], nocc[l]))
                    Ci, Cj, Ck, Cl = blocks[i], blocks[j], blocks[k], blocks[l]
                    Fi, Fj, Fk, Fl = Fs[i], Fs[j], Fs[k], Fs[l]
                    if k == j:
                        ykl += ((Ci.T @ (Fl - 2 * Fk + Fi) @ Cl).T @ xij).T
                    if l == j:
                        ykl += (-Ci.T @ (Fk - 2 * Fl + Fi) @ Ck).T @ xij
                    if l == i:
                        ykl += (xij @ (Cj.T @ (Fk - 2 * Fl + Fj) @ Ck)).T
                    if k == i:
                        ykl += xij @ (-Cj.T @ (Fl - 2 * Fk + Fj) @ Cl)
                    y[poff[q] : poff[q + 1]] += ykl.ravel()
            return y

        if poff[-1] < 2:
            Hm = np.array([hess(np.eye(poff[-1])[k]) for k in range(poff[-1])]).T
            step = -np.linalg.solve(Hm, g)
        else:
            # The preconditioner of eda.py: the diagonal blocks, diagonalized.
            x0 = np.zeros_like(g)
            precs = []
            for p, (i, j) in enumerate(pairs):
                Pi = 2 * blocks[i].T @ (Fs[j] - Fs[i]) @ blocks[i]
                Pj = 2 * blocks[j].T @ (Fs[i] - Fs[j]) @ blocks[j]
                wi, Ui = np.linalg.eigh(Pi)
                wj, Uj = np.linalg.eigh(Pj)
                d = wi[:, None] + wj[None, :]
                dinv = np.where(np.abs(d) < 1e-14, 0.0, 1.0 / np.where(d == 0, 1, d))
                precs.append((Ui, Uj, dinv))
                gij = g[poff[p] : poff[p + 1]].reshape(nocc[i], nocc[j])
                x0[poff[p] : poff[p + 1]] = (
                    Ui @ (dinv * (Ui.T @ gij @ Uj)) @ Uj.T
                ).ravel()

            def precond(r):
                z = np.zeros_like(r)
                for p, (i, j) in enumerate(pairs):
                    Ui, Uj, dinv = precs[p]
                    rij = r[poff[p] : poff[p + 1]].reshape(nocc[i], nocc[j])
                    z[poff[p] : poff[p + 1]] = (
                        Ui @ (dinv * (Ui.T @ rij @ Uj)) @ Uj.T
                    ).ravel()
                return z

            # eda.py uses plain CG (cupyx cg, x0 from the preconditioner, atol 1e-14)
            step = -conjugate_gradient(hess, g, x0, lambda r: r, 1e-14, max_iter=2000)
        A = np.zeros((C.shape[1], C.shape[1]))
        for p, (i, j) in enumerate(pairs):
            sij = step[poff[p] : poff[p + 1]].reshape(nocc[i], nocc[j])
            A[off[i] : off[i + 1], off[j] : off[j + 1]] = sij
            A[off[j] : off[j + 1], off[i] : off[i + 1]] = -sij.T
        C = C @ expm_antisymmetric(A)
        Fs, Es = fragment_fock_energy(C)
        E_prev, E = E, sum(Es)
        log(
            f"Orthogonal decomposition: cycle {cycle + 1} energy {E:.12f} "
            f"change {E - E_prev:+.2e}"
        )
        if abs(E - E_prev) < CONV_TOL:
            return C, E0, E
    raise RuntimeError("Orthogonal decomposition not converged")


def expm_antisymmetric(A):
    """exp(A) for a real antisymmetric A, via the eigenvectors of the Hermitian iA."""
    w, U = np.linalg.eigh(1j * A)
    return (U @ np.diag(np.exp(-1j * w)) @ U.conj().T).real


def scf_mi(eng, S, H, e_nuc, G_list, Cocc_list, log):
    """Stoll SCF-MI of the fragments within their FERF spaces (eda.py's algorithm).
    Returns (frozen energy, polarized energy, polarized occupied orbitals)."""
    n = len(G_list)
    nao_off = np.cumsum([0] + [g.shape[0] for g in G_list])
    np_off = np.cumsum([0] + [g.shape[1] for g in G_list])
    no_off = np.cumsum([0] + [c.shape[1] for c in Cocc_list])
    nao, nproj, nocc = nao_off[-1], np_off[-1], no_off[-1]
    G = np.zeros((nao, nproj))
    Cocc = np.zeros((nao, nocc))
    for i in range(n):
        G[nao_off[i] : nao_off[i + 1], np_off[i] : np_off[i + 1]] = G_list[i]
        Cocc[nao_off[i] : nao_off[i + 1], no_off[i] : no_off[i + 1]] = Cocc_list[i]
    gamma = G.T @ S @ G
    m = G.T @ Cocc  # G is orthonormal (Euclidean), and the occupied orbitals lie in it

    def density_energy_fock(m):
        Cao = G @ m
        sigma = m.T @ gamma @ m
        inv_sigma = np.linalg.inv(sigma)
        D = 2 * Cao @ inv_sigma @ Cao.T
        Corth = Cao @ inv_sqrt(sigma)
        E, F = eng.energy_and_fock(Corth, H, e_nuc)
        return D, inv_sigma, E, F

    D, inv_sigma, E, F = density_energy_fock(m)
    E_frozen = E
    log(f"SCF-MI: cycle 0 energy {E:.12f}")
    diis = [DIIS() for _ in range(n)]
    I2 = 2 * np.eye(nao)
    for cycle in range(MAX_CYCLE):
        FS = []
        for i in range(n):
            ci = slice(no_off[i], no_off[i + 1])
            pi = slice(np_off[i], np_off[i + 1])
            D_stoll = 2 * G @ m @ inv_sigma[:, ci] @ m[pi, ci].T @ G[:, pi].T
            F_stoll = (
                G.T @ (I2 - S @ D + S @ D_stoll.T) @ F @ (I2 - D @ S + D_stoll @ S) @ G
            )
            FS.append((F_stoll[pi, pi], gamma[pi, pi]))
        Err = np.linalg.solve(gamma, G.T @ S @ D @ F @ (D @ S - I2) @ G) / 4
        new_m = np.zeros_like(m)
        for i in range(n):
            ci = slice(no_off[i], no_off[i + 1])
            pi = slice(np_off[i], np_off[i + 1])
            g_ii = gamma[pi, pi]
            e_ii = Err[pi, pi]
            e_ii = g_ii @ e_ii - e_ii.T @ g_ii.T
            F_ii, S_ii = FS[i]
            k = F_ii.shape[0]
            F_new = unpack_tril(diis[i].update(pack_tril(F_ii), pack_tril(e_ii)), k)
            _, Cii = generalized_eigh(F_new, S_ii)
            new_m[pi, ci] = Cii[:, : no_off[i + 1] - no_off[i]]
        m = new_m
        E_prev = E
        D, inv_sigma, E, F = density_energy_fock(m)
        log(f"SCF-MI: cycle {cycle + 1} energy {E:.12f} change {E - E_prev:+.2e}")
        if abs(E - E_prev) < CONV_TOL:
            return E_frozen, E, G @ m
    raise RuntimeError("FERF subspace SCF-MI not converged")


def eda(fragments, settings, log):
    """ALMO-EDA2 of the fragments. Returns (terms in kJ/mol, energies in hartree)."""
    xc = settings.get("xc", "wb97x-v")
    if settings.get("nlc") is False:
        raise NotImplementedError("nlc: false (VV10 off in the whole functional)")
    if settings.get("disp"):
        raise NotImplementedError("dispersion corrections (disp)")
    jk_type = jk_algorithm(fragments, settings)
    n = len(fragments)
    if n < 2:
        raise ValueError("ALMO-EDA needs at least two fragments")

    # 1. Fragment SCFs and their FERF spaces
    frag_E, frag_wfn, G_list = [], [], []
    for i in range(n):
        e, wfn = run_fragment(fragments, i, xc, log)
        frag_E.append(e)
        frag_wfn.append(wfn)
        G_list.append(ferf_space(wfn, xc, settings, jk_type, log))
    Cocc_list = [np.asarray(w.Ca_subset("AO", "OCC")).copy() for w in frag_wfn]
    nocc = [c.shape[1] for c in Cocc_list]

    # The full system, its basis, and each fragment's nuclei (ghosts elsewhere)
    full = molecule(fragments)
    basis = psi4.core.BasisSet.build(
        full, "ORBITAL", settings.get("basis", "def2-tzvpd")
    )
    nbf = basis.nbf()
    if nbf != sum(w.basisset().nbf() for w in frag_wfn):
        raise RuntimeError("The full basis is not the sum of the fragment bases")
    mints = psi4.core.MintsHelper(basis)
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    e_nuc = full.nuclear_repulsion_energy()
    V_frag, e_nuc_frag = [], []
    for i in range(n):
        # Fragment i real, the others ghosts: the same basis functions in the same
        # order, and only fragment i's nuclei
        ghost = full.extract_subsets([i + 1], [j + 1 for j in range(n) if j != i])
        ghost.update_geometry()
        b = psi4.core.BasisSet.build(
            ghost, "ORBITAL", settings.get("basis", "def2-tzvpd")
        )
        V_frag.append(np.asarray(psi4.core.MintsHelper(b).ao_potential()))
        e_nuc_frag.append(frag_wfn[i].molecule().nuclear_repulsion_energy())
    H_frag = [T + Vi for Vi in V_frag]
    eng = Engine(basis, build_superfunctional(xc), jk_type)

    off = np.cumsum([0] + [w.basisset().nbf() for w in frag_wfn])
    ooff = np.cumsum([0] + nocc)
    Cfrz = np.zeros((nbf, sum(nocc)))
    for i in range(n):
        Cfrz[off[i] : off[i + 1], ooff[i] : ooff[i + 1]] = Cocc_list[i]

    def coulomb_terms(blocks):
        """Sum over pairs of the electrostatic interaction of fragment densities
        2 C_i C_i^T (in the full basis) with each other's nuclei and electrons."""
        J, _, _ = eng.jk_build(blocks)
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                Di = 2 * blocks[i] @ blocks[i].T
                Dj = 2 * blocks[j] @ blocks[j].T
                E_ee = np.vdot(Dj, 2 * J[i])
                E_en = np.vdot(Di, V_frag[j]) + np.vdot(Dj, V_frag[i])
                both = molecule(fragments, which=[i, j])
                E_nn = both.nuclear_repulsion_energy() - e_nuc_frag[i] - e_nuc_frag[j]
                total += E_ee + E_en + E_nn
        return total

    # 2. Classical electrostatics: the isolated fragments' densities
    blocks = [Cfrz[:, ooff[i] : ooff[i + 1]] for i in range(n)]
    e_cls = coulomb_terms(blocks)

    # 3. Frozen orbitals, orthonormalized; the orthogonal decomposition
    Corth = Cfrz @ inv_sqrt(Cfrz.T @ S @ Cfrz)
    C_pauli, E_unrelaxed, E_relaxed = orthogonal_decomposition(
        eng, S, H_frag, e_nuc_frag, Corth, nocc, log
    )
    blocks = [C_pauli[:, ooff[i] : ooff[i + 1]] for i in range(n)]
    e_elst = coulomb_terms(blocks)
    ke_pressure = E_relaxed - E_unrelaxed

    # Dispersion: change of the XC (exchange + correlation) energy on forming the
    # frozen state, less the same for Hartree-Fock exchange
    xc_frz, hf_frz = eng.xc_energies(C_pauli)
    xc_frag = hf_frag = 0.0
    for b in blocks:
        a, h_ = eng.xc_energies(b)
        xc_frag += a
        hf_frag += h_
    if method_name(xc) == "hf":
        e_disp = 0.0
    else:
        e_disp = (xc_frz - xc_frag) - (hf_frz - hf_frag)
    interfragment_hfx = hf_frz - hf_frag

    # 4. SCF-MI in the FERF spaces: frozen and polarized energies
    E_frozen, E_pol, C_pol = scf_mi(eng, S, H, e_nuc, G_list, Cocc_list, log)

    # 5. The full SCF
    psi4.core.clean()
    E_full = psi4.energy(method_name(xc), molecule=full)

    E_sum = sum(frag_E)
    frozen = E_frozen - E_sum
    terms = {
        "total": E_full - E_sum,
        "frozen": frozen,
        "electrostatic": e_elst,
        "classical electrostatic": e_cls,
        "dispersion": e_disp,
        "pauli": frozen - e_elst - e_disp,
        "polarization": E_pol - E_frozen,
        "charge transfer": E_full - E_pol,
    }
    log(
        f"Kinetic energy pressure {ke_pressure:.10f}, interfragment HF exchange "
        f"{interfragment_hfx:.10f}; Pauli (orthogonal decomposition) "
        f"{ke_pressure + interfragment_hfx:.10f} hartree"
    )
    return (
        {k: v * HARTREE_TO_KJMOL for k, v in terms.items()},
        frag_E + [E_full],
        int(nbf),
    )


# ---------------------------------------------------------------------------
def done_ids(path):
    """The ids already in the results file with a result."""
    ids = set()
    if os.path.exists(path):
        with open(path) as fd:
            for line in fd:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "eda_kJ_mol" in r:
                    ids.add(r["id"])
    return ids


def main(argv=None):
    global psi4
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input")
    p.add_argument("-o", "--output", default="results.jsonl")
    p.add_argument("--logdir", default="logs")
    p.add_argument("--threads", type=int, default=None, help="default: settings or 1")
    args = p.parse_args(argv)

    with open(args.input) as fd:
        data = json.load(fd)
    settings = data.get("settings", {})
    calculations = data["calculations"]

    import psi4 as _psi4

    psi4 = _psi4
    threads = int(
        args.threads or settings.get("threads") or os.environ.get("OMP_NUM_THREADS", 1)
    )
    psi4.core.set_num_threads(threads)
    psi4.set_memory(f"{float(settings.get('memory_gb') or 4)} GB")
    options = {
        "basis": settings.get("basis", "def2-tzvpd"),
        # Set per calculation to match the EDA's JK algorithm (jk_algorithm)
        "scf_type": "direct" if not settings.get("auxbasis") else "mem_df",
        "e_convergence": CONV_TOL,
        "d_convergence": 1.0e-8,
        "maxiter": MAX_CYCLE,
        **GRIDS,
    }
    if settings.get("auxbasis"):
        options["df_basis_scf"] = settings["auxbasis"]
    info = {"psi4": psi4.__version__, "numpy": np.__version__, "threads": threads}
    print(f"Psi4 {psi4.__version__}, {threads} threads", flush=True)
    os.makedirs(args.logdir, exist_ok=True)
    skip = done_ids(args.output)

    n_failed = 0
    with open(args.output, "a") as out:
        for calc in calculations:
            cid = calc["id"]
            if cid in skip:
                print(f"{cid}: already done", flush=True)
                continue
            log_path = os.path.join(args.logdir, f"{cid}.log")
            # *.log, so the step keeps it with the EDA log
            psi4.core.set_output_file(
                os.path.join(args.logdir, f"{cid}.psi4.log"), False
            )
            psi4.set_options(options)
            record = {"id": cid, "code": "psi4", "versions": info}
            t0 = time.time()
            with open(log_path, "w") as logfd:

                def log(text):
                    logfd.write(text + "\n")
                    logfd.flush()

                try:
                    fragments = calc["fragments"]
                    for i, f in enumerate(fragments):
                        Fragment(f, i)
                    jk = jk_algorithm(fragments, settings)
                    psi4.set_options({"scf_type": jk.lower()})
                    log(f"Integrals: {jk}")
                    terms, energies, nbf = eda(fragments, settings, log)
                    record["eda_kJ_mol"] = {k: float(v) for k, v in terms.items()}
                    record["energies_hartree"] = [float(e) for e in energies]
                    record["nao"] = nbf
                    for k, v in terms.items():
                        log(f"EDA {k} = {v:.10f} kJ/mol")
                except Exception:
                    n_failed += 1
                    record["error"] = traceback.format_exc()
                    log(record["error"])
                    print(f"{cid}: FAILED\n{record['error']}", flush=True)
            record["wall_s"] = round(time.time() - t0, 2)
            out.write(json.dumps(record) + "\n")
            out.flush()
            if "eda_kJ_mol" in record:
                print(
                    f"{cid}: total {record['eda_kJ_mol']['total']:.3f} kJ/mol "
                    f"({record['wall_s']} s)",
                    flush=True,
                )
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
