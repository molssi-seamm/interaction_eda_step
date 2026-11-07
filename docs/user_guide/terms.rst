.. _eda-terms:

*************
The EDA terms
*************

Second-generation ALMO-EDA
==========================
ALMO-EDA (absolutely localized molecular orbital energy decomposition analysis) builds
the interaction up in steps, with Kohn--Sham DFT [Horn2016a]_ [Horn2016b]_:

frozen
    The fragments' own orbitals, unchanged, brought together and antisymmetrized. It is
    divided into three parts:

    electrostatic
        The interaction of the fragments' frozen charge densities, including their
        overlap (charge penetration).
    Pauli
        The repulsion from antisymmetrizing the fragments' occupied orbitals.
    dispersion
        Defined against a dispersion-free partner functional (Hartree--Fock exchange).
        With a functional that adds an empirical correction (-D3, -D4) it also contains
        that correction, including the D4 three-body (ATM) term.

polarization
    The relaxation of each fragment's orbitals in the presence of the others, kept on
    its own fragment. The space they relax in is spanned by fragment electric-field
    response functions (FERF-DQ: dipole and quadrupole fields) [Horn2015]_, which keeps
    polarization nearly free of basis-set superposition error.

charge transfer
    The rest of the relaxation to the full SCF of the complex, in which electrons may
    move between fragments.

total
    frozen + polarization + charge transfer. These identities hold exactly, and the
    step checks them.

classical electrostatic
    The electrostatic interaction of the fragments' frozen densities as point-free
    classical charge distributions. It is exactly pairwise additive, so its many-body
    part is zero, which is a useful check.

Conventions
-----------
* **The fragments are at their geometry in the cluster.** There is no preparation or
  distortion term: the energies are relative to the isolated fragments at the same
  geometry.
* **Charge transfer carries the basis-set superposition error; polarization does not.**
* Energies are in kJ/mol.

Common terms
============
To compare schemes, the step also gives a few *common* terms, used only where the
physics really is the same:

======================  ===============================  =============================
Common term             ALMO-EDA                         SAPT
======================  ===============================  =============================
electrostatics          electrostatic                    electrostatics
Pauli                   Pauli                            exchange
dispersion              dispersion                       dispersion
polarization            polarization                     --
charge transfer         charge transfer                  --
induction               polarization + charge transfer   induction
total                   total                            total
======================  ===============================  =============================

**SAPT induction contains both polarization and charge transfer**, so it corresponds to
ALMO's polarization *plus* charge transfer, here called *induction*. Comparing SAPT
induction with ALMO polarization alone hides charge transfer, which matters for small
cations such as Li⁺.

References
==========
.. [Horn2016a] P. R. Horn, Y. Mao and M. Head-Gordon, "Probing non-covalent
   interactions with a second generation energy decomposition analysis using
   absolutely localized molecular orbitals", *Phys. Chem. Chem. Phys.* **18**,
   23067--23079 (2016), doi:10.1039/C6CP03784D.
.. [Horn2016b] P. R. Horn, Y. Mao and M. Head-Gordon, "Defining the contributions of
   permanent electrostatics, Pauli repulsion, and dispersion in density functional
   theory calculations of intermolecular interaction energies", *J. Chem. Phys.*
   **144**, 114107 (2016), doi:10.1063/1.4942921.
.. [Horn2015] P. R. Horn and M. Head-Gordon, "Polarization contributions to
   intermolecular interactions revisited with fragment electric-field response
   functions", *J. Chem. Phys.* **143**, 114111 (2015), doi:10.1063/1.4930534.
