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
        The permanent electrostatics of the fragments' densities after an orthogonal
        decomposition of the frozen wavefunction [Horn2016b]_ (Q-Chem's ``ELEC``). It
        is more attractive than the classical electrostatics below, by as much as
        30 kJ/mol for an anion in contact with a solvent molecule.
    Pauli
        The repulsion from antisymmetrizing the fragments' occupied orbitals, in the
        same orthogonal decomposition: frozen − electrostatic − dispersion.
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
    The Coulomb interaction of the isolated fragments' charge densities and nuclei,
    including charge penetration (Q-Chem's ``CLS ELEC``). This is what SAPT calls
    electrostatics: for Li⁺–PF₆⁻ and PF₆⁻–EC it agrees with SAPT2+/aug-cc-pVDZ to
    1.6 kJ/mol. It is exactly pairwise additive, so its many-body part is zero, which
    is a useful check.

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

======================  ===========================================  ================
Common term             ALMO-EDA                                     SAPT
======================  ===========================================  ================
electrostatics          classical electrostatic                      electrostatics
Pauli                   frozen − classical electrostatic − disp.     exchange
dispersion              dispersion                                   dispersion
frozen                  frozen                                       --
polarization            polarization                                 --
charge transfer         charge transfer                              --
induction               polarization + charge transfer               induction
total                   total                                        total
======================  ===========================================  ================

**ALMO-EDA has two electrostatic terms, and only the classical one is comparable with
SAPT.**

* SAPT's electrostatics is the Coulomb interaction of the isolated monomers' densities,
  which is ALMO's *classical electrostatic* term (Q-Chem's ``CLS ELEC``). The common
  Pauli term is what is left of the frozen term, frozen − classical electrostatic −
  dispersion (Q-Chem's ``MOD PAULI``).
* ALMO's own *electrostatic* and *Pauli* terms come from the orthogonal decomposition
  of the frozen wavefunction [Horn2016b]_ (Q-Chem's ``ELEC`` and ``PAULI``). They are
  not comparable with SAPT.

For PF₆⁻–EC (ωB97M-V/def2-TZVPD against SAPT2+/aug-cc-pVDZ):

* the classical electrostatics is within 1.6 kJ/mol of SAPT's electrostatics, and the
  common Pauli term within 1.7 kJ/mol of SAPT's exchange;
* the orthogonal-decomposition terms are each 36 kJ/mol away.

Both pairs add up to the same thing: frozen = electrostatics + Pauli + dispersion holds
in both. The step keeps the native terms under their own names as well.

Because the classical electrostatics is pairwise additive, the common ≥3-body
electrostatics is exactly zero. The many-body part of the frozen term, other than
dispersion, then appears entirely in the common Pauli term.

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
