.. _settings:

********
Settings
********

The defaults are the settings that were validated against SAPT and coupled-cluster
references and between GPUs: ωB97M-V/def2-TZVPD, exact integrals, coupled FERF-DQ
response with a singular-value threshold of 10⁻⁴.

Method
======
Code
    The code that does the EDA. GPU4PySCF runs second-generation ALMO-EDA on a GPU (see
    :ref:`eda-terms`).

Functional
    The density functional, in PySCF's names (case does not matter), e.g. ``wB97M-V``,
    ``wB97X-V``, ``B97M-V``.

Basis set
    The orbital basis set, e.g. ``def2-TZVPD``. Diffuse functions (the "D") matter for
    anions and for polarization.

Density fitting
    ``no`` for exact integrals (the default), or an auxiliary basis such as
    ``def2-universal-jkfit``, which is faster but changes the energies slightly.

VV10 nonlocal correlation
    ``as in the functional`` keeps the VV10 nonlocal correlation of a "-V" functional.
    ``off (whole functional)`` removes it everywhere -- a different functional with
    different energies -- e.g. ωB97X-V without VV10 plus D4 dispersion, which is
    Najibi and Goerigk's ωB97X-D4.

VV10 in the FERF response
    ``off`` drops VV10 from the response kernel that defines the polarization space
    only; the SCF and SCF-MI energies keep the full functional. This matches codes that
    have no VV10 response kernel, such as Psi4. For a Li⁺ shell it moved energy between
    polarization and charge transfer by at most 0.0014 kJ/mol, and does not change the
    total. Shown only when VV10 is kept.

Dispersion correction
    An empirical correction added to the functional, in PySCF's names: ``d4``,
    ``d3bj``, or a specific parameter set such as ``d4:wb97x``. It is part of the
    dispersion term; D4's three-body (ATM) part shows up in the ≥3-body dispersion.

FERF response
    ``coupled`` solves the coupled-perturbed equations for the fragments' response to
    electric fields (the default). ``uncoupled`` uses orbital-energy denominators
    without the exchange-correlation kernel, which is cheaper. Only the split between
    polarization and charge transfer changes: by about 1 % of the polarization for a Li⁺
    shell.

FERF singular-value threshold
    Response directions with singular values below this are dropped from the
    polarization space. Like the response, it affects only the polarization/charge
    transfer split. 10⁻⁴ is converged: for Li⁺–PF₆⁻ and PF₆⁻–EC, 10⁻⁵ changes the
    split by less than 10⁻⁸ kJ/mol, while 10⁻³ moves 0.06 kJ/mol from polarization to
    charge transfer for Li⁺–PF₆⁻. The step warns if the threshold is looser than
    10⁻⁴ and there is a cation.

GPUs with little FP64
    ``refuse`` (the default) stops before any calculation if the GPU is one made mainly
    for single precision (T4, L4, L40S, A10, A40, RTX ...), where the EDA runs tens of
    times slower; the job ends with a message saying so. ``allow`` runs anyway, with a
    warning.

Fragments and calculations
==========================
Fragments
    ``molecules`` (the default) makes each molecule -- each bonded group of atoms -- a
    fragment. Its charge comes from the atoms' formal charges or, for common molecules
    and ions, from a built-in catalog (water, carbonates, Li⁺, Na⁺, K⁺, PF₆⁻, BF₄⁻,
    halides). ``atom column`` assigns the atoms to fragments with an integer atom
    column, with charges summed from the formal charges. Every fragment must be
    closed-shell.

    The molecules come from the bonds, which for formats without bonds (e.g. extended
    XYZ) Read Structure finds from the distances. Alkali and alkaline-earth ions such
    as Li⁺ and Na⁺ are never bonded, so an ion stays its own fragment even in close
    contact. A badly distorted structure can still break a molecule apart: a C--H bond
    stretched to 1.3--1.6 Å, for example, leaves the hydrogen as a separate
    "molecule". The step then stops with an error naming the molecule whose electron
    count is odd, rather than guessing a partition and computing the wrong
    interaction. Check such structures, or give the fragments with ``atom column``.

Atom column
    The column for ``atom column``.

Calculations
    ``cluster and pairs`` (the default) gives each term's 2-body and exact ≥3-body
    parts. ``cluster, pairs and triples`` adds the 3-body increments of the triples.
    ``cluster`` or ``pairs`` alone give only those.

Triples within
    Use only the triples whose three pairs are all within this distance (the shortest
    atom-atom distance between the two fragments), or ``all``.
