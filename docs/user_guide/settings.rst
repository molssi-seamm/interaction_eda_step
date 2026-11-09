.. _settings:

********
Settings
********

The defaults are the settings that were validated against SAPT and coupled-cluster
references and between GPUs: ωB97M-V/def2-TZVPD, exact integrals, coupled FERF-DQ
response with a singular-value threshold of 10⁻⁴, and fine integration grids.

Structures and resources
========================
Structures
    Which structures to analyze, with SEAMM's standard choices: the current
    configuration (the default), the current configuration of every system, all
    configurations, or those whose names match. One step can analyze a whole file of
    structures read with Read Structure -- set its system and configuration names to
    ``title`` so the results carry the names from the file -- with no Loop.

    Every calculation, each set of fragments of each structure, runs as its own task:
    in the job's own cores, or, on a queue that sends tasks to the scheduler
    (``tasks = queue``), as batch jobs, several bundled together. Each task asks for
    the cores, memory and time its size needs, and a rerun keeps the calculations that
    have finished.

Cores per calculation
    With Psi4, the threads for each calculation. ``automatic`` uses 8 up to 400 basis
    functions, 16 up to 800 and 32 above, which is where the extra cores still pay
    (see :ref:`settings` under Code). GPU4PySCF calculations use one GPU and four
    cores.

Memory per calculation
    With Psi4, the memory for each calculation, of which Psi4 gets 80 %. ``automatic``
    sizes it so the density-fitted integrals of the EDA stay in memory: about 10 GB
    for a Li⁺–EC pair, 20 GB for an EC–EC pair, 130 GB for a 1121-function cluster,
    and at most 240 GB, above which they are read from disk.

Method
======
Code
    The code that does the EDA. Both run second-generation ALMO-EDA (see
    :ref:`eda-terms`) with the same algorithm and settings, and give the same terms.

    GPU4PySCF
        Runs on a GPU with good double-precision (FP64) throughput: A100, A30,
        H100/H200, B200. It handles every functional, including meta-GGAs such as
        ωB97M-V with the coupled response, and is much faster.
    Psi4
        Runs on CPUs, with the Psi4 step's environment (``psi4.ini``), so it needs no
        GPU and no environment of its own. Psi4 has the exchange-correlation response
        kernel only for LDA and GGA functionals without VV10, so:

        * VV10 is always kept in the energies and always left out of the FERF
          response (the settings for VV10 are not shown);
        * meta-GGA functionals such as ωB97M-V or B97M-V are not available: the
          uncoupled response would move energy from polarization into charge
          transfer -- +1.4 kJ/mol for Li⁺–EC at equilibrium, up to 6 kJ/mol at short
          range -- and the terms would not be comparable with coupled ones. Use
          GPU4PySCF for them;
        * dispersion corrections are not available yet.

        Use density fitting (``def2-universal-jkfit``) with Psi4: with exact integrals
        the many Fock builds of the EDA are several times slower. With density
        fitting, ωB97X-V/def2-TZVPD on AMD EPYC cores (TinkerCliffs, 2026-11-09)
        takes:

        * Li⁺–EC (284 basis functions): 16 minutes on 4 cores, 8 on 8, 7 on 16, 4
          on 32;
        * EC–EC (552): 37 minutes on 8 cores, 16 on 16, 11 on 32;
        * a Li⁺ shell of two EC and two DMC (1121): about 2 hours on 16 cores.

        Psi4 uses extra cores less and less well: 8 cores are the most economical
        up to about 400 basis functions, 16 up to about 800, and 32 above. The
        density-fitted integrals are kept in memory when they fit, which for a
        range-separated functional takes about 3 × 8 × N\ :sub:`aux` × N\ :sup:`2`
        bytes -- about 70 GB for the 1121-function cluster, so give such a
        calculation about 130 GB. With less, Psi4 reads them from the scratch disk
        at every Fock build, which is slower. The step gives Psi4 80 % of the
        calculation's memory.

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
    total. Shown only when VV10 is kept, with GPU4PySCF.

Dispersion correction
    GPU4PySCF only. An empirical correction added to the functional, in PySCF's names: ``d4``,
    ``d3bj``, or a specific parameter set such as ``d4:wb97x``. It is part of the
    dispersion term; D4's three-body (ATM) part shows up in the ≥3-body dispersion.

FERF response
    ``coupled`` solves the coupled-perturbed equations for the fragments' response to
    electric fields (the default). ``uncoupled`` uses orbital-energy denominators
    without the exchange-correlation kernel, which is cheaper. Only the split between
    polarization and charge transfer changes: uncoupled moves energy from polarization
    into charge transfer, +1.4 kJ/mol for Li⁺–EC at equilibrium and up to 6 kJ/mol at
    short range with ωB97M-V, so uncoupled terms are not comparable with coupled ones.

FERF singular-value threshold
    Response directions with singular values below this are dropped from the
    polarization space. Like the response, it affects only the polarization/charge
    transfer split. 10⁻⁴ is converged: for Li⁺–PF₆⁻ and PF₆⁻–EC, 10⁻⁵ changes the
    split by less than 10⁻⁸ kJ/mol, while 10⁻³ moves 0.06 kJ/mol from polarization to
    charge transfer for Li⁺–PF₆⁻. The step warns if the threshold is looser than
    10⁻⁴ and there is a cation.

Integration grid
    The DFT integration grids, as (radial, angular) points per atom. ``fine`` (the
    default) uses (150,974) for the exchange-correlation and (99,590) for VV10.
    ``standard`` is GPU4PySCF's own choice, (99,590) and (50,194) with pruning: about 7
    times faster for a cluster, but not converged for compact clusters. For
    Na⁺(H₂O)₆ the standard grids put 0.43 kJ/mol into the frozen term and 0.33 kJ/mol
    into dispersion, with the two grids contributing about equally; the fine grids
    agree with Psi4 to 0.012 kJ/mol in the energy of the whole cluster. Pairs hardly
    change (0.001 kJ/mol). Labels computed before this setting existed (2026.11.9 and
    earlier) used the standard grids.

GPUs with little FP64
    GPU4PySCF only. ``refuse`` (the default) stops before any calculation if the GPU is one made mainly
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
    interaction. A split or merge that leaves every piece with an even number of
    electrons is not caught this way, so check unusual structures, or give the
    fragments with ``atom column``.

Atom column
    The column for ``atom column``.

Calculations
    ``cluster and pairs`` (the default) gives each term's 2-body and exact ≥3-body
    parts. ``cluster, pairs and triples`` adds the 3-body increments of the triples.
    ``cluster`` or ``pairs`` alone give only those.

Triples within
    Use only the triples whose three pairs are all within this distance (the shortest
    atom-atom distance between the two fragments), or ``all``.
