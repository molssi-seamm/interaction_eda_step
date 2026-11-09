=======
History
=======
2026.11.9.6 -- Psi4: more memory for the EDA's integrals, so large clusters keep them in core
    * The EDA's density-fitted integrals may now use 75 % of Psi4's memory instead of
      half. The full system's SCF has freed its own integrals by then, so these are
      the only large ones held. With half, a 1121-function cluster had 1 % to spare
      and a 1388-function cluster none, and would have read its integrals from disk
      at every Fock build, which is much slower.
    * The memory asked for is unchanged in practice, now stated as enough for the
      peak: about 1.9 times the in-core integrals (98 GB for a 1121-function
      cluster), with 30 % to spare.

2026.11.9.5 -- Bugfix: Psi4 time estimates for small calculations were too short
    * The time estimate for each Psi4 calculation now includes a fixed 3 minutes for
      setting up Psi4, the basis sets, the density fitting and the grids. Small pairs
      took 4-5 minutes on 8 cores, against about 80 s estimated, so bundles of many
      small calculations asked for too little time and ran out before the last ones,
      which then had to be queued again.

2026.11.9.4 -- Every calculation as its own task, and many structures in one step
    * Each EDA calculation -- each set of fragments of each structure -- now runs as
      its own task, asking for the cores, memory and time its size needs (from its
      number of basis functions and fragments). In the job's own cores they run side
      by side; on a queue that sends tasks to the scheduler, as batch jobs bundled
      together. A cluster's pairs no longer wait for, or get the cores of, the
      cluster.
    * The step can analyze many structures at once, with SEAMM's standard structure
      selection, e.g. every system read from a file. Several structures give
      eda.jsonl, one line per structure, and one eda.extxyz with all of them.
    * New settings for Psi4: "Cores per calculation" (automatic: 8 up to 400 basis
      functions, 16 up to 800, 32 above) and "Memory per calculation" (automatic:
      enough to keep the density-fitted integrals in memory, at most 240 GB).
    * Each calculation's input, log and results are in tasks/<key> under the step.

2026.11.9.3 -- Bugfix: Psi4 density-fitted SCFs stopped for lack of memory
    * With density fitting and a range-separated functional such as ωB97X-V, Psi4's
      SCFs could stop with "SCF_SUBTYPE=INCORE was specified, but there is not
      enough memory to do in-core", even for small clusters with plenty of memory:
      Psi4 gave almost all of it to the DFT grid, because its estimate for the
      in-core integrals leaves out the range-separated ones (Na⁺(H₂O)₆: 2.1 of
      12.8 GB). The SCFs now use Psi4's DISK_DF algorithm, which keeps the integrals
      in memory when they fit and otherwise reads them from disk. The energies are
      unchanged (to 3e-8 kJ/mol).
    * The EDA's own integrals fall back to DISK_DF too when they do not fit in
      memory, rather than stopping.
    * After a calculation fails, Psi4 is reset, so the next calculation of the
      structure starts clean. Before, the next one failed as well and the one after
      that hung until the job's time ran out.

2026.11.9.2 -- Psi4 as the EDA code, on CPUs
    * The Interaction EDA step now offers Psi4 as well as GPU4PySCF. Psi4 runs the
      same second-generation ALMO-EDA on CPUs, with the Psi4 step's environment
      (psi4.ini), for GGA and hybrid functionals such as ωB97X-V and for
      Hartree-Fock. Meta-GGAs such as ωB97M-V are not offered with Psi4, because it
      has no response kernel for them. With density fitting it agrees with GPU4PySCF
      to 0.021 kJ/mol or better per term for pairs.
    * eda.json and eda.extxyz now record the structure's system and configuration
      names, so results can be matched to their input.
    * The Psi4 driver now keeps its density-fitted integrals in memory only when
      they fit: for a range-separated functional, about three tensors of 8 × N_aux ×
      N² bytes. Before, a large cluster could stop with "not enough memory to do
      in-core" when it should have read them from disk.
    * The user guide gives the measured times, the cores worth using and the memory
      needed with Psi4.

2026.11.9.1 -- Fine integration grids for GPU4PySCF by default
    * GPU4PySCF's standard grids, (99,590) for the exchange-correlation and (50,194)
      for VV10 with pruning, are not converged for compact clusters: for Na⁺(H₂O)₆ they
      put 0.43 kJ/mol into the frozen term and 0.33 kJ/mol into dispersion, compared
      with Psi4 at the same nominal grids. The new "Integration grid" setting defaults
      to "fine", (150,974) and (99,590), which agrees with Psi4 to 0.012 kJ/mol for
      the whole cluster. Pairs change by about 0.001 kJ/mol.
    * "standard" keeps the earlier grids, about 7 times faster for a cluster, e.g. to
      match labels made with 2026.11.9 or earlier. The EDA method string now records
      the grids.

2026.11.9 -- Psi4 backend: half the memory for density fitting, and memory from the allocation
    * The Psi4 backend (not yet offered by the step) held two sets of density-fitted
      integrals at once: its EDA engine's and those of the final SCF of the whole
      system. The SCF now runs first, and Psi4 frees its integrals before the engine
      builds its own, so only one set is in memory (a 1121-function cluster peaked at
      141 GB with two). The terms are unchanged (to 1e-6 kJ/mol).
    * When the integrals would not fit in memory, density fitting falls back to
      reading them from the scratch disk at each build (DISK_DF), slower but bounded.
    * The step gives Psi4 80 % of the calculation's memory; it ran with Psi4's default
      of 4 GB before.

2026.11.8.1 -- Internal: the Psi4 EDA driver's integrals
    * The Psi4 backend (not yet offered by the step) computes its integrals directly
      instead of keeping them in memory, which was 4.5 times slower for the EDA's many
      Fock builds (HF Li+-EC: 260 s against 1179 s on 16 cores, the same terms), and
      its density-fitted path, which stopped at once, now works: the fitting basis is
      built explicitly. Density fitting is about 6-9 times faster, with terms within
      0.03-0.06 kJ/mol of exact integrals.

2026.11.8 -- The SAPT-comparable Pauli term in the extxyz, and where each EDA ran
    * eda.extxyz now has ``EDA_ALMOEDA2_mod_pauli``, the SAPT-comparable Pauli
      repulsion (Q-Chem's MOD PAULI: frozen - classical electrostatic - dispersion),
      with its many-body parts, beside the orthogonal ``pauli``. It and
      ``classical_electrostatic`` are the pair to compare with SAPT.
    * Each frame and calculation records where it ran (``EDA_platform``: GPU, CUDA,
      driver and library versions).
    * The calculations use the number of cores SEAMM's ``ncores`` option allows.

2026.11.7 -- First release: interaction energy decomposition with ALMO-EDA2 on GPUs
    * A new step, Interaction EDA, that decomposes the interactions between the
      molecules of a structure into electrostatics, Pauli repulsion, dispersion,
      polarization and charge transfer, using second-generation ALMO-EDA in GPU4PySCF
      on an NVIDIA GPU.
    * It computes the whole cluster and every pair, and optionally the triples within
      a distance, and gives each term's 2-body, exact ≥3-body and (with triples)
      3-body parts. Fragments are the molecules, with ion charges from their formal
      charges or a built-in catalog, or are given by an atom column.
    * Each term is printed, stored as results and properties, and written per set of
      fragments to eda.extxyz and eda.json. The common electrostatics and Pauli terms
      are the pair comparable with SAPT; the native ALMO-EDA2 terms are kept as well.
    * The defaults are the validated settings: ωB97M-V/def2-TZVPD with exact
      integrals and coupled FERF response (singular-value threshold 10⁻⁴). Validated
      to better than 10⁻⁵ kJ/mol against standalone GPU4PySCF runs on A30 and A100
      GPUs, for cations, anions and neutral molecules.
    * The installer creates the GPU4PySCF conda environment for the machine's CUDA
      driver and checks that it can use the GPU. The step refuses to run on GPUs with
      little double-precision throughput (T4, L4, L40S ...) unless told to allow them.
    * A rerun computes only the calculations that are missing.

2026.11.6 (2026-11-06)
----------------------

* Plug-in created using the SEAMM plug-in cookiecutter.
