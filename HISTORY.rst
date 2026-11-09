=======
History
=======
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
