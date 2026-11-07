=======
History
=======
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
