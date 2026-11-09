***************
Getting Started
***************

Installation
============
The Interaction EDA step is installed with the `SEAMM Manager`_. To add it, or bring it
up to date::

  seamm-manager install interaction-eda-step
  seamm-manager update interaction-eda-step

or use the Manager's window.

The EDA itself is done by GPU4PySCF_, which runs on an NVIDIA GPU under Linux and lives
in its own conda environment, ``seamm-gpu4pyscf``. Installing the step creates it: the
Manager runs the step's installer, which

* installs GPU4PySCF 1.9.1 or later with CuPy, cuTENSOR, NVIDIA's CUDA libraries and
  pyscf-dispersion (for D3/D4), choosing the CUDA build for the machine's NVIDIA driver,
  which must support CUDA 12 or later;
* checks that GPU4PySCF imports, has the second-generation ALMO-EDA and can use the GPU;
* records the environment in ``gpu4pyscf.ini`` in the SEAMM installation's directory.

On macOS nothing is installed: flowcharts with the step can be built there and run on a
GPU machine. The installer can also be run by hand, e.g. to check a machine::

  interaction-eda-step-installer install
  interaction-eda-step-installer check
  interaction-eda-step-installer update

On a cluster, run it where the jobs will run or on a login node. A login node usually
has no GPU, so the installer uses the CUDA 12 build and cannot test the GPU there; run
``check`` on a GPU node to test it.

The GPU
-------
The calculations are in double precision, so the GPU must have good double-precision
(FP64) throughput: A100, A30, V100, H100, H200 or B200. The step has been validated on
the A30 and A100. GPUs made mainly for single precision, such as the L40S, L4 and T4, are
tens of times slower for this: the installer warns about them, and the step refuses to
run on them unless its "GPUs with little FP64" setting is "allow". A 45-atom cluster of five molecules (ωB97M-V/def2-TZVPD) needs
about 2.6 GB of GPU memory and 22 minutes on an A100, and each of its pairs about 2
minutes.

Run the step in a job on a GPU node: through a SEAMM queue that targets a GPU partition
with FP64 (for example an A100 or A30 partition), or on a workstation with such a GPU.

Without a GPU: Psi4
-------------------
With "Code" set to Psi4 the step runs the same EDA on CPUs with Psi4 1.11 or later, in
the Psi4 step's environment: install the Psi4 step (``seamm-manager install
psi4-step``) and the step finds Psi4 through its ``psi4.ini``. It runs the EDA script
with the Python of that environment -- the one beside ``psi4`` when ``code`` gives its
path, otherwise the first on the PATH once the conda environment is activated. It uses
the job's cores as threads. Psi4 suits GGA and hybrid functionals such as ωB97X-V, and
Hartree-Fock; see :ref:`settings` for what it does not offer and how long it takes.

Configuring the step: gpu4pyscf.ini
-----------------------------------
The step finds GPU4PySCF through ``gpu4pyscf.ini`` in the SEAMM installation's
directory (``~/SEAMM/gpu4pyscf.ini`` by default), which the installer writes:

.. code-block:: ini

    [local]
    installation = conda
    conda-environment = seamm-gpu4pyscf
    code = python
    cuda-build = auto

``cuda-build`` forces a build of GPU4PySCF: ``cuda12x`` (the default, which also runs
with CUDA 13 drivers) or ``cuda13x``.

To use an environment of your own instead, with GPU4PySCF 1.9.1 or later::

  python -m venv ~/envs/gpu4pyscf
  ~/envs/gpu4pyscf/bin/pip install "gpu4pyscf-cuda12x>=1.9.1" "cupy-cuda12x[ctk]" \
      cutensor-cu12 pyscf-dispersion

point ``gpu4pyscf.ini`` at its Python, and the installer will leave it alone:

.. code-block:: ini

    [local]
    installation = local
    code = /home/me/envs/gpu4pyscf/bin/python

If importing GPU4PySCF then fails with "Unable to load libcusolver", link the versioned
library to the unversioned name in the environment's ``nvidia/cusolver/lib`` directory
(``ln -s libcusolver.so.11 libcusolver.so``); the installer does this for its own
environment.

Each section of ``gpu4pyscf.ini`` is for one executor; ``local`` runs on the machine
where the job runs.

A first calculation
===================
A flowchart that reads a cluster and decomposes its interactions::

  title: Interaction EDA of a Li+ solvation shell
  steps:
  - Read Structure:
      file: shell.sdf
  - Interaction EDA: {}

The structure needs its bonds, which define the molecules, and the charges of any ions,
e.g. as formal charges (``M  CHG``) in an SDF file. Common ions (Li⁺, Na⁺, PF₆⁻, BF₄⁻,
halides, ...) get their charges automatically. With the defaults the step computes the
whole cluster and every pair with ωB97M-V/def2-TZVPD, and prints:

* the fragments, with their charges;
* each EDA term for the whole cluster, its 2-body part (the sum over the pairs) and
  its exact ≥3-body part;
* the terms of each pair.

It also writes ``eda.extxyz`` with the terms as labels, and ``eda.json`` with
everything. See the :ref:`user-guide` for what the terms mean and all the settings.

.. _SEAMM Manager: https://molssi-seamm.github.io/getting_started/installation/seamm-manager.html
.. _GPU4PySCF: https://github.com/pyscf/gpu4pyscf
