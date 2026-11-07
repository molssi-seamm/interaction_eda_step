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

The EDA itself is done by GPU4PySCF_, which runs on an NVIDIA GPU and lives in its own
Python environment. The step runs a small driver script with that environment's Python,
so the environment needs GPU4PySCF and its dependencies but nothing from SEAMM.

The GPU
-------
The calculations are in double precision, so the GPU must have good double-precision
(FP64) throughput: A100, A30, H100, H200 or B200. GPUs made mainly for single precision,
such as the L40S and T4, are very slow for this. A 45-atom cluster of five molecules
(ωB97M-V/def2-TZVPD) needs about 2.6 GB of GPU memory and 22 minutes on an A100, and
each of its pairs about 2 minutes.

Run the step in a job on a GPU node: through a SEAMM queue that targets a GPU partition
with FP64 (for example an A100 or A30 partition), or on a workstation with such a GPU.

The GPU4PySCF environment
-------------------------
Create a virtual environment with Python 3.11 or 3.12 and install GPU4PySCF for the
CUDA version of the GPU's driver, e.g. for CUDA 12::

  python -m venv ~/envs/gpu4pyscf
  ~/envs/gpu4pyscf/bin/pip install "gpu4pyscf-cuda12x>=1.9" "cupy-cuda12x[ctk]" \
      cutensor-cu12 pyscf-dispersion

GPU4PySCF 1.9 or later is needed for the second-generation ALMO-EDA.
``pyscf-dispersion`` is needed only for D3/D4 dispersion corrections. If importing
GPU4PySCF fails with "Unable to load libcusolver", link the versioned library to the
unversioned name in the environment's ``nvidia/cusolver/lib`` directory
(``ln -s libcusolver.so.11 libcusolver.so``).

Configuring the step: gpu4pyscf.ini
-----------------------------------
The step finds the environment through ``gpu4pyscf.ini`` in the SEAMM installation's
directory (``~/SEAMM/gpu4pyscf.ini`` by default). The first run of the step creates it
from the template below if it is missing; edit ``code`` to be the Python of the GPU4PySCF
environment:

.. code-block:: ini

    # How to run GPU4PySCF for the Interaction EDA step.
    [local]
    installation = local
    code = /home/me/envs/gpu4pyscf/bin/python

Each section is for one executor; ``local`` runs on the machine where the job runs.

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
