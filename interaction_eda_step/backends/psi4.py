# -*- coding: utf-8 -*-

"""Second-generation ALMO-EDA with Psi4, on CPUs.

The driver (``data/psi4_eda.py``) follows GPU4PySCF's ``eda.py`` step by step, with
Psi4 1.11 as a library for the integrals, the Coulomb and exchange builds and the
exchange-correlation potential and kernel; it runs with the Python of the Psi4
environment that ``psi4.ini`` names (the Psi4 step's), so it needs no environment
of its own.

Psi4 has the exchange-correlation response kernel only for LDA and GGA functionals
without VV10. So the coupled FERF response needs a functional such as wB97X-V with
VV10 left out of the kernel (``response_nlc`` false, the default here). Meta-GGAs
such as wB97M-V are refused (Paul, 2026-11-07): the uncoupled response would move
energy from polarization to charge transfer (+1.4 kJ/mol for Li+-EC at equilibrium,
up to 6 at short range) and not be comparable with GPU4PySCF's coupled terms.
"""

import os
from pathlib import Path

from .gpu4pyscf import BIBTEX, GPU4PySCF

DEFAULTS = {
    "xc": "wb97x-v",
    "basis": "def2-tzvpd",
    "auxbasis": None,
    "nlc": None,
    "disp": None,
    "svd_threshold": 1.0e-4,
    "ferf": "coupled",
    "response_nlc": False,
    "memory_gb": None,
}

PSI4_BIBTEX = (
    "@article{psi4_1.4,\n"
    "  title = {{Psi4} 1.4: Open-source software for high-throughput quantum"
    " chemistry},\n"
    "  author = {Smith, Daniel G. A. and others},\n"
    "  journal = {J. Chem. Phys.},\n"
    "  volume = {152},\n  pages = {184108},\n  year = {2020},\n"
    "  doi = {10.1063/5.0006002}\n}"
)


#: Fragments of the names of meta-GGA functionals (Psi4 has no kernel for them)
META = ("m-v", "b97m", "m06", "m05", "m08", "m11", "mn1", "scan", "tpss", "revtpss")


def is_meta(xc):
    """Whether the functional is a meta-GGA, by name."""
    xc = str(xc).lower()
    return any(m in xc for m in META)


class Psi4(GPU4PySCF):
    """Second-generation ALMO-EDA in Psi4 (1.11), on CPUs."""

    name = "Psi4"
    scheme = "ALMO-EDA2"
    device = "cpu"
    max_fragments = None
    #: The Psi4 step's ini file: its environment has Psi4.
    ini = "psi4.ini"
    driver = "psi4_eda.py"
    defaults = DEFAULTS

    def configure(self, config):
        """The ini section, to run the driver with the environment's Python.

        psi4.ini's ``code`` starts the psi4 program; the driver is a Python script
        that imports Psi4, so it runs with ``python`` in the same environment: the
        one beside the program when ``code`` gives its path, otherwise the first
        on the PATH (e.g. after the ini's conda environment is activated).
        """
        config = dict(config)
        code = config.get("code", "").split()
        if code and os.path.isabs(code[0]):
            # A path to the psi4 program: the environment's python is beside it
            config["code"] = str(Path(code[0]).with_name("python"))
        else:
            # Found on the PATH, e.g. once a conda environment is activated
            config["code"] = "python"
        return config

    def command(self):
        return super().command() + ["--threads", "{NTASKS}"]

    def problems(self, settings):
        """Why these settings cannot run with Psi4, or [] if they can."""
        s = self.settings(settings)
        out = []
        if is_meta(s["xc"]):
            out.append(
                f"Psi4 cannot run meta-GGA functionals such as {s['xc']} (it has no "
                "response kernel for them): use a GGA or hybrid functional such as "
                "wB97X-V, or GPU4PySCF."
            )
        if s["ferf"] == "coupled" and str(s["xc"]).lower().endswith("-v"):
            if s["response_nlc"]:
                out.append(
                    "Psi4 has no VV10 response kernel: set 'VV10 in the FERF "
                    "response' to off."
                )
        if s["nlc"] is False:
            out.append("VV10 off for the whole functional is not supported with Psi4.")
        if s["disp"]:
            out.append("Dispersion corrections are not supported with Psi4 yet.")
        return out

    def citations(self, settings):
        return [
            ("psi4_1.4", PSI4_BIBTEX, 1, "The Psi4 code."),
            ("almo_eda2", BIBTEX["almo_eda2"], 1, "Second-generation ALMO-EDA."),
            ("almo_eda_frozen", BIBTEX["frozen"], 2, "The frozen-term decomposition."),
            ("ferf", BIBTEX["ferf"], 2, "Polarization from FERFs."),
        ]
