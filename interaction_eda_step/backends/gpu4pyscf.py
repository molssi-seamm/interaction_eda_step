# -*- coding: utf-8 -*-

"""ALMO-EDA2 with GPU4PySCF on a GPU."""

import importlib.resources
import json
from pathlib import Path

from .base import Backend, Result, read_jsonl

DRIVER = "gpu4pyscf_eda.py"

# The settings validated against SAPT and CCSD(T) and on ChemAI and ARC
# (2026-11-06): wB97M-V/def2-TZVPD, no density fitting, FERF SVD threshold 1e-4,
# coupled FERF-DQ response.
DEFAULTS = {
    "xc": "wb97m-v",
    "basis": "def2-tzvpd",
    "auxbasis": None,
    "nlc": None,
    "disp": None,
    "svd_threshold": 1.0e-4,
    "ferf": "coupled",
    "response_nlc": True,
    "verbose": 3,
}

BIBTEX = {
    "gpu4pyscf": (
        "@article{gpu4pyscf,\n"
        "  title = {Introducing {GPU} Acceleration into the {Python}-Based"
        " Simulations of Chemistry Framework},\n"
        "  author = {Li, Rui and Sun, Qiming and Zhang, Xing and Chan, Garnet"
        " Kin-Lic},\n"
        "  journal = {J. Phys. Chem. A},\n"
        "  volume = {129},\n  number = {5},\n  pages = {1459--1468},\n"
        "  year = {2025},\n"
        "  doi = {10.1021/acs.jpca.4c05876}\n}"
    ),
    "pyscf": (
        "@article{pyscf2020,\n"
        "  title = {Recent developments in the {PySCF} program package},\n"
        "  author = {Sun, Qiming and others},\n"
        "  journal = {J. Chem. Phys.},\n"
        "  volume = {153},\n  pages = {024109},\n  year = {2020},\n"
        "  doi = {10.1063/5.0006074}\n}"
    ),
    "almo_eda2": (
        "@article{almo_eda2,\n"
        "  title = {Probing non-covalent interactions with a second generation"
        " energy decomposition analysis using absolutely localized molecular"
        " orbitals},\n"
        "  author = {Horn, Paul R. and Mao, Yuezhi and Head-Gordon, Martin},\n"
        "  journal = {Phys. Chem. Chem. Phys.},\n"
        "  volume = {18},\n  pages = {23067--23079},\n  year = {2016},\n"
        "  doi = {10.1039/C6CP03784D}\n}"
    ),
    "frozen": (
        "@article{almo_eda_frozen,\n"
        "  title = {Defining the contributions of permanent electrostatics,"
        " {Pauli} repulsion, and dispersion in density functional theory"
        " calculations of intermolecular interaction energies},\n"
        "  author = {Horn, Paul R. and Mao, Yuezhi and Head-Gordon, Martin},\n"
        "  journal = {J. Chem. Phys.},\n"
        "  volume = {144},\n  pages = {114107},\n  year = {2016},\n"
        "  doi = {10.1063/1.4942921}\n}"
    ),
    "ferf": (
        "@article{ferf,\n"
        "  title = {Polarization contributions to intermolecular interactions"
        " revisited with fragment electric-field response functions},\n"
        "  author = {Horn, Paul R. and Head-Gordon, Martin},\n"
        "  journal = {J. Chem. Phys.},\n"
        "  volume = {143},\n  pages = {114111},\n  year = {2015},\n"
        "  doi = {10.1063/1.4930534}\n}"
    ),
}


class GPU4PySCF(Backend):
    """Second-generation ALMO-EDA in GPU4PySCF (1.9 or later)."""

    name = "GPU4PySCF"
    scheme = "ALMO-EDA2"
    device = "gpu"
    max_fragments = None
    ini = "gpu4pyscf.ini"

    def options(self):
        return dict(DEFAULTS)

    def settings(self, settings):
        """The full settings: the defaults updated with those given."""
        out = dict(DEFAULTS)
        out.update({k: v for k, v in (settings or {}).items() if k in DEFAULTS})
        return out

    def files(self, calculations, settings):
        data = {
            "settings": self.settings(settings),
            "calculations": [
                {
                    "id": calc.id,
                    "fragments": [
                        {
                            "label": f.label(),
                            "symbols": list(f.symbols),
                            "coordinates": [
                                [float(x) for x in row] for row in f.coordinates
                            ],
                            "charge": int(f.charge),
                            "spin": int(f.multiplicity) - 1,
                        }
                        for f in calc.fragments
                    ],
                }
                for calc in calculations
            ],
        }
        driver = (
            importlib.resources.files("interaction_eda_step") / "data" / DRIVER
        ).read_text()
        return {
            "input.json": json.dumps(data, indent=1),
            DRIVER: driver,
        }

    def command(self):
        # The logs go in the step's directory, not a subdirectory: the executor's
        # in-situ clean-up (seamm_exec 2026.11.5) cannot remove a new directory
        # holding files it returns.
        return [
            "{code}",
            DRIVER,
            "input.json",
            "-o",
            self.results_file,
            "--logdir",
            ".",
        ]

    def read_results(self, directory, calculations):
        records = read_jsonl(Path(directory) / self.results_file)
        out = {}
        for calc in calculations:
            record = records.get(calc.id)
            if record is None:
                out[calc.key] = Result(calc.key, error="no result (not run?)")
            elif "error" in record and "eda_kJ_mol" not in record:
                out[calc.key] = Result(calc.key, error=record["error"], data=record)
            else:
                out[calc.key] = Result(
                    calc.key,
                    energies={k: float(v) for k, v in record["eda_kJ_mol"].items()},
                    data=record,
                )
        return out

    def method_string(self, settings):
        s = self.settings(settings)
        parts = [
            f"scheme={self.scheme}",
            f"backend={self.name}",
            f"xc={s['xc']}",
            f"basis={s['basis']}",
            f"auxbasis={s['auxbasis'] or 'none'}",
            f"nlc={'default' if s['nlc'] is None else s['nlc']}",
            f"response_nlc={'on' if s['response_nlc'] else 'off'}",
            f"disp={s['disp'] or 'none'}",
            f"ferf={s['ferf']}",
            f"svd_threshold={s['svd_threshold']:g}",
        ]
        return ";".join(parts)

    def citations(self, settings):
        return [
            ("gpu4pyscf", BIBTEX["gpu4pyscf"], 1, "The GPU4PySCF code."),
            ("pyscf2020", BIBTEX["pyscf"], 1, "PySCF, which GPU4PySCF builds on."),
            ("almo_eda2", BIBTEX["almo_eda2"], 1, "Second-generation ALMO-EDA."),
            ("almo_eda_frozen", BIBTEX["frozen"], 2, "The frozen-term decomposition."),
            ("ferf", BIBTEX["ferf"], 2, "Polarization from FERFs."),
        ]
