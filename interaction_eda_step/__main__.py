# -*- coding: utf-8 -*-

"""Handle the installation of the Interaction EDA step."""

from .installer import Installer


def run():
    """Handle the extra installation needed.

    * Create the conda environment `seamm-gpu4pyscf` with GPU4PySCF for this
      machine's NVIDIA driver.
    * Add or update information in the SEAMM gpu4pyscf.ini file.
    """
    installer = Installer()
    installer.run()


if __name__ == "__main__":
    run()
