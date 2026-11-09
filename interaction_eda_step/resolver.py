# -*- coding: utf-8 -*-

"""Resolvers for the EDA tasks, so they run wherever SEAMM runs them.

Each EDA calculation is a ``seamm_exec`` task whose program is ``psi4_eda`` or
``gpu4pyscf_eda``. There is no ini file of that name: on the machine that runs the
task, the resolver reads the code's own ini file (``psi4.ini``, the Psi4 step's,
or ``gpu4pyscf.ini``) and points ``code`` at the Python of that environment, as
the backend does for a run in the step itself. Registered as entry points in the
group ``org.molssi.seamm.exec.resolvers``.
"""

import configparser
from pathlib import Path


def _resolver(backend_class):
    def resolve(config, cmd, env, ce, root):
        backend = backend_class()
        section = dict(config or {})
        if not section.get("code"):
            path = Path(root).expanduser() / backend.ini if root else None
            if path is None or not path.exists():
                raise RuntimeError(
                    f"{backend.name} needs {path}, which its SEAMM step installs."
                )
            full = configparser.ConfigParser()
            full.read(path)
            name = "local" if "local" in full else None
            if name is None:
                raise RuntimeError(f"No [local] section in {path}.")
            section = dict(full.items(name))
        return backend.configure(section), cmd, env

    def available(root):
        return bool(root) and (Path(root).expanduser() / backend_class.ini).exists()

    resolve.available = available
    return resolve


def _psi4():
    from .backends.psi4 import Psi4

    return Psi4


def _gpu4pyscf():
    from .backends.gpu4pyscf import GPU4PySCF

    return GPU4PySCF


class _Lazy:
    """Defer importing the backends until a task is resolved."""

    def __init__(self, getter):
        self._getter = getter
        self._function = None

    def _get(self):
        if self._function is None:
            self._function = _resolver(self._getter())
        return self._function

    def __call__(self, *args, **kwargs):
        return self._get()(*args, **kwargs)

    def available(self, root):
        return self._get().available(root)


resolve_psi4 = _Lazy(_psi4)
resolve_gpu4pyscf = _Lazy(_gpu4pyscf)
