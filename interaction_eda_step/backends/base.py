# -*- coding: utf-8 -*-

"""The interface between the Interaction EDA step and the codes that do the EDA.

A backend runs a batch of EDA calculations, one per set of fragments, in its own
environment through ``seamm_exec``. The step knows only this interface:

* what the backend can do (:attr:`Backend.scheme`, :attr:`Backend.device`,
  :attr:`Backend.max_fragments`, :meth:`Backend.options`), so the GUI can offer
  only valid choices;
* :meth:`Backend.files` -- the input files for a batch;
* :attr:`Backend.ini` and :meth:`Backend.command` -- how ``seamm_exec`` runs it;
* :meth:`Backend.read_results` -- the native terms of each calculation, kJ/mol.

Backends planned: ``gpu4pyscf`` (ALMO-EDA2 on a GPU), ``psi4_almo`` (ALMO-EDA2 on
CPUs) and ``psi4_sapt`` (SAPT0/F-SAPT, dimers only).
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class Calculation:
    """One EDA calculation: a set of fragments.

    Attributes
    ----------
    key : tuple(int, ...)
        Ascending fragment indices.
    fragments : [interaction_eda_step.fragments.Fragment]
    """

    key: tuple
    fragments: list

    @property
    def id(self):
        return "+".join(str(k) for k in self.key)


@dataclass
class Result:
    """The outcome of one calculation.

    Attributes
    ----------
    key : tuple(int, ...)
    energies : {str: float} or None
        Native terms, kJ/mol; None if it failed.
    error : str or None
    data : dict
        Everything else the backend reports (timing, memory, versions, ...).
    """

    key: tuple
    energies: dict = None
    error: str = None
    data: dict = field(default_factory=dict)


class Backend:
    """Base class for the EDA backends."""

    #: Short name, used in the parameters and the ini file name.
    name = None
    #: The EDA scheme (a key of :data:`interaction_eda_step.terms.SCHEMES`).
    scheme = None
    #: "gpu" or "cpu".
    device = "cpu"
    #: The most fragments in one calculation; None for no limit.
    max_fragments = None
    #: The ini file (in the installation's root) that says how to run the code.
    ini = None
    #: Results file the code writes.
    results_file = "results.jsonl"
    #: The seamm_exec program of a task (its resolver finds the code's Python).
    program = None
    #: The version of the results the driver computes from its input. Raise it when
    #: a change to the driver could change the energies. Leave it for changes that
    #: cannot (memory, threads, logging, speed): finished calculations are then
    #: kept when a job is rerun.
    results_version = 1

    def fingerprint(self, data):
        """The fingerprint of a task: what determines its results.

        ``data`` is the task's input (``input.json``). See :func:`fingerprint`.
        """
        return fingerprint(data, self.name, self.results_version)

    def task_resources(self, calc, settings, cores="automatic", memory="automatic"):
        """(seamm_exec.Resources, estimated seconds) for one calculation."""
        raise NotImplementedError

    def options(self):
        """The settings this backend accepts: {name: default}."""
        return {}

    def files(self, calculations, settings):
        """The input files for a batch of calculations: {name: text}."""
        raise NotImplementedError

    def command(self):
        """The command line, with ``{code}`` from the ini file."""
        raise NotImplementedError

    def read_results(self, directory, calculations):
        """{key: Result} for the calculations, from the results file.

        A calculation with no result is reported with an error.
        """
        raise NotImplementedError

    def method_string(self, settings):
        """A one-line description of the method, for labels and tables."""
        raise NotImplementedError

    def configure(self, config):
        """The ini section as the executor should use it (default: unchanged)."""
        return config

    def problems(self, settings):
        """Why these settings cannot run with this backend; [] if they can."""
        return []

    def citations(self, settings):
        """[(alias, bibtex, level, note)] to cite."""
        return []


#: Settings that change how a calculation runs but not its results.
RUN_ONLY_SETTINGS = ("memory_gb", "threads")


def fingerprint(data, backend, results_version):
    """The fingerprint of an EDA task, from what determines its results.

    seamm_exec otherwise fingerprints a task by its command and input files,
    which include the driver script, so any edit to the driver -- even one that
    only changes how much memory it uses -- makes every finished calculation run
    again. This uses instead the backend, its ``results_version``, and the task's
    input (settings and fragments) without the settings that only change how it
    runs (:data:`RUN_ONLY_SETTINGS`).

    Parameters
    ----------
    data : dict or str
        The task's input, as written to ``input.json`` (or its text).
    backend : str
        The backend's name, e.g. "Psi4".
    results_version : int
        The backend's results version.

    Returns
    -------
    str
        "eda-sha256:" and the hex digest.
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    settings = {
        k: v
        for k, v in (data.get("settings") or {}).items()
        if k not in RUN_ONLY_SETTINGS
    }
    content = {
        "backend": str(backend),
        "results_version": int(results_version),
        "settings": settings,
        "calculations": data.get("calculations") or [],
    }
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return "eda-sha256:" + hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    """The records of a JSON-lines file; later records replace earlier ones
    with the same id (a rerun after a failure)."""
    records = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            old = records.get(record.get("id"))
            if old is None or "error" in old or "error" not in record:
                records[record.get("id")] = record
    return records
