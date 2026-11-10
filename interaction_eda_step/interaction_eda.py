# -*- coding: utf-8 -*-

"""Non-graphical part of the Interaction EDA step in a SEAMM flowchart"""

import importlib.resources
import json
import logging
from pathlib import Path
import pprint  # noqa: F401
import time

from tabulate import tabulate

import interaction_eda_step
from . import fragments as F
from . import manybody
from .backends import BACKENDS, Calculation, Result
from .terms import SCHEMES
import molsystem
import seamm
import seamm_exec
from seamm_util import ureg, Q_  # noqa: F401
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

# In addition to the normal logger, two logger-like printing facilities are
# defined: "job" and "printer". "job" send output to the main job.out file for
# the job, and should be used very sparingly, typically to echo what this step
# will do in the initial summary of the job.
#
# "printer" sends output to the file "step.out" in this steps working
# directory, and is used for all normal output from this step.

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Interaction EDA")

# Add this module's properties to the standard properties
path = importlib.resources.files("interaction_eda_step") / "data"
csv_file = path / "properties.csv"
if path.exists():
    molsystem.add_properties_from_file(csv_file)


def svd_warning(settings, fragments):
    """A warning when the FERF SVD threshold is looser than 1e-4 with a cation.

    The polarization/charge-transfer split for Li+-PF6- moved 0.062 kJ/mol between
    1e-4 and 1e-3; at 1e-4 it is converged to 1e-8.
    """
    threshold = settings.get("svd_threshold")
    cations = [f.label() for f in fragments if f.charge > 0]
    if threshold is not None and threshold > 1.0e-4 and cations:
        return (
            f"The FERF singular-value threshold {threshold:g} is looser than 1e-4 "
            f"and there are cations ({', '.join(cations)}): the split between "
            "polarization and charge transfer may not be converged (0.06 kJ/mol "
            "for Li+-PF6- at 1e-3). Use 1e-4 or tighter."
        )
    return None


class InteractionEda(seamm.Node):
    """
    The non-graphical part of a Interaction EDA step in a flowchart.

    Attributes
    ----------
    parser : configargparse.ArgParser
        The parser object.

    options : tuple
        It contains a two item tuple containing the populated namespace and the
        list of remaining argument strings.

    subflowchart : seamm.Flowchart
        A SEAMM Flowchart object that represents a subflowchart, if needed.

    parameters : InteractionEdaParameters
        The control parameters for Interaction EDA.

    See Also
    --------
    TkInteractionEda,
    InteractionEda, InteractionEdaParameters
    """

    def __init__(
        self, flowchart=None, title="Interaction EDA", extension=None, logger=logger
    ):
        """A step for Interaction EDA in a SEAMM flowchart.

        You may wish to change the title above, which is the string displayed
        in the box representing the step in the flowchart.

        Parameters
        ----------
        flowchart: seamm.Flowchart
            The non-graphical flowchart that contains this step.

        title: str
            The name displayed in the flowchart.
        extension: None
            Not yet implemented
        logger : Logger = logger
            The logger to use and pass to parent classes

        Returns
        -------
        None
        """
        logger.debug(f"Creating Interaction EDA {self}")

        super().__init__(
            flowchart=flowchart,
            title="Interaction EDA",
            extension=extension,
            module=__name__,
            logger=logger,
        )  # yapf: disable

        self._metadata = interaction_eda_step.metadata
        self.parameters = interaction_eda_step.InteractionEdaParameters()

    @property
    def version(self):
        """The semantic version of this module."""
        return interaction_eda_step.__version__

    @property
    def git_revision(self):
        """The git version of this module."""
        return interaction_eda_step.__git_revision__

    def description_text(self, P=None):
        """Create the text description of what this step will do.

        Parameters
        ----------
        P: dict
            An optional dictionary of the current values of the control
            parameters.
        Returns
        -------
        str
            A description of the current step.
        """
        if not P:
            P = self.parameters.values_to_dict()

        backend = BACKENDS.get(P["backend"])
        scheme = backend.scheme if backend else "EDA"
        text = f"Energy decomposition analysis ({scheme}) of the interactions between "
        if P["fragments"] == "molecules":
            text += "the molecules of the structure"
        else:
            text += f"the fragments given by the atom column '{P['fragment column']}'"
        text += f", with {P['functional']}/{P['basis']} in {P['backend']}"
        options = []
        if P["density fitting"] != "no":
            options.append(f"density fitting with {P['density fitting']}")
        applies = self.parameters.applicable(P)
        if applies["vv10"] and P["vv10"].startswith("off"):
            options.append("no VV10")
        elif applies["vv10 in response"] and P["vv10 in response"] == "off":
            options.append("no VV10 in the FERF response")
        elif P["backend"] == "Psi4" and str(P["functional"]).lower().endswith("-v"):
            options.append("no VV10 in the FERF response")
        if applies["dispersion"] and P["dispersion"] not in ("none", ""):
            options.append(f"{P['dispersion']} dispersion")
        if applies["integration grid"] and P["integration grid"] != "fine":
            options.append(f"{P['integration grid']} integration grids")
        if P["ferf response"] != "coupled":
            options.append(f"{P['ferf response']} FERF response")
        if options:
            text += " (" + ", ".join(options) + ")"
        text += f". The calculations are the {P['calculations']}"
        if "triples" in P["calculations"] and P["triples cutoff"] != "all":
            text += f", using triples within {P['triples cutoff']}"
        text += "."
        if (
            P.get("source systems", "current"),
            P.get("source configurations", "current"),
        ) != (
            "current",
            "current",
        ):
            text += " " + seamm.standard_parameters.structure_selection_description(P)
        if applies.get("cores per calculation") and (
            str(P.get("cores per calculation")) != "automatic"
            or str(P.get("memory per calculation")) != "automatic"
        ):
            text += (
                f" Each calculation has {P['cores per calculation']} cores and "
                f"{P['memory per calculation']} memory."
            )

        return self.header + "\n" + __(text, indent=4 * " ").__str__()

    def backend_settings(self, P):
        """The backend's settings from the parameters.

        Settings that do not apply to the chosen backend (see the parameters'
        rules) are left out, so the backend's own defaults hold, e.g. no VV10 in
        Psi4's response kernel.
        """
        applies = self.parameters.applicable(P)
        auxbasis = P["density fitting"]
        settings = {
            "xc": str(P["functional"]).lower(),
            "basis": str(P["basis"]).lower(),
            "auxbasis": None if auxbasis in ("no", "none", "") else auxbasis,
            "ferf": P["ferf response"],
            "svd_threshold": float(P["svd threshold"]),
        }
        vv10_off = applies["vv10"] and P["vv10"].startswith("off")
        if applies["vv10"]:
            settings["nlc"] = False if vv10_off else None
        if applies["vv10 in response"] or vv10_off:
            settings["response_nlc"] = P["vv10 in response"] != "off" or vv10_off
        if applies["dispersion"]:
            dispersion = str(P["dispersion"]).strip()
            settings["disp"] = (
                None if dispersion.lower() in ("none", "no", "") else dispersion
            )
        if applies["integration grid"]:
            from .backends.gpu4pyscf import GRIDS

            settings.update(GRIDS[P["integration grid"]])
        if applies["low fp64 gpus"]:
            settings["allow_low_fp64"] = P["low fp64 gpus"] == "allow"
        return settings

    def run(self):
        """Run the Interaction EDA step.

        Every calculation -- each set of fragments of each selected structure --
        is a ``seamm_exec`` task, sized by its number of basis functions. The
        tasks run together: in the job's own cores, or bundled on the job's queue,
        and a rerun keeps those already finished.

        Returns
        -------
        seamm.Node
            The next node object in the flowchart.
        """
        next_node = super().run(printer)
        P = self.parameters.current_values_to_dict(
            context=seamm.flowchart_variables._data
        )
        printer.important(__(self.description_text(P), indent=self.indent))

        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)

        backend = BACKENDS[P["backend"]]()
        scheme = SCHEMES[backend.scheme]
        settings = backend.settings(self.backend_settings(P))
        problems = backend.problems(settings)
        if problems:
            raise RuntimeError(
                f"These settings cannot run with {backend.name}:\n  "
                + "\n  ".join(problems)
            )
        self.check_ini(backend)

        # The structures, their fragments and their calculations
        structures = [
            self.structure_calculations(i, configuration, P, backend)
            for i, configuration in enumerate(self.select_configurations(P))
        ]
        single = len(structures) == 1
        indent = self.indent + 4 * " "
        if single:
            self.print_fragments(structures[0]["fragments"])
        warnings = {svd_warning(settings, s["fragments"]) for s in structures}
        for warning in sorted(w for w in warnings if w):
            self.logger.warning(warning)
            printer.normal(__("Warning: " + warning + "\n", indent=indent))
        n_calculations = sum(len(s["calculations"]) for s in structures)
        sizes = {}
        for s in structures:
            for calc in s["calculations"]:
                sizes[len(calc.key)] = sizes.get(len(calc.key), 0) + 1
        text = f"There are {n_calculations} EDA calculations"
        if not single:
            text += f" for {len(structures)} structures"
        text += ": " + ", ".join(
            f"{n} of {m} fragments" for m, n in sorted(sizes.items())
        )
        printer.normal(__(text + ".\n", indent=indent))

        # One task per calculation
        cores = P.get("cores per calculation", "automatic")
        memory = P.get("memory per calculation", "automatic")
        if str(memory) != "automatic":
            memory = Q_(memory) if not hasattr(memory, "m_as") else memory
            memory = int(memory.m_as("byte"))
        task_set = seamm_exec.TaskSet(self, directory=directory)
        owners = {}
        for s in structures:
            for calc in s["calculations"]:
                key = calc.id if single else f"{s['index']}.{calc.id}"
                resources, seconds = backend.task_resources(
                    calc, settings, cores=cores, memory=memory
                )
                files = backend.files([calc], settings)
                task_set.add(
                    seamm_exec.Task(
                        key=key,
                        program=backend.program,
                        cmd=backend.command(),
                        files=files,
                        # Only what determines the results: a change to the
                        # driver that cannot change them keeps finished work
                        fingerprint=backend.fingerprint(files["input.json"]),
                        return_files=[backend.results_file, "*.log"],
                        resources=resources,
                        estimated_seconds=seconds,
                        estimate_source="formula" if seconds else "none",
                        shell=True,
                    )
                )
                owners[key] = (s, calc)

        t0 = time.time()
        refused = []
        for result in task_set.run():
            s, calc = owners[result.key]
            outcome = None
            if result.directory is not None:
                outcome = backend.read_results(result.directory, [calc])[calc.key]
            if outcome is None or outcome.energies is None:
                stderr = result.stderr or ""
                refused += [
                    line[len("REFUSED: ") :]
                    for line in stderr.splitlines()
                    if line.startswith("REFUSED: ")
                ]
                no_result = outcome is None or (outcome.error or "").startswith(
                    "no result"
                )
                if no_result:
                    tail = stderr.strip().splitlines()[-3:]
                    reason = result.reason or (" ".join(tail) if tail else "failed")
                    outcome = Result(calc.key, error=reason)
            s["results"][calc.key] = outcome
        wall = time.time() - t0
        printer.normal(
            __(
                f"The {n_calculations} calculations took {wall:.1f} s.\n\n",
                indent=indent,
            )
        )
        if refused:
            raise RuntimeError(refused[-1])

        # Each structure: the many-body analysis, the output and the results
        n_failed = 0
        all_failed = True
        for path in (directory / "eda.extxyz", directory / "eda.jsonl"):
            if path.exists() and not single:
                path.unlink()
        for s in structures:
            failed = self.finish_structure(s, backend, scheme, settings, P, single)
            n_failed += failed
            if failed < len(s["calculations"]):
                all_failed = False
        if all_failed:
            raise RuntimeError("All of the EDA calculations failed.")

        for alias, bibtex, level, note in backend.citations(settings):
            self.references.cite(
                raw=bibtex,
                alias=alias,
                module="interaction_eda_step",
                level=level,
                note=note,
            )
        if n_failed:
            printer.normal(
                __(
                    f"{n_failed} of the {n_calculations} calculations failed; "
                    "rerun the step to retry them.",
                    indent=indent,
                )
            )
        return next_node

    def structure_calculations(self, index, configuration, P, backend):
        """The fragments and calculations of one structure."""
        column = P["fragment column"] if P["fragments"] == "atom column" else None
        fragments = F.from_configuration(configuration, column=column)
        n = len(fragments)
        sets = {
            "cluster and pairs": ("cluster", "pairs"),
            "cluster, pairs and triples": ("cluster", "pairs", "triples"),
            "cluster": ("cluster",),
            "pairs": ("pairs",),
        }[P["calculations"]]
        triples = None
        if "triples" in sets:
            cutoff = P["triples cutoff"]
            if cutoff != "all":
                cutoff = cutoff.m_as("Å") if hasattr(cutoff, "m_as") else float(cutoff)
                triples = F.triples_within(fragments, cutoff)
        keys = manybody.fragment_sets(n, sets, triples)
        if backend.max_fragments is not None:
            too_big = [k for k in keys if len(k) > backend.max_fragments]
            if too_big:
                raise RuntimeError(
                    f"{backend.name} handles at most {backend.max_fragments} "
                    f"fragments per calculation; choose 'pairs'."
                )
        system = configuration.system
        return {
            "index": index,
            "configuration": configuration,
            "structure": {
                "system_name": system.name,
                "configuration_name": configuration.name,
            },
            "fragments": fragments,
            "calculations": [Calculation(k, [fragments[i] for i in k]) for k in keys],
            "results": {},
        }

    def finish_structure(self, s, backend, scheme, settings, P, single):
        """Analyze, print, write and store one structure's results; returns the
        number of its calculations that failed."""
        indent = self.indent + 4 * " "
        directory = Path(self.directory)
        fragments = s["fragments"]
        results = s["results"]
        name = s["structure"]["configuration_name"] or s["structure"]["system_name"]
        failed = {k: r for k, r in results.items() if r.energies is None}
        for key, r in failed.items():
            last = (r.error or "").strip().splitlines()[-1:] or ["?"]
            printer.normal(
                __(
                    f"{'' if single else name + ': '}the calculation for fragments "
                    f"{'+'.join(map(str, key))} failed: {last[0]}",
                    indent=indent,
                )
            )
        energies = {k: r.energies for k, r in results.items() if r.energies}
        for key, values in energies.items():
            for term, value, total in scheme.check(values, tolerance=1.0e-4):
                self.logger.warning(
                    f"EDA terms of {key} are inconsistent: {term} = {value:.6f} "
                    f"but its parts sum to {total:.6f} kJ/mol"
                )
        if not energies:
            return len(failed)

        mb = manybody.analyze(energies, len(fragments), scheme.terms)
        method = backend.method_string(settings)
        if single:
            self.print_results(mb, scheme, fragments)
        else:
            values = mb.cluster if mb.cluster is not None else None
            if values is None and len(mb.pairs) == 1:
                values = next(iter(mb.pairs.values()))
            if values is not None:
                common = scheme.to_common(values)
                printer.normal(
                    __(
                        f"{name}: total {common.get('total', values.get('total')):.3f}"
                        " kJ/mol",
                        indent=indent,
                    )
                )
        self.write_extxyz(
            directory / "eda.extxyz",
            fragments,
            mb,
            scheme,
            method,
            results,
            structure=s["structure"],
            append=not single,
        )
        record = self.json_record(
            fragments, results, mb, scheme, method, settings, s["structure"]
        )
        if single:
            (directory / "eda.json").write_text(json.dumps(record, indent=2))
        else:
            with open(directory / "eda.jsonl", "a") as fd:
                fd.write(json.dumps(record) + "\n")

        # Results
        n = len(fragments)
        self.model = f"{backend.scheme}/{P['functional']}/{P['basis']}"
        data = {"EDA method": method, "number of fragments": n}
        parts = {
            "": mb.cluster,
            " 2-body": mb.sum_pairs,
            " ≥3-body": mb.ge3_exact,
            " 3-body (triples)": mb.sum_three_body,
        }
        for suffix, values in parts.items():
            if values is None:
                continue
            for term, value in scheme.to_common(values).items():
                data[f"EDA {term}{suffix}"] = value
        if mb.pairs:
            data["EDA pairs"] = {
                "+".join(map(str, k)): v for k, v in sorted(mb.pairs.items())
            }
        if mb.three_body:
            data["EDA triples"] = {
                "+".join(map(str, k)): v for k, v in sorted(mb.three_body.items())
            }
        self.store_results(configuration=s["configuration"], data=data)
        return len(failed)

    def check_ini(self, backend):
        """Make sure the backend's ini file exists here, writing the default from
        the package's data if there is one."""
        ini_dir = Path(self.global_options["root"]).expanduser()
        ini_path = ini_dir / backend.ini
        if ini_path.exists():
            return
        resources = importlib.resources.files("interaction_eda_step") / "data"
        if not (resources / backend.ini).is_file():
            raise RuntimeError(
                f"{backend.name} needs {ini_path}, which its own SEAMM step "
                f"installs: install the {backend.name} step "
                f"({backend.name.lower()}-step) and its code."
            )
        ini_path.write_text((resources / backend.ini).read_text())
        self.logger.info(f"Wrote a default {ini_path}")

    def n_cores(self, available):
        """The cores to use: those available, capped by SEAMM's ``ncores`` option.

        ``ncores`` in the [DEFAULT] section of seamm.ini (or ``--ncores``) limits
        every step, e.g. to a laptop's performance cores.
        """
        n = int(available)
        limit = str(self.global_options.get("ncores", "available")).strip()
        if limit.lower() != "available":
            n = min(n, int(limit))
        return max(n, 1)

    def print_fragments(self, fragments):
        rows = [
            [f.index, f.name, len(f.atoms), f.charge, f.multiplicity] for f in fragments
        ]
        text = tabulate(
            rows,
            headers=["#", "Fragment", "Atoms", "Charge", "Mult."],
            tablefmt="rounded_outline",
        )
        printer.normal(__("The fragments:\n", indent=self.indent + 4 * " "))
        printer.normal(__(text, indent=self.indent + 8 * " ", wrap=False, dedent=False))
        printer.normal("")

    def print_results(self, mb, scheme, fragments):
        """Print the many-body table and the pairs, by term."""
        indent = self.indent + 4 * " "
        terms = list(scheme.terms) + [t for t in scheme.common if t not in scheme.terms]
        labels = {t: t for t in scheme.terms}
        labels.update(
            {t: f"{t} (common)" for t in scheme.common if t not in scheme.terms}
        )

        def values(d):
            if d is None:
                return None
            out = dict(d)
            out.update({k: v for k, v in scheme.to_common(d).items() if k not in out})
            return out

        columns = [("Cluster", values(mb.cluster)), ("2-body", values(mb.sum_pairs))]
        columns.append(("≥3-body (exact)", values(mb.ge3_exact)))
        if mb.sum_three_body is not None:
            columns.append(
                (
                    f"3-body ({len(mb.three_body)} of {mb.n_triples_total} triples)",
                    values(mb.sum_three_body),
                )
            )
        columns = [(name, d) for name, d in columns if d is not None]
        rows = [[labels[t]] + [d.get(t) for _, d in columns] for t in terms]
        text = tabulate(
            rows,
            headers=["Term (kJ/mol)"] + [name for name, _ in columns],
            tablefmt="rounded_outline",
            floatfmt=".3f",
        )
        printer.normal(__(f"{scheme.name} interaction energies:\n", indent=indent))
        printer.normal(__(text, indent=indent + 4 * " ", wrap=False, dedent=False))
        printer.normal("")

        if mb.pairs:
            # The common (SAPT-comparable) terms
            short = {
                "electrostatics": "elst",
                "Pauli": "Pauli",
                "dispersion": "disp",
                "polarization": "pol",
                "charge transfer": "CT",
                "total": "total",
            }
            rows = []
            for (i, j), d in sorted(mb.pairs.items()):
                common = scheme.to_common(d)
                short = {t: h for t, h in short.items() if t in common}
                rows.append(
                    [f"{i}+{j}", f"{fragments[i].name}-{fragments[j].name}"]
                    + [common[t] for t in short]
                )
            text = tabulate(
                rows,
                headers=["Pair", "Fragments"] + list(short.values()),
                tablefmt="rounded_outline",
                floatfmt=".3f",
            )
            printer.normal(
                __(
                    "The pairs (common terms, kJ/mol; elst is the classical "
                    "electrostatics):\n",
                    indent=indent,
                )
            )
            printer.normal(__(text, indent=indent + 4 * " ", wrap=False, dedent=False))
            printer.normal("")

        for note in scheme.notes:
            printer.normal(__(note, indent=indent))
        printer.normal(
            __(
                "≥3-body (exact) = cluster - sum of all pairs. The 3-body column, if "
                "present, sums the 3-body increments of the triples computed: it is "
                "truncated and is not the ≥3-body total.\n",
                indent=indent,
            )
        )

    def write_extxyz(
        self,
        path,
        fragments,
        mb,
        scheme,
        method,
        results=None,
        structure=None,
        append=False,
    ):
        """One frame per computed set of fragments, with the EDA terms as
        labels ``EDA_<scheme>_<term>`` (kJ/mol), plus ``EDA_<scheme>_mod_pauli``,
        the SAPT-comparable Pauli; the cluster's frame also has the many-body
        parts, and each frame says where it was computed (``EDA_platform``) and
        which structure it came from (``system_name``, ``configuration_name``)."""
        tag = "EDA_" + scheme.name.replace("-", "")

        def key(term):
            return f"{tag}_{term.replace(' ', '_')}"

        def labels(values, suffix=""):
            out = {}
            if values is None:
                return out
            full = dict(values)
            full.update(scheme.to_common(values))
            for term in list(scheme.terms) + ["induction"]:
                if term in full:
                    out[key(term) + suffix] = full[term]
            # The SAPT-comparable Pauli (Q-Chem MOD PAULI = frozen - classical
            # electrostatic - dispersion); 'pauli' is the orthogonal one
            if "Pauli" in full:
                out[key("mod pauli") + suffix] = full["Pauli"]
            return out

        conventions = (
            "kJ/mol; fragments at the cluster geometry (no distortion term); "
            "charge transfer carries the BSSE; induction = polarization + charge "
            "transfer; classical_electrostatic (Q-Chem CLS ELEC) is SAPT-comparable "
            "electrostatics, electrostatic and pauli (Q-Chem ELEC, PAULI; Horn, Mao "
            "and Head-Gordon 2016) are the orthogonal decomposition and are not "
            "SAPT-comparable; mod_pauli = frozen - classical_electrostatic - "
            "dispersion (Q-Chem MOD PAULI) is the SAPT-comparable Pauli; frozen = "
            "elst + Pauli + disp in both sets; ge3body_exact = cluster - sum of "
            "pairs; 3body_triples is truncated to the triples computed"
        )
        frames = []
        sets = []
        if mb.cluster is not None:
            sets.append((tuple(range(len(fragments))), mb.cluster))
        sets += sorted(mb.pairs.items())
        sets += sorted(mb.triples.items())
        seen = set()
        for k, values in sets:
            if k in seen:
                continue
            seen.add(k)
            info = {"fragments": "+".join(map(str, k))}
            for name, value in (structure or {}).items():
                if value:
                    info[name] = value
            info.update(labels(values))
            if len(k) == len(fragments):
                info.update(labels(mb.sum_pairs, "_2body"))
                info.update(labels(mb.ge3_exact, "_ge3body_exact"))
                info.update(labels(mb.sum_three_body, "_3body_triples"))
            if len(k) == 3 and k in mb.three_body:
                info.update(labels(mb.three_body[k], "_3body"))
            info["EDA_method"] = method
            info["EDA_conventions"] = conventions
            if results is not None and k in results:
                platform = platform_text(results[k].data)
                if platform:
                    info["EDA_platform"] = platform
            atoms = []
            for i in k:
                f = fragments[i]
                for s, xyz in zip(f.symbols, f.coordinates):
                    atoms.append((s, xyz, i, f.charge))
            frames.append(_extxyz_frame(atoms, info))
        if append:
            with open(path, "a") as fd:
                fd.write("".join(frames))
        else:
            path.write_text("".join(frames))

    def json_record(
        self, fragments, results, mb, scheme, method, settings, structure=None
    ):
        """The eda.json record of one structure."""
        data = {
            "structure": dict(structure or {}),
            "method": method,
            "scheme": scheme.name,
            "settings": settings,
            "notes": list(scheme.notes),
            "fragments": [
                {
                    "index": f.index,
                    "name": f.name,
                    "atoms": f.atoms,
                    "charge": f.charge,
                    "multiplicity": f.multiplicity,
                }
                for f in fragments
            ],
            "calculations": {
                "+".join(map(str, k)): {
                    "energies_kJ_mol": r.energies,
                    "error": r.error,
                    "wall_s": r.data.get("wall_s"),
                    "peak_gpu_mem_GB": r.data.get("peak_gpu_mem_GB"),
                    "gpu": r.data.get("gpu"),
                    "versions": r.data.get("versions"),
                }
                for k, r in sorted(results.items())
            },
            "cluster": mb.cluster,
            "sum_pairs": mb.sum_pairs,
            "ge3body_exact": mb.ge3_exact,
            "three_body": {
                "+".join(map(str, k)): v for k, v in sorted(mb.three_body.items())
            },
            "sum_3body_triples_truncated": mb.sum_three_body,
            "n_triples_total": mb.n_triples_total,
        }
        return data


def platform_text(data):
    """Where a calculation ran, e.g. 'GPU4PySCF 1.9.1, PySCF 2.15.0, CuPy 14.2.0 on
    NVIDIA A30 (sm 8.0, CUDA 12.9, driver 580.82.07)' or 'Psi4 1.11 on CPU, 8
    threads', from a backend's record; '' if it says nothing."""
    versions = data.get("versions") or {}
    gpu = data.get("gpu")
    codes = [
        f"{label} {versions[name]}"
        for name, label in (
            ("gpu4pyscf", "GPU4PySCF"),
            ("pyscf", "PySCF"),
            ("cupy", "CuPy"),
            ("psi4", "Psi4"),
        )
        if versions.get(name)
    ]
    if gpu:
        details = [
            f"{label} {versions[name]}"
            for name, label in (
                ("compute_capability", "sm"),
                ("cuda_driver", "CUDA"),
                ("nvidia_driver", "driver"),
            )
            if versions.get(name)
        ]
        where = f"on {gpu}" + (f" ({', '.join(details)})" if details else "")
    elif versions.get("threads"):
        where = f"on CPU, {versions['threads']} threads"
    else:
        where = ""
    return " ".join(part for part in (", ".join(codes), where) if part)


def _extxyz_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value)
    if any(c in text for c in ' ="'):
        return '"' + text.replace('"', "'") + '"'
    return text


def _extxyz_frame(atoms, info):
    """An extended-XYZ frame: species, positions (Å), fragment and charge."""
    comment = " ".join(f"{k}={_extxyz_value(v)}" for k, v in info.items())
    comment += ' Properties=species:S:1:pos:R:3:frag:I:1:frag_charge:I:1 pbc="F F F"'
    lines = [str(len(atoms)), comment]
    for s, xyz, frag, charge in atoms:
        lines.append(
            f"{s:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f} {frag:3d} {charge:3d}"
        )
    return "\n".join(lines) + "\n"
