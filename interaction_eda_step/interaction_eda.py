# -*- coding: utf-8 -*-

"""Non-graphical part of the Interaction EDA step in a SEAMM flowchart"""

import configparser
import importlib.resources
import json
import logging
from pathlib import Path
import pprint  # noqa: F401
import shutil
import time

from tabulate import tabulate

import interaction_eda_step
from . import fragments as F
from . import manybody
from .backends import BACKENDS, Calculation
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
        if P["vv10"].startswith("off"):
            options.append("no VV10")
        elif P["vv10 in response"] == "off":
            options.append("no VV10 in the FERF response")
        if P["dispersion"] not in ("none", ""):
            options.append(f"{P['dispersion']} dispersion")
        if P["ferf response"] != "coupled":
            options.append(f"{P['ferf response']} FERF response")
        if options:
            text += " (" + ", ".join(options) + ")"
        text += f". The calculations are the {P['calculations']}"
        if "triples" in P["calculations"] and P["triples cutoff"] != "all":
            text += f", using triples within {P['triples cutoff']}"
        text += "."

        return self.header + "\n" + __(text, indent=4 * " ").__str__()

    def backend_settings(self, P):
        """The backend's settings from the parameters."""
        auxbasis = P["density fitting"]
        dispersion = str(P["dispersion"]).strip()
        return {
            "xc": str(P["functional"]).lower(),
            "basis": str(P["basis"]).lower(),
            "auxbasis": None if auxbasis in ("no", "none", "") else auxbasis,
            "nlc": False if P["vv10"].startswith("off") else None,
            "response_nlc": (
                P["vv10 in response"] != "off" or P["vv10"].startswith("off")
            ),
            "disp": None if dispersion.lower() in ("none", "no", "") else dispersion,
            "ferf": P["ferf response"],
            "svd_threshold": float(P["svd threshold"]),
            "allow_low_fp64": P["low fp64 gpus"] == "allow",
        }

    def run(self):
        """Run the Interaction EDA step.

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

        _, configuration = self.get_system_configuration(None)

        backend = BACKENDS[P["backend"]]()
        scheme = SCHEMES[backend.scheme]
        settings = backend.settings(self.backend_settings(P))
        problems = backend.problems(settings)
        if problems:
            raise RuntimeError(
                f"These settings cannot run with {backend.name}:\n  "
                + "\n  ".join(problems)
            )

        # The fragments and the calculations
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
        calculations = [Calculation(k, [fragments[i] for i in k]) for k in keys]

        self.print_fragments(fragments)
        warning = svd_warning(settings, fragments)
        if warning:
            self.logger.warning(warning)
            printer.normal(
                __("Warning: " + warning + "\n", indent=self.indent + 4 * " ")
            )
        printer.normal(
            __(
                f"There are {len(calculations)} EDA calculations: "
                + ", ".join(
                    f"{sum(1 for k in keys if len(k) == m)} of {m} fragments"
                    for m in sorted({len(k) for k in keys})
                )
                + ".\n",
                indent=self.indent + 4 * " ",
            )
        )

        # Run what has not been done (a rerun skips finished calculations)
        results = backend.read_results(directory, calculations)
        pending = [c for c in calculations if results[c.key].energies is None]
        if pending:
            t0 = time.time()
            self.run_backend(backend, calculations, settings)
            wall = time.time() - t0
            results = backend.read_results(directory, calculations)
            printer.normal(
                __(
                    f"The {len(pending)} calculations took {wall:.1f} s.\n\n",
                    indent=self.indent + 4 * " ",
                )
            )

        failed = {k: r for k, r in results.items() if r.energies is None}
        for key, r in failed.items():
            last = (r.error or "").strip().splitlines()[-1:] or ["?"]
            printer.normal(
                __(
                    f"The calculation for fragments {'+'.join(map(str, key))} "
                    f"failed: {last[0]}",
                    indent=self.indent + 4 * " ",
                )
            )
        energies = {k: r.energies for k, r in results.items() if r.energies}
        for key, values in energies.items():
            for name, value, total in scheme.check(values, tolerance=1.0e-4):
                self.logger.warning(
                    f"EDA terms of {key} are inconsistent: {name} = {value:.6f} "
                    f"but its parts sum to {total:.6f} kJ/mol"
                )
        if not energies:
            message = "All of the EDA calculations failed."
            stderr = Path(self.directory) / "stderr.txt"
            refused = [
                line[len("REFUSED: ") :]
                for line in (stderr.read_text() if stderr.exists() else "").splitlines()
                if line.startswith("REFUSED: ")
            ]
            if refused:
                raise RuntimeError(refused[-1])
            if stderr.exists():
                tail = stderr.read_text().strip().splitlines()[-5:]
                if tail:
                    message += " The end of the code's error output:\n" + "\n".join(
                        tail
                    )
            raise RuntimeError(message)

        mb = manybody.analyze(energies, n, scheme.terms)
        method = backend.method_string(settings)
        self.print_results(mb, scheme, fragments)
        self.write_extxyz(directory / "eda.extxyz", fragments, mb, scheme, method)
        self.write_json(
            directory / "eda.json", fragments, results, mb, scheme, method, settings
        )

        # Results
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

        for alias, bibtex, level, note in backend.citations(settings):
            self.references.cite(
                raw=bibtex,
                alias=alias,
                module="interaction_eda_step",
                level=level,
                note=note,
            )

        self.store_results(configuration=configuration, data=data)

        if failed:
            printer.normal(
                __(
                    f"{len(failed)} of the {len(calculations)} calculations failed; "
                    "rerun the step to retry them.",
                    indent=self.indent + 4 * " ",
                )
            )
        return next_node

    def run_backend(self, backend, calculations, settings):
        """Run the backend through the executor, with its ini file."""
        executor = self.flowchart.executor
        executor_type = executor.name
        ini_dir = Path(self.global_options["root"]).expanduser()
        ini_path = ini_dir / backend.ini
        if not ini_path.exists():
            resources = importlib.resources.files("interaction_eda_step") / "data"
            ini_path.write_text((resources / backend.ini).read_text())
            self.logger.info(f"Wrote a default {ini_path}")
        full_config = configparser.ConfigParser()
        full_config.read(ini_path)
        if executor_type not in full_config:
            raise RuntimeError(
                f"No section for '{executor_type}' in {ini_path}. Add one giving "
                f"the Python of the {backend.name} environment, e.g.\n"
                f"[{executor_type}]\ninstallation = local\n"
                "code = /path/to/env/bin/python"
            )
        config = backend.configure(dict(full_config.items(executor_type)))
        code = config.get("code", "")
        if config.get("installation", "local") == "local" and not (
            Path(code.split()[0]).expanduser().exists() if code else False
        ):
            found = shutil.which(code) if code else None
            if found is None:
                raise RuntimeError(
                    f"The {backend.name} Python '{code}' in {ini_path} "
                    f"[{executor_type}] does not exist. Point 'code' at the Python "
                    f"of the environment where {backend.name} is installed."
                )

        files = backend.files(calculations, settings)
        ce = seamm_exec.computational_environment()
        result = executor.run(
            cmd=backend.command(),
            config=config,
            directory=self.directory,
            files=files,
            return_files=[backend.results_file, "*.log"],
            in_situ=True,
            shell=True,
            ce=ce,
        )
        if not result:
            self.logger.error(f"There was an error running {backend.name}")
        else:
            for name in ("stdout", "stderr"):
                text = result.get(name)
                if text:
                    (Path(self.directory) / f"{name}.txt").write_text(text)
        return result

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

    def write_extxyz(self, path, fragments, mb, scheme, method):
        """One frame per computed set of fragments, with the EDA terms as
        labels ``EDA_<scheme>_<term>`` (kJ/mol); the cluster's frame also has
        the many-body parts."""
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
            return out

        conventions = (
            "kJ/mol; fragments at the cluster geometry (no distortion term); "
            "charge transfer carries the BSSE; induction = polarization + charge "
            "transfer; classical_electrostatic (Q-Chem CLS ELEC) is SAPT-comparable "
            "electrostatics, electrostatic and pauli (Q-Chem ELEC, PAULI; Horn, Mao "
            "and Head-Gordon 2016) are the orthogonal decomposition and are not "
            "SAPT-comparable; SAPT-comparable Pauli = frozen - "
            "classical_electrostatic - dispersion (Q-Chem MOD PAULI); frozen = "
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
            info.update(labels(values))
            if len(k) == len(fragments):
                info.update(labels(mb.sum_pairs, "_2body"))
                info.update(labels(mb.ge3_exact, "_ge3body_exact"))
                info.update(labels(mb.sum_three_body, "_3body_triples"))
            if len(k) == 3 and k in mb.three_body:
                info.update(labels(mb.three_body[k], "_3body"))
            info["EDA_method"] = method
            info["EDA_conventions"] = conventions
            atoms = []
            for i in k:
                f = fragments[i]
                for s, xyz in zip(f.symbols, f.coordinates):
                    atoms.append((s, xyz, i, f.charge))
            frames.append(_extxyz_frame(atoms, info))
        path.write_text("".join(frames))

    def write_json(self, path, fragments, results, mb, scheme, method, settings):
        data = {
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
        path.write_text(json.dumps(data, indent=2))


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
