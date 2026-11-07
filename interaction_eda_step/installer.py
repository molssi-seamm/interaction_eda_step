# -*- coding: utf-8 -*-

"""Installer for the Interaction EDA plug-in.

The GPU4PySCF backend runs in its own conda environment, ``seamm-gpu4pyscf`` by
default, which this installer creates and registers in ``gpu4pyscf.ini``. The
environment holds Python 3.12 from conda-forge and, from PyPI, GPU4PySCF with
CuPy, cuTENSOR and the CUDA libraries for the machine's NVIDIA driver, plus
pyscf-dispersion for D3/D4. It needs Linux and an NVIDIA GPU with good double
precision; on macOS nothing is installed.
"""

import importlib.resources
import importlib.util
import json
import logging
import platform

import seamm_manager
from seamm_manager.torch_support import detect_cuda_driver

logger = logging.getLogger(__name__)

#: The GPU4PySCF version the step was validated with; ALMO-EDA2 needs 1.9.
MINIMUM_GPU4PYSCF = "1.9.1"

#: The pip requirements for each CUDA build. cuda12x is the validated one
#: (ARC, 2026-11-07: gpu4pyscf 1.9.1, cupy 14.2, cuTENSOR 2.8, CUDA 12.9).
BUILDS = {
    "cuda12x": (
        f"gpu4pyscf-cuda12x>={MINIMUM_GPU4PYSCF}",
        "cupy-cuda12x[ctk]",
        "cutensor-cu12",
        "pyscf-dispersion",
    ),
    "cuda13x": (
        f"gpu4pyscf-cuda13x>={MINIMUM_GPU4PYSCF}",
        "cupy-cuda13x[ctk]",
        "cutensor-cu13",
        "pyscf-dispersion",
    ),
}

#: Links the unversioned libcusolver.so to the versioned library in the NVIDIA
#: wheels. GPU4PySCF loads it by the bare name, which the wheels do not provide
#: ("Unable to load libcusolver").
LINK_CUSOLVER = r"""
import glob, json, os, re, site
made = []
for sp in site.getsitepackages():
    for lib in glob.glob(os.path.join(sp, "nvidia", "**", "libcusolver.so.*"),
                         recursive=True):
        if not re.search(r"libcusolver\.so\.\d+$", lib):
            continue
        link = os.path.join(os.path.dirname(lib), "libcusolver.so")
        if not os.path.lexists(link):
            os.symlink(os.path.basename(lib), link)
            made.append(link)
print("LINKED" + json.dumps(made))
"""

#: Imports GPU4PySCF and, if there is a GPU, does a small calculation on it.
PROBE = r"""
import json
out = {}
try:
    import cupy
    out["cupy"] = cupy.__version__
    n = cupy.cuda.runtime.getDeviceCount()
    out["gpus"] = [
        cupy.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in range(n)
    ]
    a = cupy.arange(10, dtype=cupy.float64)
    out["gpu_ok"] = float((a * a).sum()) == 285.0
except Exception as e:
    out["gpu_error"] = f"{type(e).__name__}: {e}"
try:
    import pyscf
    import gpu4pyscf
    from gpu4pyscf.properties import eda
    out["gpu4pyscf"] = gpu4pyscf.__version__
    out["pyscf"] = pyscf.__version__
    out["almo_eda2"] = hasattr(eda, "eval_ALMO_EDA_2_energies")
except Exception as e:
    out["import_error"] = f"{type(e).__name__}: {e}"
print("PROBE" + json.dumps(out))
"""


def choose_build(forced=None, system=None, driver="detect"):
    """Which CUDA build of GPU4PySCF to install.

    Parameters
    ----------
    forced : str, optional
        ``cuda12x`` or ``cuda13x`` from ``gpu4pyscf.ini`` or the command line;
        ``auto``, ``""`` or None to decide from the driver.
    system : str, optional
        ``platform.system()``; for tests.
    driver : (int, int) or None or "detect"
        The driver's CUDA version; ``"detect"`` runs nvidia-smi.

    Returns
    -------
    (str or None, str)
        The build, or None if GPU4PySCF cannot be installed here, and why.
    """
    system = system or platform.system()
    if system != "Linux":
        return None, (
            f"GPU4PySCF needs Linux and an NVIDIA GPU, so it is not installed on "
            f"{system}. Flowcharts with the Interaction EDA step can still be built "
            "here and run on a GPU machine."
        )
    forced = (forced or "").strip().lower()
    if forced and forced != "auto":
        if forced not in BUILDS:
            return None, (
                f"Unknown cuda-build '{forced}'; use auto or one of "
                f"{', '.join(BUILDS)}."
            )
        return forced, f"forced to {forced}"
    detected = detect_cuda_driver() if driver == "detect" else driver
    if detected is None:
        return "cuda12x", (
            "no NVIDIA driver found here (e.g. a cluster login node), so the CUDA 12 "
            "build, which runs with any driver for CUDA 12 or later"
        )
    if detected[0] < 12:
        return None, (
            f"the NVIDIA driver supports CUDA {detected[0]}.{detected[1]}, but "
            "GPU4PySCF needs a driver for CUDA 12 or later. Update the driver."
        )
    # The CUDA 12 build is the validated one and runs on CUDA 13 drivers too.
    return "cuda12x", f"driver CUDA {detected[0]}.{detected[1]}"


def parse(output, tag):
    """The JSON (or repr) after ``tag`` in a script's output, or None."""
    if not output:
        return None
    for line in output.splitlines():
        if line.startswith(tag):
            text = line[len(tag) :]
            try:
                return json.loads(text)
            except ValueError:
                return text
    return None


def low_fp64(name):
    """Whether a GPU has poor double precision, using the driver's own test."""
    path = importlib.resources.files("interaction_eda_step") / "data"
    spec = importlib.util.spec_from_file_location(
        "gpu4pyscf_eda", str(path / "gpu4pyscf_eda.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # imports nothing from GPU4PySCF at load
    return module.low_fp64(name)


def assess(probe, have_driver=True):
    """Lines describing a probe of the environment, and whether it is usable.

    Parameters
    ----------
    probe : dict or None
        From :data:`PROBE`.
    have_driver : bool
        Whether this machine has an NVIDIA driver. Without one, failing to import
        CuPy or GPU4PySCF is expected and is not an error.

    Returns
    -------
    (bool, [str])
    """
    if not isinstance(probe, dict):
        return False, ["!   Could not run Python in the environment."]
    lines = []
    ok = True
    if "import_error" in probe:
        if have_driver:
            ok = False
            lines.append(f"!   GPU4PySCF does not import: {probe['import_error']}")
        else:
            lines.append(
                "    GPU4PySCF cannot be imported without a GPU, as expected here. "
                "Check it on a GPU node with 'interaction-eda-step-installer check'."
            )
    else:
        lines.append(
            f"    GPU4PySCF {probe['gpu4pyscf']}, PySCF {probe['pyscf']}, CuPy "
            f"{probe.get('cupy', '?')}."
        )
        if not probe.get("almo_eda2"):
            ok = False
            lines.append(
                f"!   This GPU4PySCF has no ALMO-EDA2: {MINIMUM_GPU4PYSCF} or later "
                "is needed."
            )
    gpus = probe.get("gpus") or []
    if gpus:
        if probe.get("gpu_ok"):
            lines.append(f"    The GPU works: {', '.join(gpus)}.")
        else:
            ok = False
            lines.append(f"!   A test calculation on the GPU ({gpus[0]}) failed.")
        slow = sorted({g for g in gpus if low_fp64(g)})
        if slow:
            lines.append(
                f"!   {', '.join(slow)} has poor double precision: the EDA will run, "
                "but very slowly. Use an A100, A30, H100/H200 or B200 if possible."
            )
    elif have_driver:
        ok = False
        lines.append(f"!   CuPy cannot use the GPU: {probe.get('gpu_error', '?')}")
    return ok, lines


class Installer(seamm_manager.InstallerBase):
    """Handle the installation of GPU4PySCF for the Interaction EDA step.

    #. ``install`` creates the conda environment ``seamm-gpu4pyscf`` (Python
       3.12) and installs GPU4PySCF in it for the machine's NVIDIA driver, then
       checks that it imports and can use the GPU, and writes ``gpu4pyscf.ini``.
       ``cuda-build`` in ``gpu4pyscf.ini`` (or ``--cuda-build``) forces a build,
       e.g. on a cluster login node without a GPU.
    #. ``update`` upgrades GPU4PySCF and its CUDA libraries in that environment.
    #. ``check`` and ``show`` say what is installed and whether it works.

    An environment named in ``gpu4pyscf.ini`` that SEAMM did not create, or an
    ``installation = local`` Python, is never changed.
    """

    def __init__(self, logger=logger):
        super().__init__(logger=logger)
        self.environment = "seamm-gpu4pyscf"
        self.section = "interaction-eda-step"
        self.init_file_name = "gpu4pyscf.ini"
        self.executables = ["python"]
        self.resource_path = importlib.resources.files("interaction_eda_step") / "data"
        self.environment_file = self.resource_path / "seamm-gpu4pyscf.yml"

    def setup_parser(self):
        parser = super().setup_parser()
        text = (
            "The CUDA build of GPU4PySCF: cuda12x or cuda13x, instead of the one "
            "chosen from this machine's NVIDIA driver."
        )
        for name in ("install", "update"):
            self.subparser[name].add_argument(
                "--cuda-build", metavar="BUILD", help=text
            )
        return parser

    # ------------------------------------------------------------------
    def _build(self):
        """The build to install and why, from the options, the ini, the driver."""
        forced = getattr(self.options, "cuda_build", None)
        if not forced and self.exe_config.path is not None:
            forced = self.exe_config.get_values("local").get("cuda-build")
        return choose_build(forced)

    def _finish(self, environment):
        """Link libcusolver, then check the environment. Returns True if usable."""
        linked = parse(self.conda.run_python(environment, LINK_CUSOLVER), "LINKED")
        if linked:
            print(f"    Linked {', '.join(linked)}.")
        ok, lines = assess(
            parse(self.conda.run_python(environment, PROBE), "PROBE"),
            have_driver=detect_cuda_driver() is not None,
        )
        for line in lines:
            print(line)
        return ok

    def _register(self, environment, build):
        self.check_exe_configuration_file()
        self.exe_config.set_value("local", "installation", "conda")
        conda_exe = seamm_manager.find_conda()
        if conda_exe is not None:
            self.exe_config.set_value("local", "conda", conda_exe)
        self.exe_config.set_value("local", "conda-environment", environment)
        self.exe_config.set_value("local", "code", "python")
        if build is not None:
            self.exe_config.set_value("local", "cuda-build", build)
        for key in ("modules", "container", "platform"):
            self.exe_config.set_value("local", key, None)
        self.exe_config.save()

    # ------------------------------------------------------------------
    def install(self):
        """Create the environment and install GPU4PySCF for this machine."""
        if self.shared_codes:
            self._use_shared_code()
            return
        if self.code_environments == "prefixed":
            from seamm_manager.naming import compute_tag
            from seamm_manager.policy import prefixed_environment

            self.environment = prefixed_environment(
                self.environment, compute_tag(self.root)
            )
        self.check_exe_configuration_file()
        build, reason = self._build()
        if build is None:
            print(f"    {reason}")
            return
        data = self.exe_config.get_values("local")
        if data.get("installation") == "local":
            print(
                f"    gpu4pyscf.ini uses the local Python '{data.get('code')}', "
                "which is left as it is."
            )
            return
        environment = data.get("conda-environment") or self.environment
        if self.conda.exists(environment):
            print(f"    Using the existing Conda environment '{environment}'.")
            if self._not_ours(environment):
                print(
                    f"    '{environment}' was not created by SEAMM, so nothing is "
                    "installed in it."
                )
                self._finish(environment)
                return
        else:
            print(
                f"    Installing Conda environment '{environment}' with GPU4PySCF "
                f"({build}: {reason}). This may take a few minutes."
            )
            self.conda.create_environment(
                self.environment_file, name=environment, conda_only=True
            )
        self.conda.pip_install(environment, BUILDS[build])
        self._register(environment, build)
        if self._finish(environment):
            print("    Done!\n")
        else:
            print(
                "!   The environment was made but does not work yet; see above. "
                "'interaction-eda-step-installer check' checks it again.\n"
            )

    def update(self):
        """Upgrade GPU4PySCF and its CUDA libraries in the environment."""
        if self.shared_codes:
            self._use_shared_code()
            return
        self.check_exe_configuration_file()
        data = self.exe_config.get_values("local")
        if data.get("installation") != "conda":
            print(
                "!   GPU4PySCF was not installed by SEAMM (installation is not "
                "'conda' in gpu4pyscf.ini), so it is not updated."
            )
            return
        environment = data.get("conda-environment") or self.environment
        if not self.conda.exists(environment):
            print(f"!   The Conda environment '{environment}' does not exist; install.")
            return
        if self._not_ours(environment):
            print(
                f"    The Conda environment '{environment}' was not created by SEAMM, "
                "so it is left as it is."
            )
            return
        build, reason = self._build()
        if build is None:
            print(f"    {reason}")
            return
        print(f"    Updating GPU4PySCF in '{environment}' ({build}: {reason}).")
        self.conda.pip_install(environment, BUILDS[build], upgrade=True)
        self._register(environment, build)
        if self._finish(environment):
            print("    Done!\n")

    def check(self):
        """Check that gpu4pyscf.ini points at a working GPU4PySCF."""
        if not self.configuration.section_exists(self.section):
            self.check_configuration_file()
        self.check_exe_configuration_file()
        data = self.exe_config.get_values("local")
        if data.get("installation") == "conda":
            environment = data.get("conda-environment") or self.environment
            if not self.conda.exists(environment):
                print(
                    f"!   The Conda environment '{environment}' in gpu4pyscf.ini does "
                    "not exist. Run 'interaction-eda-step-installer install'."
                )
                return False
            print(f"    The Conda environment is '{environment}'.")
            return self._finish(environment)
        print(
            f"    gpu4pyscf.ini runs '{data.get('code')}' "
            f"(installation = {data.get('installation')}), which is not checked."
        )
        return True

    def exe_version(self, config):
        """The version of GPU4PySCF in its environment."""
        environment = config.get("conda-environment") or self.environment
        output = self.conda.run_python(
            environment,
            "import importlib.metadata as m\n"
            "print('VERSION' + next((d.version for d in m.distributions() if "
            "d.metadata['Name'].lower().startswith('gpu4pyscf-cuda')), 'unknown'))",
        )
        version = parse(output, "VERSION")
        return "GPU4PySCF", version if isinstance(version, str) else "unknown"
