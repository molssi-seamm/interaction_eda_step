# -*- coding: utf-8 -*-
"""
Control parameters for the Interaction EDA step in a SEAMM flowchart
"""

import logging
import seamm
import pprint  # noqa: F401

logger = logging.getLogger(__name__)


class InteractionEdaParameters(seamm.Parameters):
    """
    The control parameters for Interaction EDA.

    You need to replace the "time" entry in dictionary below these comments with the
    definitions of parameters to control this step. The keys are parameters for the
    current plugin,the values are dictionaries as outlined below.

    Examples
    --------
    ::

        parameters = {
            "time": {
                "default": 100.0,
                "kind": "float",
                "default_units": "ps",
                "enumeration": tuple(),
                "format_string": ".1f",
                "description": "Simulation time:",
                "help_text": ("The time to simulate in the dynamics run.")
            },
        }

    parameters : {str: {str: str}}
        A dictionary containing the parameters for the current step.
        Each key of the dictionary is a dictionary that contains the
        the following keys:

    parameters["default"] :
        The default value of the parameter, used to reset it.

    parameters["kind"] : enum()
        Specifies the kind of a variable. One of  "integer", "float", "string",
        "boolean", or "enum"

        While the "kind" of a variable might be a numeric value, it may still have
        enumerated custom values meaningful to the user. For instance, if the parameter
        is a convergence criterion for an optimizer, custom values like "normal",
        "precise", etc, might be adequate. In addition, any parameter can be set to a
        variable of expression, indicated by having "$" as the first character in the
        field. For example, $OPTIMIZER_CONV.

    parameters["default_units"] : str
        The default units, used for resetting the value.

    parameters["enumeration"] : tuple
        A tuple of enumerated values.

    parameters["format_string"] : str
        A format string for "pretty" output.

    parameters["description"] : str
        A short string used as a prompt in the GUI.

    parameters["help_text"] : str
        A longer string to display as help for the user.

    See Also
    --------
    InteractionEda, TkInteractionEda, InteractionEdaParameters, InteractionEdaStep
    """

    parameters = {
        "backend": {
            "default": "GPU4PySCF",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("GPU4PySCF",),
            "format_string": "s",
            "description": "Code:",
            "help_text": (
                "The code that does the EDA. GPU4PySCF runs second-generation "
                "ALMO-EDA on a GPU with good double precision (A100, A30, H100/H200, "
                "B200)."
            ),
        },
        "functional": {
            "default": "wB97M-V",
            "kind": "string",
            "default_units": "",
            "enumeration": ("wB97M-V", "wB97X-V", "B97M-V", "wB97X-D3BJ", "B3LYP"),
            "format_string": "s",
            "description": "Functional:",
            "help_text": (
                "The density functional, in PySCF's names (case does not matter). "
                "wB97M-V/def2-TZVPD is the validated default."
            ),
        },
        "basis": {
            "default": "def2-TZVPD",
            "kind": "string",
            "default_units": "",
            "enumeration": ("def2-TZVPD", "def2-TZVPPD", "def2-SVPD", "def2-QZVPPD"),
            "format_string": "s",
            "description": "Basis set:",
            "help_text": "The orbital basis set, in PySCF's names.",
        },
        "density fitting": {
            "default": "no",
            "kind": "string",
            "default_units": "",
            "enumeration": ("no", "def2-universal-jkfit"),
            "format_string": "s",
            "description": "Density fitting:",
            "help_text": (
                "'no' for exact integrals (the validated default), or the auxiliary "
                "basis for density fitting."
            ),
        },
        "vv10": {
            "default": "as in the functional",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("as in the functional", "off (whole functional)"),
            "format_string": "s",
            "description": "VV10 nonlocal correlation:",
            "help_text": (
                "Whether to keep the VV10 nonlocal correlation that a -V functional "
                "includes. 'off (whole functional)' removes it everywhere, which is "
                "a different functional with different energies, e.g. wB97X-V with "
                "D4 dispersion (Najibi and Goerigk's wB97X-D4)."
            ),
        },
        "vv10 in response": {
            "default": "on",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("on", "off"),
            "format_string": "s",
            "description": "VV10 in the FERF response:",
            "help_text": (
                "Whether the FERF response kernel includes VV10. 'off' drops it from "
                "the response only; the SCF and SCF-MI energies keep the full "
                "functional. This matches codes without a VV10 response kernel "
                "(Psi4); on shell A it moved energy between polarization and charge "
                "transfer by at most 0.0014 kJ/mol."
            ),
        },
        "dispersion": {
            "default": "none",
            "kind": "string",
            "default_units": "",
            "enumeration": ("none", "d4", "d3bj"),
            "format_string": "s",
            "description": "Dispersion correction:",
            "help_text": (
                "An empirical dispersion correction added to the functional, in "
                "PySCF's names (e.g. 'd4', or 'd4:wb97x' for a parameter set). It is "
                "included in the dispersion term; D4's three-body (ATM) part shows "
                "up in the ≥3-body dispersion."
            ),
        },
        "ferf response": {
            "default": "coupled",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("coupled", "uncoupled"),
            "format_string": "s",
            "description": "FERF response:",
            "help_text": (
                "How the fragment electric-field response functions (FERF-DQ) that "
                "define the polarization space are made: 'coupled' solves the "
                "coupled-perturbed equations (the validated default); 'uncoupled' "
                "uses orbital-energy denominators without the XC kernel. Only the "
                "split between polarization and charge transfer changes."
            ),
        },
        "svd threshold": {
            "default": 1.0e-4,
            "kind": "float",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": ".1e",
            "description": "FERF singular-value threshold:",
            "help_text": (
                "Virtual singular values below this are dropped from the "
                "polarization space (GPU4PySCF's default, 1e-4, is the validated one)."
            ),
        },
        "fragments": {
            "default": "molecules",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": ("molecules", "atom column"),
            "format_string": "s",
            "description": "Fragments:",
            "help_text": (
                "Each molecule (bonded group of atoms) is a fragment, with its charge "
                "from the formal charges or the known ions; or an integer atom "
                "column assigns the atoms to fragments."
            ),
        },
        "fragment column": {
            "default": "fragment",
            "kind": "string",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "s",
            "description": "Atom column:",
            "help_text": "The integer atom column giving each atom's fragment.",
        },
        "calculations": {
            "default": "cluster and pairs",
            "kind": "enumeration",
            "default_units": "",
            "enumeration": (
                "cluster and pairs",
                "cluster, pairs and triples",
                "cluster",
                "pairs",
            ),
            "format_string": "s",
            "description": "Calculations:",
            "help_text": (
                "Which sets of fragments to compute. The cluster and all pairs give "
                "the exact ≥3-body part of each term; triples add the 3-body "
                "increments, a truncated estimate."
            ),
        },
        "triples cutoff": {
            "default": "all",
            "kind": "float",
            "default_units": "Å",
            "enumeration": ("all",),
            "format_string": ".2f",
            "description": "Triples within:",
            "help_text": (
                "Use only the triples whose three pairs are all within this distance "
                "(shortest atom-atom distance), or 'all'."
            ),
        },
        "results": {
            "default": {},
            "kind": "dictionary",
            "default_units": "",
            "enumeration": tuple(),
            "format_string": "",
            "description": "results",
            "help_text": "The results to save to variables or in tables.",
        },
    }

    def __init__(self, defaults={}, data=None):
        """
        Initialize the parameters, by default with the parameters defined above

        Parameters
        ----------
        defaults: dict
            A dictionary of parameters to initialize. The parameters
            above are used first and any given will override/add to them.
        data: dict
            A dictionary of keys and a subdictionary with value and units
            for updating the current, default values.

        Returns
        -------
        None
        """

        logger.debug("InteractionEdaParameters.__init__")

        super().__init__(
            defaults={**InteractionEdaParameters.parameters, **defaults}, data=data
        )
