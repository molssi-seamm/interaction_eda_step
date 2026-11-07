# -*- coding: utf-8 -*-

"""
interaction_eda_step
A SEAMM plug-in for Interaction EDA
"""

# Bring up the classes so that they appear to be directly in
# the interaction_eda_step package.

from .interaction_eda import InteractionEda  # noqa: F401, E501
from .interaction_eda_parameters import InteractionEdaParameters  # noqa: F401, E501
from .interaction_eda_step import InteractionEdaStep  # noqa: F401, E501
from .tk_interaction_eda import TkInteractionEda  # noqa: F401, E501

from .metadata import metadata  # noqa: F401

# Handle versioneer
from ._version import get_versions

__author__ = "Paul Saxe"
__email__ = "psaxe@molssi.org"
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
