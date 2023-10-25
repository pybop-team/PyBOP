#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# This file is adapted from Pints
# (see https://github.com/pints-team/pints)
#

import sys
import os

#
# Version info
#
from pybop.version import __version__

#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = "{: .17e}"
# Absolute path to the pybop module
script_path = os.path.dirname(__file__)

#
# Model Classes
#
from .models import lithium_ion
from .models.BaseModel import BaseModel

#
# Parameterisation class
#
from .identification import Parameterisation, ParameterSet, Parameter, Observed

#
# Priors class
#
from .priors import Gaussian, Uniform, Exponential

#
# Optimisation class
#
from .optimisation import BaseOptimisation
from .optimisation.NLoptOptimize import NLoptOptimize
from .optimisation.SciPyMinimize import SciPyMinimize

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
