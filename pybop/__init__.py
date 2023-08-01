#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# The code in this file is adapted from Pints
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
# Absolute path to the PyBOP repo
script_path = os.path.abspath(__file__)

#
# Model Classes
#
from .models import lithium_ion

#
# Parameterisation class
#

from .parameterisation import Parameterisation
from .parameters import Parameter

#
# Observation class
#
from .observations import Observed

#
# Priors class
#
from .priors import Gaussian, Uniform, Exponential

#
# Simulation class
#
from .simulation import Simulation


#
# Optimisation class
#
from .optimisation.base_optimisation import BaseOptimisation
from .optimisation.nlopt_opt import nlopt_opt
from .optimisation.scipy_opt import scipy_opt

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
