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
# Cost function class
#
from .costs.error_costs import RMSE

#
# Dataset class
#
from .datasets.base_dataset import Dataset

#
# Model classes
#
from .models.base_model import BaseModel
from .models import lithium_ion

#
# Main optimisation class
#
from .optimisation import Optimisation

#
# Optimiser class
#
from .optimisers.base_optimiser import BaseOptimiser
from .optimisers.nlopt_optimize import NLoptOptimize
from .optimisers.scipy_minimize import SciPyMinimize

#
# Parameter classes
#
from .parameters.base_parameter import Parameter
from .parameters.base_parameter_set import ParameterSet
from .parameters.priors import Gaussian, Uniform, Exponential

#
# Plotting class
#
from .plotting.quick_plot import QuickPlot

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
