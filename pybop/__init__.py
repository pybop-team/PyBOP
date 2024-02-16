#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# This file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
from __future__ import annotations
import sys
from os import path

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
# Absolute path to the pybop repo
script_path = path.dirname(__file__)

#
# Utilities
#
from ._utils import is_numeric

#
# Cost class
#
from .costs.base_cost import BaseCost
from .costs.fitting_costs import (
    RootMeanSquaredError,
    SumSquaredError,
    ObserverCost,
)
from .costs.design_costs import (
    DesignCost,
    GravimetricEnergyDensity,
    VolumetricEnergyDensity,
)

#
# Dataset class
#
from ._dataset import Dataset

#
# Model classes
#
from .models.base_model import BaseModel
from .models import lithium_ion
from .models import empirical
from .models.base_model import TimeSeriesState
from .models.base_model import Inputs

#
# Experiment class
#
from ._experiment import Experiment

#
# Main optimisation class
#
from ._optimisation import Optimisation

#
# Optimiser class
#
from .optimisers.base_optimiser import BaseOptimiser
from .optimisers.scipy_optimisers import SciPyMinimize, SciPyDifferentialEvolution
from .optimisers.pints_optimisers import (
    GradientDescent,
    Adam,
    CMAES,
    IRPropMin,
    PSO,
    SNES,
    XNES,
)

#
# Parameter classes
#
from .parameters.parameter import Parameter
from .parameters.parameter_set import ParameterSet
from .parameters.priors import Gaussian, Uniform, Exponential

#
# Problem class
#
from ._problem import FittingProblem, DesignProblem

#
# Observer classes
#
from .observers.unscented_kalman import UnscentedKalmanFilterObserver
from .observers.observer import Observer

#
# Plotting class
#
from .plotting.plotly_manager import PlotlyManager
from .plotting.quick_plot import StandardPlot, StandardSubplot, plot_trajectories
from .plotting.plot_cost2d import plot_cost2d
from .plotting.plot_dataset import plot_dataset
from .plotting.plot_convergence import plot_convergence, plot_optim2d
from .plotting.plot_parameters import plot_parameters
from .plotting.plot_problem import quick_plot

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
