#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# This file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import sys
from os import path

#
# Multiprocessing
#
try:
    import multiprocessing as mp

    if sys.platform == "win32":
        mp.set_start_method("spawn")
    else:
        mp.set_start_method("fork")
except Exception as e:  # pragma: no cover
    error_message = (
        "Multiprocessing context could not be set. "
        "Continuing import without setting context.\n"
        f"Error: {e}"
    )  # pragma: no cover
    print(error_message)  # pragma: no cover
    pass  # pragma: no cover

#
# Version info
#
from pybop._version import __version__

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
from ._utils import is_numeric, SymbolReplacer


#
# Dataset class
#
from ._dataset import Dataset

#
# Transformation classes
#
from .transformation.base_transformation import Transformation
from .transformation.transformations import (
    IdentityTransformation,
    ScaledTransformation,
    LogTransformation,
    ComposedTransformation,
    UnitHyperCube,
)

#
# Parameter classes
#
from .parameters.priors import BasePrior, Gaussian, Uniform, Exponential, JointLogPrior
from .parameters.parameter import Parameter, Parameters, Inputs, ParameterError, ParameterNotFoundError, ParameterValidationError, ParameterValueValidator
from .parameters.priors import BasePrior, Gaussian, Uniform, Exponential, JointLogPrior

#
# Model classes
#
from .models import lithium_ion
from .models._exponential_decay import ExponentialDecayModel

# Cost classes
#
from . import costs
from .costs.base_cost import CallableCost

#
# Problem classes
#
from .problems.base_problem import Problem
from .problems.pybamm_problem import PybammProblem
from .problems.pybamm_eis_problem import PybammEISProblem
from .problems.python_problem import PythonProblem

#
# Cost classes
#
from .costs.error_measures import (
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredError,
    SumSquaredError,
    Minkowski,
    SumOfPower,
)

#
# Problem Builder
#
from .builders.base import BaseBuilder
from .builders.pybamm import Pybamm
from .builders.pybamm_eis import PybammEIS
from .builders.python import Python


#
# Evaluation
#
from ._evaluation import SciPyEvaluator

#
# Optimiser classes
#

from .optimisers._cuckoo import CuckooSearchImpl
from .optimisers._random_search import RandomSearchImpl
from .optimisers._adamw import AdamWImpl
from .optimisers._gradient_descent import GradientDescentImpl
from .optimisers._simulated_annealing import SimulatedAnnealingImpl
from .optimisers._irprop_plus import IRPropPlusImpl
from .optimisers._result import OptimisationResult
from .optimisers.base_optimiser import (
    BaseOptimiser,
    OptimisationLogger,
    OptimiserOptions,
)
from .optimisers.base_pints_optimiser import BasePintsOptimiser, PintsOptions
from .optimisers.scipy_optimisers import (
    BaseSciPyOptimiser,
    SciPyMinimize,
    ScipyMinimizeOptions,
    SciPyDifferentialEvolution,
    SciPyDifferentialEvolutionOptions,
)
from .optimisers.pints_optimisers import (
    GradientDescent,
    CMAES,
    IRPropMin,
    IRPropPlus,
    NelderMead,
    PSO,
    SNES,
    XNES,
    CuckooSearch,
    RandomSearch,
    AdamW,
    SimulatedAnnealing,
)

#
# Monte Carlo classes
#
from .samplers.chain_processor import (
    ChainProcessor,
    MultiChainProcessor,
    SingleChainProcessor,
)
from .samplers.base_sampler import BaseSampler, SamplerOptions
from .samplers.base_pints_sampler import BasePintsSampler, PintsSamplerOptions
from .samplers.pints_samplers import (
    NUTS,
    DREAM,
    AdaptiveCovarianceMCMC,
    DifferentialEvolutionMCMC,
    DramACMC,
    EmceeHammerMCMC,
    HaarioACMC,
    HaarioBardenetACMC,
    HamiltonianMCMC,
    MALAMCMC,
    MetropolisRandomWalkMCMC,
    MonomialGammaHamiltonianMCMC,
    PopulationMCMC,
    RaoBlackwellACMC,
    RelativisticMCMC,
    SliceDoublingMCMC,
    SliceRankShrinkingMCMC,
    SliceStepoutMCMC,
)

#
# Classification classes
#
from ._classification import classify_using_hessian

#
# Plotting classes
#
from . import plot as plot
from .samplers.mcmc_summary import PosteriorSummary

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
