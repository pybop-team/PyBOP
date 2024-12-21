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
except Exception as e: # pragma: no cover
    error_message = (
        "Multiprocessing context could not be set. "
        "Continuing import without setting context.\n"
        f"Error: {e}"
    ) # pragma: no cover
    print(error_message) # pragma: no cover
    pass # pragma: no cover

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
# Experiment class
#
from ._experiment import Experiment

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
from .parameters.parameter import Parameter, Parameters
from .parameters.parameter_set import ParameterSet
from .parameters.priors import BasePrior, Gaussian, Uniform, Exponential, JointLogPrior

#
# Model classes
#
from .models.base_model import BaseModel
from .models import lithium_ion
from .models import empirical
from .models._exponential_decay import ExponentialDecayModel
from .models.base_model import TimeSeriesState
from .models.base_model import Inputs

#
# Problem classes
#
from .problems.base_problem import BaseProblem
from .problems.fitting_problem import FittingProblem
from .problems.multi_fitting_problem import MultiFittingProblem
from .problems.design_problem import DesignProblem

#
# Cost classes
#
from .costs.base_cost import BaseCost
from .costs.fitting_costs import (
    RootMeanSquaredError,
    SumSquaredError,
    Minkowski,
    SumofPower,
    ObserverCost,
)
from .costs.design_costs import (
    DesignCost,
    GravimetricEnergyDensity,
    VolumetricEnergyDensity,
    GravimetricPowerDensity,
    VolumetricPowerDensity,
)
from .costs._likelihoods import (
    BaseLikelihood,
    GaussianLogLikelihood,
    GaussianLogLikelihoodKnownSigma,
    ScaledLogLikelihood,
    LogPosterior,
)
from .costs._weighted_cost import WeightedCost

#
# Experimental
#
from .experimental.jax_costs import BaseJaxCost, JaxSumSquaredError, JaxLogNormalLikelihood, JaxGaussianLogLikelihoodKnownSigma

#
# Evaluation
#
from ._evaluation import SequentialJaxEvaluator, SciPyEvaluator

#
# Optimiser classes
#

from .optimisers._cuckoo import CuckooSearchImpl
from .optimisers._random_search import RandomSearchImpl
from .optimisers._adamw import AdamWImpl
from .optimisers._gradient_descent import GradientDescentImpl
from .optimisers.base_optimiser import BaseOptimiser, OptimisationResult, MultiOptimisationResult
from .optimisers.base_pints_optimiser import BasePintsOptimiser
from .optimisers.scipy_optimisers import (
    BaseSciPyOptimiser,
    SciPyMinimize,
    SciPyDifferentialEvolution
)
from .optimisers.pints_optimisers import (
    GradientDescent,
    CMAES,
    IRPropMin,
    NelderMead,
    PSO,
    SNES,
    XNES,
    CuckooSearch,
    RandomSearch,
    AdamW,
)
from .optimisers.optimisation import Optimisation

#
# Monte Carlo classes
#
from .samplers.base_sampler import BaseSampler
from .samplers.base_pints_sampler import BasePintsSampler
from .samplers.pints_samplers import (
    NUTS, DREAM, AdaptiveCovarianceMCMC,
    DifferentialEvolutionMCMC, DramACMC,
    EmceeHammerMCMC,
    HaarioACMC, HaarioBardenetACMC,
    HamiltonianMCMC, MALAMCMC,
    MetropolisRandomWalkMCMC, MonomialGammaHamiltonianMCMC,
    PopulationMCMC, RaoBlackwellACMC,
    RelativisticMCMC, SliceDoublingMCMC,
    SliceRankShrinkingMCMC, SliceStepoutMCMC,
)
from .samplers.mcmc_sampler import MCMCSampler

#
# Observer classes
#
from .observers.unscented_kalman import UnscentedKalmanFilterObserver
from .observers.observer import Observer

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
