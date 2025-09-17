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
from ._utils import add_spaces, is_numeric, FailedVariable, FailedSolution, SymbolReplacer, RecommendedSolver

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
from .costs.fitting_costs import FittingCost
from .costs.error_measures import (
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredError,
    SumSquaredError,
    Minkowski,
    SumOfPower,
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
# Evaluation
#
from ._evaluation import PopulationEvaluator, ScalarEvaluator, SequentialEvaluator

#
# Optimisation logging
#
from ._logging import Logger
from ._result import OptimisationResult

#
# Optimiser classes
#
from .optimisers.base_optimiser import BaseOptimiser, OptimiserOptions
from .optimisers.base_pints_optimiser import BasePintsOptimiser, PintsOptions
from .optimisers.scipy_optimisers import (
    BaseSciPyOptimiser,
    SciPyMinimize,
    SciPyMinimizeOptions,
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
from .analysis.classification import classify_using_hessian

#
# Applications
#
from .applications.base_method import BaseApplication, Interpolant, InverseOCV
from .applications.ocp_methods import OCPMerge, OCPAverage, OCPCapacityToStoichiometry
from .applications.gitt_methods import GITTPulseFit, GITTFit

#
# Plotting classes
#
from . import plot as plot
from .samplers.mcmc_summary import PosteriorSummary

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
