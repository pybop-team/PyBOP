#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# This file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import sys
from pathlib import Path

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
script_path = Path(__file__).parent

#
# Utilities
#
from ._utils import add_spaces, add_noise, is_numeric, load_data_dict, save_data_dict

#
# Dataset class
#
from .processing.dataset import Dataset, import_pybamm_solution, import_pyprobe_result
from .processing.interpolate_current import generate_consistent_current, downsample_constant_current

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
from .parameters.distributions import Distribution, Exponential, Gaussian, JointDistribution, LogNormal, LogUniform, Unbounded, Uniform
from .parameters.multivariate_distributions import MultivariateNonparametric, MultivariateUniform, MultivariateGaussian, MultivariateLogNormal, MarginalDistribution

#
# Model classes
#
from .models import lithium_ion
from .models._exponential_decay import ExponentialDecayModel
from .models.lithium_ion.utils import Interpolant, InverseOCV

#
# PyBaMM utility classes
#
from . import pybamm

#
# Problem classes
#
from .problems.problem import Problem
from .problems.meta_problem import MetaProblem
from .problems.log_pdf import LogPDF, LogPosterior

#
# Simulator classes
#
from .simulators.base_simulator import BaseSimulator
from .simulators.solution import Solution
from .simulators.failed_solution import FailedVariable, FailedSolution

#
# Cost classes
#
from .costs.error_measures import (
    ErrorMeasure,
    RootMeanSquaredError,
    MeanAbsoluteError,
    MeanSquaredError,
    SumSquaredError,
    Minkowski,
    SumOfPower,
)
from .costs.log_likelihoods import (
    LogLikelihood,
    GaussianLogLikelihood,
    GaussianLogLikelihoodKnownSigma,
)
from .costs.base_cost import LogPrior
from .costs.weighted_cost import WeightedCost
from .costs.design_cost import DesignCost
from .costs.feature_distances import SquareRootFeatureDistance, ExponentialFeatureDistance

#
# Evaluation
#
from ._evaluation import PopulationEvaluator, ScalarEvaluator, SequentialEvaluator

#
# Optimisation logging and result
#
from ._logging import Logger
from ._result import Result

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
from .optimisers.ep_bolfi_optimiser import EP_BOLFI, EPBOLFIOptions

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
# Analysis
#
from .analysis.classification import classify_using_hessian, plot_hessian_eigenvectors

#
# Applications
#
from .applications.base_method import BaseApplication
from .applications.ocp_methods import OCPMerge, OCPAverage, OCPCapacityToStoichiometry
from .applications.gitt_methods import GITTPulseFit, GITTFit

#
# Plotting classes
#
from . import plot as plot

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys
