from pints import MALAMCMC as PintsMALAMCMC
from pints import AdaptiveCovarianceMCMC as PintsAdaptiveCovarianceMCMC
from pints import DifferentialEvolutionMCMC as PintsDifferentialEvolutionMCMC
from pints import DramACMC as PintsDramACMC
from pints import DreamMCMC as PintsDREAM
from pints import EmceeHammerMCMC as PintsEmceeHammerMCMC
from pints import HaarioACMC as PintsHaarioACMC
from pints import HaarioBardenetACMC as PintsHaarioBardenetACMC
from pints import HamiltonianMCMC as PintsHamiltonianMCMC
from pints import MetropolisRandomWalkMCMC as PintsMetropolisRandomWalkMCMC
from pints import MonomialGammaHamiltonianMCMC as PintsMonomialGammaHamiltonianMCMC
from pints import NoUTurnMCMC
from pints import PopulationMCMC as PintsPopulationMCMC
from pints import RaoBlackwellACMC as PintsRaoBlackwellACMC
from pints import RelativisticMCMC as PintsRelativisticMCMC
from pints import SliceDoublingMCMC as PintsSliceDoublingMCMC
from pints import SliceRankShrinkingMCMC as PintsSliceRankShrinkingMCMC
from pints import SliceStepoutMCMC as PintsSliceStepoutMCMC

from pybop import BasePintsSampler, PintsSamplerOptions
from pybop.costs._likelihoods import LogPosterior

__all__: list[str] = [
    "NUTS",
    "DREAM",
    "AdaptiveCovarianceMCMC",
    "DifferentialEvolutionMCMC",
    "DramACMC",
    "EmceeHammerMCMC",
    "HaarioACMC",
    "HaarioBardenetACMC",
    "HamiltonianMCMC",
    "MALAMCMC",
    "MetropolisRandomWalkMCMC",
    "MonomialGammaHamiltonianMCMC",
    "PopulationMCMC",
    "RaoBlackwellACMC",
    "RelativisticMCMC",
    "SliceDoublingMCMC",
    "SliceRankShrinkingMCMC",
    "SliceStepoutMCMC",
]


class NUTS(BasePintsSampler):
    """
    Implements the No-U-Turn Sampler (NUTS) algorithm.

    This class wraps the NUTS sampler from the PINTS library.
    NUTS is a Markov chain Monte Carlo (MCMC) method for sampling
    from a probability distribution, extending Hamiltonian Monte Carlo
    with an adaptive integration time for efficient exploration.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, NoUTurnMCMC, options=options)


class DREAM(BasePintsSampler):
    """
    Implements the DiffeRential Evolution Adaptive Metropolis (DREAM) algorithm.

    This class wraps the DREAM sampler from the PINTS library.
    DREAM combines Differential Evolution and Adaptive Metropolis to efficiently
    explore complex parameter spaces using a population of chains.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsDREAM, options=options)


class AdaptiveCovarianceMCMC(BasePintsSampler):
    """
    Implements the Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class wraps the Adaptive Covariance MCMC sampler from the PINTS library.
    The proposal distribution's covariance matrix is adapted during sampling
    to improve efficiency and convergence.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsAdaptiveCovarianceMCMC, options=options)


class DifferentialEvolutionMCMC(BasePintsSampler):
    """
    Implements the Differential Evolution Markov Chain Monte Carlo (MCMC) algorithm.

    This class wraps the Differential Evolution MCMC sampler from the PINTS library.
    Uses a population of chains and differential evolution proposals for efficient
    exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsDifferentialEvolutionMCMC, options=options)


class DramACMC(BasePintsSampler):
    """
    Implements the Delayed Rejection Adaptive Metropolis (DRAM) Adaptive Covariance MCMC algorithm.

    This class wraps the DRAM Adaptive Covariance MCMC sampler from the PINTS library.
    Combines delayed rejection and adaptive covariance for robust and efficient sampling.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsDramACMC, options=options)


class EmceeHammerMCMC(BasePintsSampler):
    """
    Implements the Emcee Hammer Markov Chain Monte Carlo (MCMC) algorithm.

    This class wraps the Emcee Hammer MCMC sampler from the PINTS library.
    The Emcee Hammer is an affine-invariant ensemble sampler, effective for
    high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsEmceeHammerMCMC, options=options)


class HaarioACMC(BasePintsSampler):
    """
    Implements the Haario Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Haario Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method adapts the proposal distribution's covariance matrix based on the
    history of the chain, improving sampling efficiency and convergence.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsHaarioACMC, options=options)


class HaarioBardenetACMC(BasePintsSampler):
    """
    Implements the Haario-Bardenet Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Haario-Bardenet Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method combines the adaptive covariance approach with an additional
    mechanism to improve performance in high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsHaarioBardenetACMC, options=options)


class HamiltonianMCMC(BasePintsSampler):
    """
    Implements the Hamiltonian Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Hamiltonian MCMC sampler from the PINTS library.
    This MCMC method uses Hamiltonian dynamics to propose new states,
    allowing for efficient exploration of high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsHamiltonianMCMC, options=options)


class MALAMCMC(BasePintsSampler):
    """
    Implements the Metropolis Adjusted Langevin Algorithm (MALA) Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the MALA MCMC sampler from the PINTS library.
    This MCMC method combines the Metropolis-Hastings algorithm with
    Langevin dynamics to improve sampling efficiency and convergence.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsMALAMCMC, options=options)


class MetropolisRandomWalkMCMC(BasePintsSampler):
    """
    Implements the Metropolis Random Walk Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Metropolis Random Walk MCMC sampler from the PINTS library.
    This classic MCMC method uses a simple random walk proposal distribution
    and the Metropolis-Hastings acceptance criterion.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsMetropolisRandomWalkMCMC, options=options)


class MonomialGammaHamiltonianMCMC(BasePintsSampler):
    """
    Implements the Monomial Gamma Hamiltonian Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Monomial Gamma Hamiltonian MCMC sampler from the PINTS library.
    This MCMC method uses Hamiltonian dynamics with a monomial gamma distribution
    for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsMonomialGammaHamiltonianMCMC, options=options)


class PopulationMCMC(BasePintsSampler):
    """
    Implements the Population Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Population MCMC sampler from the PINTS library.
    This MCMC method uses a population of chains at different temperatures
    to explore the parameter space more efficiently and avoid local minima.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsPopulationMCMC, options=options)


class RaoBlackwellACMC(BasePintsSampler):
    """
    Implements the Rao-Blackwell Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Rao-Blackwell Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method improves sampling efficiency by combining Rao-Blackwellisation
    with adaptive covariance strategies.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsRaoBlackwellACMC, options=options)


class RelativisticMCMC(BasePintsSampler):
    """
    Implements the Relativistic Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Relativistic MCMC sampler from the PINTS library.
    This MCMC method uses concepts from relativistic mechanics to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsRelativisticMCMC, options=options)


class SliceDoublingMCMC(BasePintsSampler):
    """
    Implements the Slice Doubling Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Doubling MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a doubling procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsSliceDoublingMCMC, options=options)


class SliceRankShrinkingMCMC(BasePintsSampler):
    """
    Implements the Slice Rank Shrinking Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Rank Shrinking MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a rank shrinking procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options for the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsSliceRankShrinkingMCMC, options=options)


class SliceStepoutMCMC(BasePintsSampler):
    """
    Implements the Slice Stepout Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Stepout MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a stepout procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : pybop.LogPosterior
        The log-posterior to sample.
    options : pybop.PintsSamplerOptions, optional
        Additional options to pass to the sampler.
    """

    def __init__(
        self, log_pdf: LogPosterior, options: PintsSamplerOptions | None = None
    ):
        super().__init__(log_pdf, PintsSliceStepoutMCMC, options=options)
