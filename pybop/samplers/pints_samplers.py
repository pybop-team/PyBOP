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

from pybop import BasePintsSampler


class NUTS(BasePintsSampler):
    """
    Implements the No-U-Turn Sampler (NUTS) algorithm.

    This class extends the NUTS sampler from the PINTS library.
    NUTS is a Markov chain Monte Carlo (MCMC) method for sampling
    from a probability distribution. It is an extension of the
    Hamiltonian Monte Carlo (HMC) method, which uses a dynamic
    integration time to explore the parameter space more efficiently.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the NUTS sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf, NoUTurnMCMC, chains=chains, x0=x0, cov0=cov0, **kwargs
        )


class DREAM(BasePintsSampler):
    """
    Implements the DiffeRential Evolution Adaptive Metropolis (DREAM) algorithm.

    This class extends the DREAM sampler from the PINTS library.
    DREAM is a Markov chain Monte Carlo (MCMC) method for sampling
    from a probability distribution. It combines the Differential
    Evolution (DE) algorithm with the Adaptive Metropolis (AM) algorithm
    to explore the parameter space more efficiently.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the DREAM sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(log_pdf, PintsDREAM, chains=chains, x0=x0, cov0=cov0, **kwargs)


class AdaptiveCovarianceMCMC(BasePintsSampler):
    """
    Implements the Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method adapts the proposal distribution covariance matrix
    during the sampling process to improve efficiency and convergence.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Adaptive Covariance MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsAdaptiveCovarianceMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class DifferentialEvolutionMCMC(BasePintsSampler):
    """
    Implements the Differential Evolution Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Differential Evolution MCMC sampler from the PINTS library.
    This MCMC method uses the Differential Evolution algorithm to explore the
    parameter space more efficiently by evolving a population of chains.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Differential Evolution MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsDifferentialEvolutionMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class DramACMC(BasePintsSampler):
    """
    Implements the Delayed Rejection Adaptive Metropolis (DRAM) Adaptive Covariance Markov Chain
    Monte Carlo (MCMC) algorithm.

    This class extends the DRAM Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method combines Delayed Rejection with Adaptive Metropolis to enhance
    the efficiency and robustness of the sampling process.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the DRAM Adaptive Covariance MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsDramACMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class EmceeHammerMCMC(BasePintsSampler):
    """
    Implements the Emcee Hammer Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Emcee Hammer MCMC sampler from the PINTS library.
    The Emcee Hammer is an affine-invariant ensemble sampler for MCMC, which is
    particularly effective for high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Emcee Hammer MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsEmceeHammerMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class HaarioACMC(BasePintsSampler):
    """
    Implements the Haario Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Haario Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method adapts the proposal distribution's covariance matrix based on the
    history of the chain, improving sampling efficiency and convergence.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Haario Adaptive Covariance MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsHaarioACMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class HaarioBardenetACMC(BasePintsSampler):
    """
    Implements the Haario-Bardenet Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Haario-Bardenet Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method combines the adaptive covariance approach with an additional
    mechanism to improve performance in high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Haario-Bardenet Adaptive Covariance MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsHaarioBardenetACMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class HamiltonianMCMC(BasePintsSampler):
    """
    Implements the Hamiltonian Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Hamiltonian MCMC sampler from the PINTS library.
    This MCMC method uses Hamiltonian dynamics to propose new states,
    allowing for efficient exploration of high-dimensional parameter spaces.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Hamiltonian MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsHamiltonianMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class MALAMCMC(BasePintsSampler):
    """
    Implements the Metropolis Adjusted Langevin Algorithm (MALA) Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the MALA MCMC sampler from the PINTS library.
    This MCMC method combines the Metropolis-Hastings algorithm with
    Langevin dynamics to improve sampling efficiency and convergence.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the MALA MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsMALAMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class MetropolisRandomWalkMCMC(BasePintsSampler):
    """
    Implements the Metropolis Random Walk Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Metropolis Random Walk MCMC sampler from the PINTS library.
    This classic MCMC method uses a simple random walk proposal distribution
    and the Metropolis-Hastings acceptance criterion.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Metropolis Random Walk MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsMetropolisRandomWalkMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class MonomialGammaHamiltonianMCMC(BasePintsSampler):
    """
    Implements the Monomial Gamma Hamiltonian Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Monomial Gamma Hamiltonian MCMC sampler from the PINTS library.
    This MCMC method uses Hamiltonian dynamics with a monomial gamma distribution
    for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Monomial Gamma Hamiltonian MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsMonomialGammaHamiltonianMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class PopulationMCMC(BasePintsSampler):
    """
    Implements the Population Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Population MCMC sampler from the PINTS library.
    This MCMC method uses a population of chains at different temperatures
    to explore the parameter space more efficiently and avoid local minima.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Population MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsPopulationMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class RaoBlackwellACMC(BasePintsSampler):
    """
    Implements the Rao-Blackwell Adaptive Covariance Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Rao-Blackwell Adaptive Covariance MCMC sampler from the PINTS library.
    This MCMC method improves sampling efficiency by combining Rao-Blackwellisation
    with adaptive covariance strategies.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Rao-Blackwell Adaptive Covariance MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsRaoBlackwellACMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class RelativisticMCMC(BasePintsSampler):
    """
    Implements the Relativistic Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Relativistic MCMC sampler from the PINTS library.
    This MCMC method uses concepts from relativistic mechanics to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Relativistic MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsRelativisticMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class SliceDoublingMCMC(BasePintsSampler):
    """
    Implements the Slice Doubling Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Doubling MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a doubling procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Slice Doubling MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsSliceDoublingMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class SliceRankShrinkingMCMC(BasePintsSampler):
    """
    Implements the Slice Rank Shrinking Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Rank Shrinking MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a rank shrinking procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Slice Rank Shrinking MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsSliceRankShrinkingMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )


class SliceStepoutMCMC(BasePintsSampler):
    """
    Implements the Slice Stepout Markov Chain Monte Carlo (MCMC) algorithm.

    This class extends the Slice Stepout MCMC sampler from the PINTS library.
    This MCMC method uses slice sampling with a stepout procedure to propose new states,
    allowing for efficient exploration of the parameter space.

    Parameters
    ----------
    log_pdf : (pybop.LogPosterior or List[pybop.LogPosterior])
        A function that calculates the log-probability density.
    chains : int
        The number of chains to run.
    x0 : ndarray, optional
        Initial positions for the chains.
    cov0 : ndarray, optional
        Initial covariance matrix.
    **kwargs
        Additional arguments to pass to the Slice Stepout MCMC sampler.
    """

    def __init__(self, log_pdf, chains, x0=None, cov0=None, **kwargs):
        super().__init__(
            log_pdf,
            PintsSliceStepoutMCMC,
            chains=chains,
            x0=x0,
            cov0=cov0,
            **kwargs,
        )
