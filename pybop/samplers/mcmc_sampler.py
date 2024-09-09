from pybop import AdaptiveCovarianceMCMC


class MCMCSampler:
    """
    A high-level class for MCMC sampling.

    This class provides an alternative API to the `PyBOP.Sampler()` API,
    specifically allowing for single user-friendly interface for the
    optimisation process.
    """

    def __init__(
        self,
        log_pdf,
        chains,
        sampler=AdaptiveCovarianceMCMC,
        x0=None,
        cov0=None,
        **kwargs,
    ):
        """
        Initialize the MCMCSampler.

        Parameters
        ----------
        log_pdf : pybop.BaseCost
            The log-probability density function to be sampled.
        chains : int
            The number of MCMC chains to be run.
        sampler : pybop.MCMCSampler, optional
            The MCMC sampler class to be used. Defaults to `pybop.MCMC`.
        x0 : np.ndarray, optional
            Initial positions for the MCMC chains. Defaults to None.
        cov0 : np.ndarray, optional
            Initial step sizes for the MCMC chains. Defaults to None.
        **kwargs : dict
            Additional keyword arguments to pass to the sampler.

        Raises
        ------
        ValueError
            If the sampler could not be constructed due to an exception.
        """

        self.sampler = sampler(log_pdf, chains, x0=x0, sigma0=cov0, **kwargs)

    def run(self):
        """
        Run the MCMC sampling process.

        Returns
        -------
        list
            The result of the sampling process.
        """
        return self.sampler.run()

    def __getattr__(self, attr):
        """
        Delegate attribute access to the underlying sampler if the attribute
        is not found in the MCMCSampler instance.

        Parameters
        ----------
        attr : str
            The attribute name to be accessed.

        Returns
        -------
        Any
            The attribute value from the underlying sampler.

        Raises
        ------
        AttributeError
            If the attribute is not found in both the MCMCSampler instance
            and the underlying sampler.
        """
        if "sampler" in self.__dict__ and hasattr(self.sampler, attr):
            return getattr(self.sampler, attr)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, name: str, value) -> None:
        """
        Delegate attribute setting to the underlying sampler if the attribute
        exists in the sampler and not in the MCMCSampler instance.

        Parameters
        ----------
        name : str
            The attribute name to be set.
        value : Any
            The value to be set to the attribute.
        """
        if (
            name in self.__dict__
            or "sampler" not in self.__dict__
            or not hasattr(self.sampler, name)
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self.sampler, name, value)
