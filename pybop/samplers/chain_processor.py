import numpy as np


class ChainProcessor:
    """
    Base class for chain processing.

    This clas architecture implements a strategy-pattern for selection
    between multi-chain and single-chain samplers as implemented
    in child classes.

    Parameters
    ----------
    mcmc_sampler : pybop.BasePintsSampler
        A BasePintsSampler object.
    """

    def __init__(self, mcmc_sampler):
        self.sampler = mcmc_sampler

    def process_chain(self):
        """Process the chain"""
        raise NotImplementedError

    def _extract_log_pdf(self, fy_value, chain_idx):
        """Extract log-pdf for an accepted sample."""
        raise NotImplementedError

    def store_samples(self, values, chain_idx):
        """
        Store samples based on memory configuration.
        Samples shape: [n_chains, n_iterations, n_parameters]
        """
        if self.sampler.chains_in_memory:
            # Create the index array using the `np.s_` slice method
            idx = (
                np.s_[chain_idx, self.sampler.n_samples[chain_idx]]
                if self.sampler.single_chain
                else np.s_[:, self.sampler.iteration]
            )
        else:
            # If not storing, direct assignment with appropriate slicing
            idx = np.s_[chain_idx] if self.sampler.single_chain else np.s_[:]

        self.sampler._samples[idx] = values  # noqa: SLF001

    def update_accepted_sample(self, chain_idx, y, fy_value):
        """
        Update stored values for an accepted sample.
        """
        log_pdf = self._extract_log_pdf(fy_value, chain_idx)
        self.sampler.sampled_logpdf[chain_idx] = log_pdf

        if self.sampler.prior:
            self.sampler.sampled_prior[chain_idx] = self.sampler.prior(y)

    def get_evaluation_metrics(self, chain_idx):
        """
        Get evaluation metrics for the current sample.
        """
        e = self.sampler.sampled_logpdf[chain_idx]

        if self.sampler.prior:
            e = [
                e,  # Log posterior
                self.sampler.sampled_logpdf[chain_idx]
                - self.sampler.sampled_prior[chain_idx],  # Log likelihood
                self.sampler.sampled_prior[chain_idx],  # Log prior
            ]

        return e


class SingleChainProcessor(ChainProcessor):
    """
    Processor for individual chains.
    """

    def __init__(self, mcmc_sampler):
        super().__init__(mcmc_sampler)

    def process_chain(self):
        self.sampler.fxs_iterator = iter(self.sampler.fxs)
        for i in list(self.sampler.active):
            reply = self.sampler.samplers[i].tell(next(self.sampler.fxs_iterator))
            if not reply:
                continue

            y, fy, accepted = reply
            y_store = self.sampler.transform_values(y)

            # Store samples
            self.store_samples(y_store, i)

            if accepted:
                self.update_accepted_sample(i, y, fy)

            # Store evaluation results
            e = self.get_evaluation_metrics(i)
            self.sampler._evaluations[i][self.sampler.n_samples[i]] = e  # noqa: SLF001

            # Increment sample counter and check if chain is complete
            self.sampler.n_samples[i] += 1
            if self.sampler.n_samples[i] == self.sampler.max_iterations:
                self.sampler.active.remove(i)

    def _extract_log_pdf(self, fy_value, chain_idx):
        """
        Extract log PDF value for single chain mode.
        """
        return fy_value[0] if self.sampler.needs_sensitivities else fy_value


class MultiChainProcessor(ChainProcessor):
    """
    Processor for simultaneous chains.
    """

    def __init__(self, mcmc_sampler):
        super().__init__(mcmc_sampler)

    def process_chain(self):
        reply = self.sampler.samplers[0].tell(self.sampler.fxs)
        self.sampler._intermediate_step = reply is None  # noqa: SLF001

        if reply:
            ys, fys, accepted = reply
            ys_store = np.asarray(self.sampler.transform_list_of_values(ys))

            # Store samples
            self.store_samples(ys_store, self.sampler.iteration)

            # Loop across chain's and store results
            for i, y in enumerate(ys):
                if accepted[i]:
                    self.update_accepted_sample(i, y, fys)

                # Get evaluations and store
                e = self.get_evaluation_metrics(i)
                self.sampler._evaluations[i, self.sampler.iteration] = e  # noqa: SLF001

    def _extract_log_pdf(self, fy_value, chain_idx):
        """
        Extract log PDF value for multi-chain mode.
        """
        return (
            fy_value[0][chain_idx]
            if self.sampler.needs_sensitivities
            else fy_value[chain_idx]
        )
