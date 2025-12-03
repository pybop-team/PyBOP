import copy
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from sys import stderr, stdout

import numpy as np
from pybamm import citations

import pybop
from pybop import BaseOptimiser
from pybop._logging import Logger
from pybop._result import BayesianOptimisationResult
from pybop.parameters.multivariate_distributions import MultivariateGaussian


@dataclass
class EPBOLFIOptions(pybop.OptimiserOptions):
    """
    A class to hold EP-BOLFI options for the optimisation process.

    For detailed descriptions of the options, consult the EP-BOLFI
    documentation at https://github.com/YannickNoelStephanKuhn/EP-BOLFI;
    you'll find its PDF attached to the newest release. Note that the
    variable names have been rewritten for clarity here, so you'll have
    to look up their "original" names in the `ep_bolfi.EP_BOLFI`
    constructor call within the `_set_up_optimiser` routine of the
    `pybop.EP_BOLFI` class.
    """

    # Since the performance of Bayesian optimisers has an optimum around
    # a problem-specific sample size, rather than increasing
    # monotonically with it, no stopping criteria are available to set
    # here. Trial and error from the default values, which are set as
    # the lowest reasonable ones.
    parallel: bool = False
    seed: int = 0

    # Each EP iterations consists of one pass over every feature.
    ep_iterations: int = 1
    ep_randomise_feature_order: bool = False

    # Dampening refers to updating the prior with only a fraction of
    # the likelihood. May be given stepwise per feature update which
    # over the EP iterations quickly leads to almost complete dampening,
    # or calculated to match a final effective fraction of prior update.
    # Defaults to calculating a final effective fraction of 0.5.
    ep_stepwise_dampener: float | None = None
    ep_total_dampening: float | None = None

    # Adjusts the hard parameter boundaries relative to their
    # standard deviations. Defaults to 95 % confidence regions.
    boundaries_in_standard_deviations: int = 0

    # Internal state variables of the optimisation process. May be
    # copied from a previous run of EP_BOLFI to continue it with
    # more EP iterations.
    precision_matrix: np.ndarray | None = None
    covariance_scaled_mean: np.ndarray | None = None
    precision_matrices_per_feature: np.ndarray | None = None
    covariance_scaled_means_per_feature: np.ndarray | None = None

    # Samples taken by BOLFI per feature update. The initial Sobol
    # samples give an unsupervised initial base for BOLFI to then
    # acquire more samples based on optimal sampling efficiency.
    # With N = len(parameters), the default values are:
    # - Initial sample default: 1 + 2**N.
    # - Total (initial + acquired) sample default: 2 (1 + 2**N).
    bolfi_initial_sobol_samples: int | None = None
    bolfi_optimally_acquired_samples: int | None = None

    # Target quality / "Dimensionality" of the sampling to approximate
    # the posterior with. Defaults to N² + 3N with N = len(parameters).
    bolfi_posterior_effective_sample_size: int | None = None

    # Settings for heuristics that EP-BOLFI may employ to self-correct
    # in case of poor convergence. Effective Sample Size (ESS) and the
    # Gelman-Rubin Statistic are popular convergence heuristics for
    # Markov Chain Monte Carlo (MCMC) approaches. Here, the MCMC variant
    # NUTS is used. Based on the set thresholds for the ratio of actual
    # sample size to ESS, EP-BOLFI may try to incorporate more model
    # evaluations, try to start NUTS at the initial parameter values
    # rather than the best predicted ones, or skip to the next feature.
    posterior_actual_sample_size_increase: float = 1.2
    posterior_ess_ratio_threshold_resampling: int = 5
    posterior_model_resample_size_increase: float = 1.1
    posterior_ess_ratio_threshold_evaluation_at_centre: int = -1
    posterior_ess_ratio_threshold_skip_feature: int = -1
    posterior_gelman_rubin_threshold: float | None = None
    max_posterior_sampling_retries: int = 10

    def validate(self):
        super().validate()

        if self.parallel:
            raise ValueError(
                "EP-BOLFI is not parallelisable by design for sample "
                "efficiency. Use SOBER instead for parallelisation."
            )

        if (
            self.ep_stepwise_dampener is not None
            and (self.ep_stepwise_dampener < 0 or self.ep_stepwise_dampener >= 1)
            or (
                self.ep_total_dampening is not None
                and (self.ep_total_dampening < 0 or self.ep_total_dampening >= 1)
            )
        ):
            raise ValueError(
                "The EP dampening has to be a positive number smaller than 1."
            )

        if self.boundaries_in_standard_deviations < 0:
            raise ValueError(
                "Hard parameter boundaries can't be negative multiples of σ."
            )

        if (
            self.bolfi_initial_sobol_samples is not None
            and self.bolfi_initial_sobol_samples < 0
        ):
            raise ValueError(
                "Initial Sobol parameter samples can not be a negative number."
            )

        if (
            self.bolfi_optimally_acquired_samples is not None
            and self.bolfi_optimally_acquired_samples < 0
        ):
            raise ValueError(
                "Optimally acquired parameter samples can not be a negative number."
            )

        if (
            self.bolfi_posterior_effective_sample_size is not None
            and self.bolfi_posterior_effective_sample_size < 0
        ):
            raise ValueError(
                "Effective Sample Size for posterior evaluation can not be a negative number."
            )

        if self.posterior_actual_sample_size_increase <= 1:
            raise ValueError(
                "The factor by which to increase posterior samples has to be greater than 1."
            )

        if self.posterior_model_resample_size_increase <= 1:
            raise ValueError(
                "The factor by which to increase model samples has to be greater than 1."
            )

        if (
            self.posterior_gelman_rubin_threshold is not None
            and self.posterior_gelman_rubin_threshold <= 1
        ):
            raise ValueError(
                "The Gelman-Rubin threshold has to be a number greater than 1."
            )


class EP_BOLFI(BaseOptimiser):
    """
    Wraps the Bayesian Optimization algorithm EP-BOLFI.

    For implementation details and background information, consult the
    relevant publication at https://doi.org/10.1002/batt.202200374 and
    visit https://github.com/YannickNoelStephanKuhn/EP-BOLFI.

    Note that all properties may and should be given here as PyBOP
    objects, but will be converted to an ep_bolfi.EP_BOLFI instance
    upon instantation of this class. To change attributes, re-init.

    Only compatible with MultivariateParameters with
    MultivariateGaussian prior and a MetaProblem.
    """

    def __init__(
        self,
        problem: pybop.MetaProblem,
        options: EPBOLFIOptions | None = None,
    ):
        super().__init__(problem, options)
        # citations.register("""@article{
        #     Minka2013,
        #     title={{Expectation Propagation for approximate Bayesian inference}},
        #     author={Minka, T},
        #     journal={Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence (UAI2001)},
        #     pages={362-369},
        #     year={2013}
        # }""")
        citations.register("""@article{
            Barthelme2014,
            title={{Expectation propagation for likelihood-free inference}},
            author={Barthelmé, S and Chopin, N},
            journal={Journal of the American Statistical Association},
            volume={109},
            pages={315-333},
            year={2014}
        }""")
        citations.register("""@article{
            Gutmann2016,
            title={{Bayesian optimization for likelihood-free inference of simulator-based statistical models}},
            author={Gutmann, M and Corander, J},
            journal={Journal of Machine Learning Research},
            volume={17},
            pages={1-47},
            year={2016}
        }""")
        citations.register("""@article{
            Kuhn2022,
            title={{Bayesian Parameterization of Continuum Battery Models from Featurized Electrochemical Measurements Considering Noise}},
            author={Kuhn, Y and Wolf, H and Latz, A and Horstmann, B},
            journal={Batteries & Supercaps},
            volume={6},
            pages={e202200374},
            year={2023},
            publisher={Chemistry Europe}
        }""")

    def _set_up_optimiser(self):
        import ep_bolfi

        transposed_boundaries = {}
        model_bounds = self.problem.parameters.get_bounds(transformed=False)  # noqa: SLF001
        for i, name in enumerate(self.problem.parameters.keys()):  # noqa: SLF001
            transposed_boundaries[name] = [
                model_bounds["lower"][i],
                model_bounds["upper"][i],
            ]
        simulators = [problem._simulator for problem in self.problem.problems]  # noqa: SLF001
        experimental_datasets = [
            problem.target_data for problem in self.problem.problems
        ]  # noqa: SLF001
        feature_extractors = [
            lambda y: [problem._cost(y)] for problem in self.problem.problems
        ]  # noqa: SLF001
        self.optimiser = ep_bolfi.EP_BOLFI(
            simulators,
            experimental_datasets,
            feature_extractors,
            fixed_parameters={},  # probably baked into self.problem.model
            free_parameters={
                name: par.get_initial_value_transformed()
                for name, par in self.problem.parameters.items()  # noqa: SLF001
            },
            initial_covariance=self.problem.parameters.distribution.properties["cov"],  # noqa: SLF001
            free_parameters_boundaries=transposed_boundaries,
            boundaries_in_deviations=self._options.boundaries_in_standard_deviations,
            Q=self._options.precision_matrix,
            r=self._options.covariance_scaled_mean,
            Q_features=self._options.precision_matrices_per_feature,
            r_features=self._options.covariance_scaled_means_per_feature,
            transform_parameters={
                name: (par.transformation.to_model, par.transformation.to_search)
                for name, par in self.problem.parameters.items()  # noqa: SLF001
            },
            weights=None,  # only applicable within vector-valued features and better handled within PyBOP costs
            display_current_feature=None,  # ToDo: costs with names
            fixed_parameter_order=list(enumerate(self.problem.parameters.keys())),  # noqa: SLF001
        )
        self._logger = Logger(
            minimising=True,
            verbose=self.verbose,
            verbose_print_rate=self.verbose_print_rate,
        )

    def _run(self):
        verbose_log_target = stdout if self._options.verbose else None
        verbose_err_target = stderr if self._options.verbose else None
        with redirect_stdout(verbose_log_target):
            with redirect_stderr(verbose_err_target):
                start = time.time()
                # bolfi_posterior is the full GPy object containing the state at
                # the end of the last feature iteration, while the
                # MultivariateGaussian is a slight approximation.
                self.bolfi_posterior = self.optimiser.run(
                    self._options.bolfi_initial_sobol_samples,
                    self._options.bolfi_initial_sobol_samples
                    + self._options.bolfi_optimally_acquired_samples,
                    self._options.bolfi_posterior_effective_sample_size,
                    self._options.ep_iterations,
                    self._options.ep_stepwise_dampener,
                    self._options.ep_total_dampening,
                    -1,  # ep_dampener_reduction_steps; better re-init with another dampening factor
                    self._options.posterior_gelman_rubin_threshold,
                    self._options.posterior_ess_ratio_threshold_resampling,
                    self._options.posterior_ess_ratio_threshold_evaluation_at_centre,
                    self._options.posterior_ess_ratio_threshold_skip_feature,
                    self._options.max_posterior_sampling_retries,
                    self._options.posterior_actual_sample_size_increase,
                    self._options.posterior_model_resample_size_increase,
                    4,  # independent_mcmc_chains; 4 generally works well
                    self._options.ep_randomise_feature_order,
                    False,  # normalize_features; does not work when features assume 0, normalise within PyBOP
                    False,  # show_trials; use the PyBOP visualization tools instead
                    self._options.verbose,
                    self._options.seed,
                )
                end = time.time()
        ep_bolfi_result = json.loads(
            self.optimiser.result_to_json(seed=self._options.seed)
        )
        ep_bolfi_log = json.loads(self.optimiser.log_to_json())
        x_list = np.array(list(ep_bolfi_log["tried parameters"].values())).T
        # Collect all features into one cost. Note: they are logarithms,
        # so this is a multiplicative combination.
        feature_costs = np.array(list(ep_bolfi_log["discrepancies"].values()))
        cost_list = copy.deepcopy(feature_costs[0])
        for i in range(1, len(feature_costs)):
            for j in range(len(cost_list)):
                cost_list[j][0] += feature_costs[i][j][0]
        cost_list = np.array([np.exp(value[0]) for value in cost_list])
        x_best = copy.deepcopy(x_list)
        cost_best = copy.deepcopy(cost_list)
        for i in range(1, len(cost_list)):
            if cost_list[i] < cost_best[i - 1]:
                x_best[i:, None] = x_list[i, None]
                cost_best[i:] = cost_list[i]

        self._logger.x_model = x_list.tolist()
        self._logger.x_search = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values(), strict=False)  # noqa: SLF001
            ]
            for entry in x_list
        ]
        self._logger.cost = cost_list
        self._logger.iterations = [
            i // (self._options.ep_iterations * len(feature_costs))
            for i in range(len(cost_list))
        ]
        self._logger.evaluations = [i + 1 for i in range(len(cost_list))]
        self._logger.x_model_best = x_best
        self._logger.x_search_best = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values(), strict=False)  # noqa: SLF001
            ]
            for entry in x_best
        ]
        self._logger.cost_best = cost_best[0]
        model_mean_dict = {
            key: value[0]
            for key, value in ep_bolfi_result["inferred parameters"].items()
        }
        model_mean_array = np.array(list(model_mean_dict.values()))
        search_mean_array = [
            par.transformation.to_search(entry)[0]
            for entry, par in zip(
                model_mean_array,
                self.problem.parameters.values(),  # noqa: SLF001
                strict=False,
            )
        ]
        lower_bounds = np.array(
            [bounds[0][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )
        upper_bounds = np.array(
            [bounds[1][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )
        # The re-use of `parameters` makes transformations easily usable.
        posterior = copy.deepcopy(self.problem.parameters)  # noqa: SLF001
        posterior.prior = MultivariateGaussian(
            search_mean_array, np.array(ep_bolfi_result["covariance"])
        )
        self._logger.iteration = {
            "EP iterations": self._options.ep_iterations,
            "total feature iterations": self._options.ep_iterations
            * len(self.problem.problems),  # noqa: SLF001
        }
        self._logger.evaluations = {
            "model evaluations": len(
                list(ep_bolfi_log["tried parameters"].values())[0]
            ),
            # "surrogate evaluations" are not directly accessible
        }
        return BayesianOptimisationResult(
            optim=self,
            logger=self._logger,
            time=end - start,
            optim_name="EP-BOLFI",
            posterior=posterior,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def name(self):
        return (
            "Expectation Propagation with Bayesian Optimization for "
            "Likelihood-Free Inference"
        )
