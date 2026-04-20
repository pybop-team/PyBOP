import json
import time
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from sys import stderr, stdout

import numpy as np
from pybamm import citations

from pybop import plot
from pybop._logging import Logger
from pybop.optimisers.base_optimiser import (
    BaseOptimiser,
    OptimisationResult,
    OptimiserOptions,
)
from pybop.parameters.multivariate_distributions import (
    MarginalDistribution,
    MultivariateGaussian,
    MultivariateLogNormal,
)
from pybop.parameters.parameter import Parameter, Parameters
from pybop.problems.meta_problem import MetaProblem
from pybop.problems.problem import Problem
from pybop.processing.dataset import Dataset


@dataclass
class EPBOLFIOptions(OptimiserOptions):
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

    # Parameter boundaries in the model space. Samples are not taken
    # from outside these boundaries.
    model_parameter_boundaries: dict = field(default_factory=dict)

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
    upon instantiation of this class. To change attributes, re-init.

    Only compatible with MultivariateParameters with a MultivariateGaussian
    distribution, or a MultivariateLogNormal with a LogTransformation.
    """

    def __init__(
        self,
        problem: Problem,
        options: EPBOLFIOptions | None = None,
    ):
        if type(problem) is not MetaProblem:
            problem = MetaProblem(problem)
        super().__init__(problem, options)
        # citations.register("""@article{
        #     Minka2013,
        #     title={{Expectation Propagation for approximate Bayesian inference}},
        #     author={Minka, T},
        #     journal={Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence (UAI2001)},
        #     pages={362-369},
        #     year={2013},
        #     doi={10.48550/arXiv.1301.2294}
        # }""")
        citations.register("""@article{
            Barthelme2014,
            title={{Expectation propagation for likelihood-free inference}},
            author={Barthelmé, S and Chopin, N},
            journal={Journal of the American Statistical Association},
            volume={109},
            pages={315-333},
            year={2014},
            doi={10.1080/01621459.2013.864178}
        }""")
        citations.register("""@article{
            Gutmann2016,
            title={{Bayesian optimization for likelihood-free inference of simulator-based statistical models}},
            author={Gutmann, M and Corander, J},
            journal={Journal of Machine Learning Research},
            volume={17},
            pages={1-47},
            year={2016},
            doi={arXiv.1501.03291}
        }""")
        citations.register("""@article{
            Kuhn2022,
            title={{Bayesian Parameterization of Continuum Battery Models from Featurized Electrochemical Measurements Considering Noise}},
            author={Kuhn, Y and Wolf, H and Latz, A and Horstmann, B},
            journal={Batteries & Supercaps},
            volume={6},
            pages={e202200374},
            year={2023},
            publisher={Chemistry Europe},
            doi={10.1002/batt.202200374}
        }""")

    def _set_up_optimiser(self):
        import ep_bolfi

        # Define separate simulators for multiple target variables.
        simulators = [
            lambda inputs, problem=problem: problem.simulate(inputs=inputs)
            for problem in self.problem.problems
        ]
        experimental_datasets = [
            Dataset(problem.target_data, domain=problem.domain)
            for problem in self.problem.problems
        ]
        feature_extractors = [
            lambda solution, problem=problem: [
                problem.cost.evaluate(solution=solution).values
            ]
            for problem in self.problem.problems
        ]
        self.optimiser = ep_bolfi.EP_BOLFI(
            simulators,
            experimental_datasets,
            feature_extractors,
            fixed_parameters={},  # probably baked into each problem.simulator
            free_parameters={
                name: par.get_mean(transformed=True)
                for name, par in self.problem.parameters.items()
            },
            initial_covariance=self.problem.parameters.get_covariance(transformed=True),
            free_parameters_boundaries=self._options.model_parameter_boundaries,
            boundaries_in_deviations=self._options.boundaries_in_standard_deviations,
            Q=self._options.precision_matrix,
            r=self._options.covariance_scaled_mean,
            Q_features=self._options.precision_matrices_per_feature,
            r_features=self._options.covariance_scaled_means_per_feature,
            transform_parameters={
                name: (par.transformation.to_model, par.transformation.to_search)
                for name, par in self.problem.parameters.items()
            },
            weights=None,  # only applicable within vector-valued features and better handled within PyBOP costs
            display_current_feature=None,  # ToDo: costs with names
            fixed_parameter_order=list(enumerate(self.problem.parameters.keys())),
        )
        assert self.problem.minimising is True
        self._logger = Logger(minimising=self.problem.minimising, verbose=False)

    def _run(self) -> "BayesianOptimisationResult":
        verbose_log_target = stdout if self._options.verbose else None
        verbose_err_target = stderr if self._options.verbose else None
        with redirect_stdout(verbose_log_target):
            with redirect_stderr(verbose_err_target):
                start = time.time()
                # bolfi_posterior is the full GPy object containing the state at
                # the end of the last feature iteration, while the
                # MultivariateGaussian is a slight approximation.
                total_samples = (
                    None
                    if self._options.bolfi_initial_sobol_samples is None
                    or self._options.bolfi_optimally_acquired_samples is None
                    else self._options.bolfi_initial_sobol_samples
                    + self._options.bolfi_optimally_acquired_samples
                )
                self.bolfi_posterior = self.optimiser.run(
                    bolfi_initial_evidence=self._options.bolfi_initial_sobol_samples,
                    bolfi_total_evidence=total_samples,
                    bolfi_posterior_samples=self._options.bolfi_posterior_effective_sample_size,
                    ep_iterations=self._options.ep_iterations,
                    ep_dampener=self._options.ep_stepwise_dampener,
                    final_dampening=self._options.ep_total_dampening,
                    ep_dampener_reduction_steps=-1,  # better re-init with another dampening factor
                    gelman_rubin_threshold=self._options.posterior_gelman_rubin_threshold,
                    ess_ratio_resample=self._options.posterior_ess_ratio_threshold_resampling,
                    ess_ratio_sampling_from_zero=self._options.posterior_ess_ratio_threshold_evaluation_at_centre,
                    ess_ratio_abort=self._options.posterior_ess_ratio_threshold_skip_feature,
                    max_heuristic_steps=self._options.max_posterior_sampling_retries,
                    posterior_sampling_increase=self._options.posterior_actual_sample_size_increase,
                    model_resampling_increase=self._options.posterior_model_resample_size_increase,
                    independent_mcmc_chains=4,  # 4 generally works well
                    scramble_ep_feature_order=self._options.ep_randomise_feature_order,
                    normalize_features=False,  # does not work when features assume 0, normalise within PyBOP
                    show_trials=False,  # use the PyBOP visualization tools instead
                    verbose=self._options.verbose,
                    seed=self._options.seed,
                )
                end = time.time()
        ep_bolfi_result = json.loads(
            self.optimiser.result_to_json(seed=self._options.seed)
        )

        # Get the optimiser log
        ep_bolfi_log = json.loads(self.optimiser.log_to_json())
        x_list = np.array(list(ep_bolfi_log["tried parameters"].values())).T
        # Collect all features into one cost. Note: they are logarithms,
        # so this is a multiplicative combination.
        feature_costs = np.array(list(ep_bolfi_log["discrepancies"].values()))
        cost_list = deepcopy(feature_costs[0])
        for i in range(1, len(feature_costs)):
            for j in range(len(cost_list)):
                cost_list[j][0] += feature_costs[i][j][0]
        cost_list = np.array([np.exp(value[0]) for value in cost_list])
        x_best_over_time = deepcopy(x_list)
        cost_best = deepcopy(cost_list)
        for i in range(1, len(cost_list)):
            if cost_list[i] < cost_best[i - 1]:
                x_best_over_time[i:, None] = x_list[i, None]
                cost_best[i:] = cost_list[i]

        self._logger.x_model = x_list.tolist()
        self._logger.x_search = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values(), strict=False)
            ]
            for entry in x_list
        ]
        self._logger.cost = cost_list
        self._logger.iterations = [
            i // (self._options.ep_iterations * len(feature_costs))
            for i in range(len(cost_list))
        ]
        self._logger.evaluations = [i + 1 for i in range(len(cost_list))]
        self._logger.x_model_best = x_best_over_time[-1]
        x_search_best_over_time = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values(), strict=False)
            ]
            for entry in x_best_over_time
        ]
        self._logger.x_search_best = x_search_best_over_time[-1]
        self._logger.cost_best = cost_best[-1]
        self._logger.iteration = {
            "EP iterations": self._options.ep_iterations,
            "total feature iterations": self._options.ep_iterations
            * len(self.problem.problems),
        }
        self._logger.evaluations = {
            "model evaluations": len(
                list(ep_bolfi_log["tried parameters"].values())[0]
            ),
            # "surrogate evaluations" are not directly accessible
        }

        # Get the mean and the 95% confidence error bounds
        model_mean = np.array(
            [val[0] for val in ep_bolfi_result["inferred parameters"].values()]
        )
        lower_bounds = np.array(
            [bounds[0][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )
        upper_bounds = np.array(
            [bounds[1][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )

        # Create the posterior distribution, using the existing parameter transformations
        n = len(self.problem.parameters)
        if isinstance(self.problem.parameters.distribution, MultivariateLogNormal):
            covariance_log_x = np.array(ep_bolfi_result["covariance"])
            mean_log_x = np.zeros(n)
            for i in range(n):
                mean_log_x[i] = np.log(model_mean[i]) - 0.5 * covariance_log_x[i, i]
            posterior_distribution = MultivariateLogNormal(
                mean_log_x=mean_log_x, covariance_log_x=covariance_log_x
            )
        else:
            posterior_distribution = MultivariateGaussian(
                mean=model_mean, covariance=np.array(ep_bolfi_result["covariance"])
            )
        posterior_parameters = {
            key: Parameter(
                distribution=MarginalDistribution(posterior_distribution, i),
                initial_value=p.initial_value,
                transformation=p.transformation,
            )
            for i, (key, p) in enumerate(self.problem.parameters.items())
        }

        return BayesianOptimisationResult(
            optim=self,
            time=end - start,
            method_name="EP-BOLFI",
            posterior=Parameters(posterior_parameters),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def name(self):
        return (
            "Expectation Propagation with Bayesian Optimization for "
            "Likelihood-Free Inference"
        )


class BayesianOptimisationResult(OptimisationResult):
    """
    Stores the result of a Bayesian optimisation or a Bayesian model
    selection.

    Attributes
    ----------
    problem: pybop.Problem
        The optimisation problem used to generate the results.
    x : ndarray
        The solution of the optimisation (in model space).
    best_cost : float
        The cost associated with the solution x.
    n_iterations : int or dict
        Number of iterations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their iteration counts may be put individually.
    n_evaluations : int or dict
        Number of evaluations performed by the optimiser. Since Bayesian
        optimisers tend to have layers of various optimisation
        algorithms, their evaluation counts my be put individually.
    message : str
        The reason for stopping given by the optimiser.
    lower_bounds: ndarray
        The lower confidence parameter boundaries.
    upper_bounds: ndarray
        The upper confidence parameter boundaries.
    posterior : MultivariateParameters
        The probability distribution of the optimisation.
    maximum_a_posteriori : Inputs or ndarray
        Complementing the best observed value in `x`, this is the
        prediction for the best parameter value.
    log_evidence_mean : float
        The logarithm of the evidence of the parameterization. Higher
        values are better. May only be interpreted relative to a
        calibration case, e.g., a test-run with synthetic data.
    log_evidence_variance : float
        The logarithm of the variance in the calculation of the
        evidence. For reliable comparisons based on the evidence, should
        be at or below the scale of the evidence itself.
    """

    def __init__(
        self,
        optim: EP_BOLFI,
        time: float | dict,
        method_name: str | None = None,
        message: str | None = None,
        lower_bounds: np.ndarray | None = None,
        upper_bounds: np.ndarray | None = None,
        posterior: Parameters | None = None,
        maximum_a_posteriori: np.ndarray | None = None,
        log_evidence_mean: float | None = None,
        log_evidence_variance: float | None = None,
    ):
        super().__init__(
            optim=optim,
            time=time,
            method_name=method_name,
            message=message,
        )
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.posterior = posterior
        self.maximum_a_posteriori = maximum_a_posteriori
        self.log_evidence_mean = log_evidence_mean
        self.log_evidence_variance = log_evidence_variance

    def plot_predictive(self, **kwargs):
        """
        Plot the predictive posterior of a Bayesian parameterisation result.
        """
        return plot.predictive(result=self, **kwargs)
