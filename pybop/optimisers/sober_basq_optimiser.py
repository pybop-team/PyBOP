import copy
from dataclasses import dataclass
import time
import torch
from contextlib import redirect_stderr, redirect_stdout
from sys import stderr, stdout

from collections.abc import Callable

import numpy as np
from os import cpu_count
from pybamm import citations
from torch import tensor

import pybop
from pybop import BaseOptimiser
from pybop._logging import Logger
from pybop._result import BayesianOptimisationResult


@dataclass
class SOBER_BASQ_Options(pybop.OptimiserOptions):
    """
    A class to hold SOBER and BASQ options for the optimisation as well
    as the optional model selection process.
    """

    model_initial_samples: int = cpu_count()
    maximise: bool = False
    set_up_parabolic_hyperparameters: bool = False
    weights: np.ndarray | None = None
    custom_objective_and_loglikelihood: Callable | None = None
    seed: int | None = None
    batched_input: bool = False

    sober_iterations: int = 1
    model_samples_per_iteration: int = cpu_count()

    integration_nodes: int | None = cpu_count()

    normalise_evidence: bool = True

    verbose: bool = False

    def validate(self):
        super().validate()


class SOBER_BASQ(BaseOptimiser):
    """
    Wraps the Bayesian Optimization algorithm SOBER. SOBER includes the
    Bayesian Model Selection algorithm BASQ.

    For implementation details and background information, consult the
    relevant publications at https://doi.org/10.48550/arXiv.2404.12219
    and https://proceedings.neurips.cc/paper_files/paper/2022/file/697200c9d1710c2799720b660abd11bb-Paper-Conference.pdf
    and visit https://github.com/ma921/SOBER/.

    Note that all properties may and should be given here as PyBOP
    objects, but will be converted to sober.SoberWrapper instance
    upon instantiation of this class. To change attributes, re-init.

    Only compatible with MulitvariateParameters and multivariate priors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        citations.register("""@article{
            Adachi2024,
            title={{A Quadrature Approach for General-Purpose Batch Bayesian Optimization via Probabilistic Lifting}},
            author={Adachi, M and Hawakawa, S and Jørgensen, M, and Hamid, S and Oberhauser, H and Osborne, M},
            journal={arXiv},
            year={2024},
            doi={10.48550/arXiv.2404.12219}
        }""")
        citations.register("""@article{
            Adachi2022,
            title={{Fast Bayesian Inference with Batch Bayesian Quadrature via Kernel Recombination}},
            author={Adachi, M and Hayakawa, S and Jørgensen, M and Oberhauser, H and Osborne, M},
            journal={Advances in Neural Information Processing Systems},
            volume={35},
            pages={16533-16547},
            year={2022}
        }""")
    
    def evaluate_problem(self, inputs_array):
        sim = self.problem._simulator
        inputs = {k: i for k, i in zip(self.problem.parameters.keys(), inputs_array.T)}
        return sim.solve(inputs)[sim.output_variables[0]].data

    def _set_up_optimiser(self, **kwargs):
        import sober

        prior = self.problem.parameters.distribution

        self.mean = prior.properties.get("mean")
        if self.mean is not None:
            self.mean = torch.tensor(self.mean)
        self.covariance = prior.properties.get("cov")
        self.bounds = prior.properties.get("bounds")
        if self.bounds is None and hasattr(prior, "bounds"):
            self.bounds = prior.bounds

        if isinstance(prior, pybop.MultivariateUniform):
            self.prior_name = "Uniform"
        elif isinstance(prior, pybop.MultivariateGaussian):
            self.prior_name = "TruncatedGaussian"
        else:
            raise ValueError(
                "The provided prior must be a multivariate uniform or multivariate Gaussian one."
            )

        # ToDo: generalise to other transformations (has to be PyTorch,
        # else the vmap-vectorisation for that within SoberWrapper fails).
        self.transform_parameters = [
            (
                torch.log, torch.exp
            ) if isinstance(par.transformation, pybop.LogTransformation) else (
                torch.nn.Identity(), torch.nn.Identity()
            )
            for par in self.problem.parameters.values()
        ]

        # ToDo: can only use one problem function for now, else multiprocessing breaks.
        if isinstance(self.problem.target_data, dict):
            target_data = tensor(np.asarray(list(self.problem.target_data.values())[0]))
        else:
            target_data = tensor(self.problem.target_data)
        self.optimiser = sober.SoberWrapper(
            self.evaluate_problem,
            target_data,
            self._options.model_initial_samples,
            self.mean,
            None if self.covariance is None else tensor(self.covariance),
            None if self.bounds is None else tensor(self.bounds).T,
            self.prior_name,
            self._options.maximise,
            self._options.set_up_parabolic_hyperparameters,
            self._options.weights,
            self._options.custom_objective_and_loglikelihood,
            self.transform_parameters,
            self._options.seed,
            False,  # disable_numpy_mode
            not self._options.batched_input,  # parallelization (i.e., False has to parallelise itself)
            False,  # visualizations,
            None,  # true_optimum,
            True,  # standalone
            None,  # names,
        )
        self._logger = Logger(
            minimising=not self._options.maximise,
            verbose=self._options.verbose,
            verbose_print_rate=self.verbose_print_rate,
        )

    def _run(self):
        verbose_log_target = stdout if self._options.verbose else None
        verbose_err_target = stderr if self._options.verbose else None
        with redirect_stdout(verbose_log_target):
            with redirect_stderr(verbose_err_target):
                start = time.time()
                self.optimiser.run_SOBER(
                    self._options.sober_iterations,
                    self._options.model_samples_per_iteration,
                    visualizations=False,
                    verbose=self._options.verbose,
                )
                (
                    raw_taken_samples,
                    MAP,
                    best_observed,
                    log_expected_marginal_likelihood,
                    log_approx_variance_marginal_likelihood,
                ) = self.optimiser.run_BASQ(
                    self._options.integration_nodes,
                    return_raw_samples=True,
                    visualizations=False,
                    verbose=self._options.verbose,
                )
                end = time.time()

        x_list = self.optimiser.X_all.numpy()
        cost_list = self.optimiser.Y_all.numpy()
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
                for e, par in zip(entry, self.problem.parameters.values())
            ]
            for entry in x_list
        ]
        self._logger.cost = cost_list
        self._logger.iterations = [i // (self._options.sober_iterations) for i in range(len(cost_list))]
        self._logger.evaluations = [i + 1 for i in range(len(cost_list))]
        self._logger.x_model_best = x_best[-1]
        x_search_best_over_time = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values())
            ]
            for entry in x_best
        ]
        self._logger.x_search_best = x_search_best_over_time[-1]
        self._logger.cost_best = cost_best[-1]
        posterior = pybop.MultivariateParameters(
            self.problem.parameters,
            distribution=pybop.MultivariateNonparametric(
                self.optimiser.denormalize_input(raw_taken_samples).T
            )
        )

        self._logger.iteration = {"SOBER iterations": self._options.sober_iterations}
        self._logger.evaluations = {"model evaluations": self._options.sober_iterations * self._options.model_samples_per_iteration}

        if self._options.normalise_evidence:
            log_expected_marginal_likelihood *= (2 * np.pi)**(len(self.problem.parameters) / 2)

        return BayesianOptimisationResult(
            optim=self,
            logger=self._logger,
            time=end - start,
            optim_name="SOBER + BASQ",
            posterior=posterior,
            lower_bounds=None if self.bounds is None else self.bounds[:, 0],
            upper_bounds=None if self.bounds is None else self.bounds[:, 1],
            maximum_a_posteriori=MAP,
            log_evidence_mean=log_expected_marginal_likelihood,
            log_evidence_variance=log_approx_variance_marginal_likelihood,
        )

    def name(self):
        return (
            "Solving Optimisation as Bayesian Estimation via Recombination"
        )


@dataclass
class SOBER_BASQ_EPLFI_Options(SOBER_BASQ_Options):
    """
    Extends SOBER and BASQ options with EP-LFI-specific options.
    """

    ep_iterations: int = 2
    ep_total_dampening: float = 0.5
    ep_integration_nodes: int | None = cpu_count()

    def validate(self):
        super().validate()

        if self.maximise:
            raise ValueError("EP-LFI only minimises; consider negating the cost function.")
        if self.weights is not None:
            raise ValueError("EP-LFI performs automatic importance weighting; you can't set the weights.")
        if self.custom_objective_and_loglikelihood is not None:
            raise ValueError("EP-LFI builds a custom objective from the features already; you can't set your own.")


class SOBER_BASQ_EPLFI(SOBER_BASQ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _collect_functions(self, inputs_array):
        inputs = {k: i for k, i in zip(self.problem.parameters.keys(), inputs_array.T)}
        return np.array([
            problem._simulator.solve(inputs)[problem._simulator.output_variables[0]].data
            for problem in self.problem.problems
        ]).T

    def _features(self, y):
        # Since in PyBOP, costs are already part of the problems, just report y.
        return y

    def _set_up_optimiser(self, **kwargs):
        import sober

        prior = self.problem.parameters.distribution

        self.mean = prior.properties.get("mean")
        self.covariance = prior.properties.get("cov")
        self.bounds = prior.properties.get("bounds")
        if self.bounds is None and hasattr(prior, "bounds"):
            self.bounds = prior.bounds

        if isinstance(prior, pybop.MultivariateUniform):
            self.prior_name = "Uniform"
        elif isinstance(prior, pybop.MultivariateGaussian):
            self.prior_name = "TruncatedGaussian"
        else:
            raise ValueError(
                "The provided prior must be a multivariate uniform or multivariate Gaussian one."
            )

        # ToDo: generalise to other transformations (has to be PyTorch,
        # else the vmap-vectorisation for that within SoberWrapper fails).
        self.transform_parameters = [
            (
                torch.log, torch.exp
            ) if isinstance(par.transformation, pybop.LogTransformation) else (
                torch.nn.Identity(), torch.nn.Identity()
            )
            for par in self.problem.parameters.values()
        ]

        # ToDo: can only use one Python problem function for now, else
        # multiprocessing breaks.
        self.optimiser = sober.ExpectationPropagationLFI(
            self._collect_functions,
            tensor(np.asarray([list(problem.target_data.values())[0] for problem in self.problem.problems])),
            self._features,
            self._options.model_initial_samples,
            self.mean,
            self.covariance,
            None if self.bounds is None else tensor(self.bounds).T,
            self._options.set_up_parabolic_hyperparameters,
            self.transform_parameters,
            self._options.seed,
            False,  # disable_numpy_mode
            not self._options.batched_input,  # parallelization (i.e., False has to parallelise itself)
            False,  # visualizations,
            None,  # true_optimum,
        )
        self._logger = Logger(
            minimising=not self._options.maximise,
            verbose=self._options.verbose,
            verbose_print_rate=self.verbose_print_rate,
        )

    def _run(self):
        verbose_log_target = stdout if self._options.verbose else None
        verbose_err_target = stderr if self._options.verbose else None
        with redirect_stdout(verbose_log_target):
            with redirect_stderr(verbose_err_target):
                start = time.time()
                self.optimiser.run_Expectation_Propagation(
                    ep_iterations=self._options.ep_iterations,
                    final_dampening=self._options.ep_total_dampening,
                    sober_iterations=self._options.sober_iterations,
                    model_samples_per_iteration=self._options.model_samples_per_iteration,
                    integration_nodes=self._options.ep_integration_nodes,
                    visualizations=False,
                    verbose=False
                )
                (
                    raw_taken_samples,
                    MAP,
                    best_observed,
                    log_expected_marginal_likelihood,
                    log_approx_variance_marginal_likelihood,
                ) = self.optimiser.run_BASQ(
                    self._options.integration_nodes,
                    return_raw_samples=True,
                    visualizations=False,
                    verbose=False,
                )
                end = time.time()

        x_list = self.optimiser.X_all.numpy()
        cost_list = self.optimiser.Y_all.numpy()
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
                for e, par in zip(entry, self.problem.parameters.values())
            ]
            for entry in x_list
        ]
        self._logger.cost = cost_list
        self._logger.iterations = [i // (self._options.sober_iterations) for i in range(len(cost_list))]
        self._logger.evaluations = [i + 1 for i in range(len(cost_list))]
        self._logger.x_model_best = x_best[-1]
        x_search_best_over_time = [
            [
                par.transformation.to_search(e)[0]
                for e, par in zip(entry, self.problem.parameters.values())
            ]
            for entry in x_best
        ]
        self._logger.x_search_best = x_search_best_over_time[-1]
        self._logger.cost_best = cost_best
        self._logger.cost_best = cost_best[-1]
        posterior = pybop.MultivariateParameters(
            self.problem.parameters,
            distribution=pybop.MultivariateNonparametric(
                self.optimiser.denormalize_input(raw_taken_samples).T
            )
        )
    
        self._logger.iteration = {"SOBER iterations": self._options.sober_iterations}
        self._logger.evaluations = {"model evaluations": self._options.sober_iterations * self._options.model_samples_per_iteration}

        return BayesianOptimisationResult(
            optim=self,
            logger=self._logger,
            time=end - start,
            optim_name="EP-LFI + SOBER + BASQ",
            posterior=posterior,
            lower_bounds=None if self.bounds is None else self.bounds[:, 0],
            upper_bounds=None if self.bounds is None else self.bounds[:, 1],
            maximum_a_posteriori=MAP,
            log_evidence_mean=log_expected_marginal_likelihood,
            log_evidence_variance=log_approx_variance_marginal_likelihood,
        )
    
    def name(self):
        return (
            "Solving Optimisation as Bayesian Estimation via Recombination "
            "within Expectation Propagation for Likelihood-Free Inference"
        )
