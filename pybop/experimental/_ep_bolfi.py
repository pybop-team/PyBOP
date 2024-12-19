import json

import ep_bolfi
import numpy as np

from pybop import BaseOptimiser
from pybop.experimental.base_bayes_optimiser import BayesianOptimisationResult
from pybop.experimental.multivariate_priors import MultivariateGaussian


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
    MultivariateGaussian prior and an ExpectationPropagationCost.
    """

    def model_wrapper(self, parameter_set):
        return self.cost.problem.model.predict(
            self.inputs, self.t_eval, parameter_set, self.experiment, self.initial_state
        )[self.output_variable]

    def _set_up_optimiser(self):
        # Read in EP-BOLFI-specific settings.
        self.boundaries_in_deviations = self.unset_options.pop(
            "boundaries_in_deviations", 0
        )
        self.bolfi_initial_evidence = self.unset_options.pop(
            "bolfi_initial_evidence", None
        )
        self.bolfi_total_evidence = self.unset_options.pop("bolfi_total_evidence", None)
        self.bolfi_posterior_samples = self.unset_options.pop(
            "bolfi_posterior_samples", None
        )
        self.ep_iterations = self.unset_options.pop("ep_iterations", 3)
        self.ep_dampener = self.unset_options.pop("ep_dampener", None)
        self.final_dampening = self.unset_options.pop("final_dampening", None)
        self.ep_dampener_reduction_steps = self.unset_options.pop(
            "ep_dampener_reduction_steps", -1
        )
        self.ess_ratio_resample = self.unset_options.pop("ess_ratio_resample", 5)
        self.ess_ratio_sampling_from_zero = self.unset_options.pop(
            "ess_ratio_sampling_from_zero", -1
        )
        self.ess_ratio_abort = self.unset_options.pop("ess_ratio_abort", 20)
        # Copy the state of a previous EP-BOLFI call, if given.
        self.Q = self.unset_options.pop("Q", None)
        self.r = self.unset_options.pop("r", None)
        self.Q_features = self.unset_options.pop("Q_features", None)
        self.r_features = self.unset_options.pop("r_features", None)
        # Read in live feedback options.
        self.show_trials = self.unset_options.pop("show_trials", None)
        self.verbose = self.unset_options.pop("verbose", None)
        # Read in auxiliary EP-BOLFI settings.
        self.gelman_rubin_threshold = self.unset_options.pop(
            "gelman_rubin_threshold", None
        )
        self.max_heuristic_steps = self.unset_options.pop("max_heuristic_steps", 10)
        self.posterior_sampling_increase = self.unset_options.pop(
            "posterior_sampling_increase", 1.2
        )
        self.model_resampling_increase = self.unset_options.pop(
            "model_resampling_increase", 1.1
        )
        self.independent_mcmc_chains = self.unset_options.pop(
            "independent_mcmc_chains", 4
        )
        self.seed = self.unset_options.pop("seed", -1)
        transposed_boundaries = {}
        for i, name in enumerate(self.parameters.param.keys()):
            transposed_boundaries[name] = [
                self.bounds["lower"][i],
                self.bounds["upper"][i],
            ]
        # EP-BOLFI can handle multiple simulators at once, hence the
        # lists. ToDo: mediate this between EP-BOLFI and PyBOP.
        self.optimiser = ep_bolfi.EP_BOLFI(
            [self.model_wrapper],
            [self.cost.problem.dataset],
            self.cost.costs,
            fixed_parameters={},  # probably baked into self.problem.model
            free_parameters={k: v.initial_value for k, v in self.parameters.items()},
            initial_covariance=self.parameters.prior.properties["cov"],
            free_parameters_boundaries=transposed_boundaries,
            boundaries_in_deviations=self.boundaries_in_deviations,
            Q=self.Q,
            r=self.r,
            Q_features=self.Q_features,
            r_features=self.r_features,
            transform_parameters={},  # might be handled within PyBOP
            weights=None,  # only applicable within vector-valued features
            display_current_feature=None,  # ToDo: costs with names
            fixed_parameter_order=list(self.parameters.param.keys()),
        )

    def _run(self):
        # bolfi_posterior is the full GPy object containing the state at
        # the end of the last feature iteration, while the
        # MultivariateGaussian is a slight approximation.
        self.bolfi_posterior = self.optimiser.run(
            self.bolfi_initial_evidence,
            self.bolfi_total_evidence,
            self.bolfi_posterior_samples,
            self.ep_iterations,
            self.ep_dampener,
            self.final_dampening,
            self.ep_dampener_reduction_steps,
            self.gelman_rubin_threshold,
            self.ess_ratio_resample,
            self.ess_ratio_sampling_from_zero,
            self.ess_ratio_abort,
            self.max_heuristic_steps,
            self.posterior_sampling_increase,
            self.model_resampling_increase,
            self.independent_mcmc_chains,
            self.scramble_ep_feature_order,
            self.show_trials,
            self.verbose,
            self.seed,
        )
        ep_bolfi_result = json.loads(self.optimiser.result_to_json(seed=self.seed))
        len_log = len(
            self.optimiser.log_of_tried_parameters[
                list(self.parameters.param.values())[0]
            ]
        )
        transposed_log = [[] for _ in range(len_log)]
        for log in self.optimiser.log_of_tried_parameters.values():
            for j, l in enumerate(log):
                transposed_log[j].append(l)
        for key in self.log.keys():
            self.log[key] = []
        self.log_update(x=transposed_log)
        mean = np.array(ep_bolfi_result["inferred_parameters"].values())
        return BayesianOptimisationResult(
            x=mean,
            posterior=MultivariateGaussian(
                mean, np.array(ep_bolfi_result["covariance"])
            ),
            cost=self.cost,
            n_iterations={
                "model evaluations": len_log,
                "EP iterations": self.ep_iterations,
                "total feature iterations": self.ep_iterations * len(self.cost.costs),
            },
            optim=self.optimiser,
        )

    def name(self):
        return (
            "Expectation Propagation with Bayesian Optimization for "
            "Likelihood-Free Inference"
        )
