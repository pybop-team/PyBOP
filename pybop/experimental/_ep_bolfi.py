import copy
import json
import time

import ep_bolfi
import numpy as np
from pybamm import citations

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        citations.register("""@article{
            Minka2013,
            title={{Expectation Propagation for approximate Bayesian inference}},
            author={Minka, T},
            journal={Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence (UAI2001)},
            pages={362-369},
            year={2013}
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

    def model_wrapper(self, inputs):
        evaluation = self.cost.problem.model.predict(
            inputs, self.t_eval, None, self.experiment, self.initial_state
        )
        return {
            signal: evaluation[signal].entries for signal in self.cost.problem.signal
        }

    def _set_up_optimiser(self):
        # Read in predictor settings.
        self.inputs = self.unset_options.pop("inputs", None)
        self.t_eval = self.unset_options.pop("t_eval", None)
        self.experiment = self.unset_options.pop("experiment", None)
        self.initial_state = self.unset_options.pop("initial_state", None)
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
        self.scramble_ep_feature_order = self.unset_options.pop(
            "scramble_ep_feature_order", False
        )
        self.seed = self.unset_options.pop("seed", 0)
        transposed_boundaries = {}
        model_bounds = self.parameters.get_bounds(apply_transform=False)
        for i, name in enumerate(self.parameters.param.keys()):
            transposed_boundaries[name] = [
                model_bounds["lower"][i],
                model_bounds["upper"][i],
            ]
        # EP-BOLFI can handle multiple simulators at once, hence the
        # lists. ToDo: mediate this between EP-BOLFI and PyBOP.
        self.optimiser = ep_bolfi.EP_BOLFI(
            [self.model_wrapper],
            [self.cost.problem.dataset],
            [
                lambda y, stored_cost=cost: [stored_cost.compute(y)]
                for cost in self.cost.costs
            ],
            fixed_parameters={},  # probably baked into self.problem.model
            free_parameters={
                name: par.get_initial_value(apply_transform=True)
                for name, par in self.parameters.param.items()
            },
            initial_covariance=self.parameters.prior.properties["cov"],
            free_parameters_boundaries=transposed_boundaries,
            boundaries_in_deviations=self.boundaries_in_deviations,
            Q=self.Q,
            r=self.r,
            Q_features=self.Q_features,
            r_features=self.r_features,
            transform_parameters={
                name: (par.transformation.to_model, par.transformation.to_search)
                for name, par in self.parameters.param.items()
            },
            weights=None,  # only applicable within vector-valued features
            display_current_feature=None,  # ToDo: costs with names
            fixed_parameter_order=list(enumerate(self.parameters.param.keys())),
        )

    def _run(self):
        start = time.time()
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
            False,  # normalize_features
            self.show_trials,
            self.verbose,
            self.seed,
        )
        end = time.time()
        ep_bolfi_result = json.loads(self.optimiser.result_to_json(seed=self.seed))
        ep_bolfi_log = json.loads(self.optimiser.log_to_json())
        x_list = np.array(list(ep_bolfi_log["tried parameters"].values())).T
        # Collect all features into one cost. Note: they are logarithms,
        # so this is a multiplicative combination.
        feature_costs = np.array(list(ep_bolfi_log["discrepancies"].values()))
        cost_list = copy.deepcopy(feature_costs[0])
        for i in range(1, len(feature_costs)):
            for j in len(cost_list):
                cost_list[j][0] += feature_costs[i][j][0]
        cost_list = np.array([np.exp(value[0]) for value in cost_list])
        x_best = copy.deepcopy(x_list)
        cost_best = copy.deepcopy(cost_list)
        for i in range(1, len(cost_list)):
            if cost_list[i] < cost_best[i - 1]:
                x_best[i:, None] = x_list[i, None]
                cost_best[i:] = cost_list[i]
        # EP-BOLFI may be run several times with different settings,
        # while retaining its internal log. So we clear the PyBOP log.
        for key in self.log.keys():
            self.log[key] = []
        # x0 = [
        #     par.get_initial_value(apply_transform=True)
        #     for par in self.parameters.param.values()
        # ]
        # Temporarily remove _transformation, else log_update would apply it again.
        stored_transformation = self._transformation
        self._transformation = None
        self.log_update(
            x=x_list,
            x_best=x_best,
            cost=cost_list,
            cost_best=cost_best,  # , x0=x0
        )
        self._transformation = stored_transformation
        model_mean = np.array(
            [result[0] for result in ep_bolfi_result["inferred parameters"].values()]
        )
        search_mean = [
            par.transformation.to_search(entry)
            for entry, par in zip(model_mean, self.parameters.param.values())
        ]
        lower_bounds = np.array(
            [bounds[0][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )
        upper_bounds = np.array(
            [bounds[1][0] for bounds in ep_bolfi_result["error bounds"].values()]
        )
        # The re-use of `parameters` makes transformations easily usable.
        posterior = copy.deepcopy(self.parameters)
        posterior.prior = MultivariateGaussian(
            search_mean, np.array(ep_bolfi_result["covariance"])
        )
        final_cost = self.cost(model_mean)
        return BayesianOptimisationResult(
            optim=self,
            x=model_mean,
            final_cost=final_cost,
            n_iterations={
                "EP iterations": self.ep_iterations,
                "total feature iterations": self.ep_iterations * len(self.cost.costs),
            },
            n_evaluations={
                "model evaluations": len(
                    list(ep_bolfi_log["tried parameters"].values())[0]
                ),
                # "surrogate evaluations" are not directly accessible
            },
            time=end - start,
            posterior=posterior,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def run(self):
        """
        Run the optimisation and return the optimised parameters and final cost.

        Returns
        -------
        results: BayesOptimisationResult
            The pybop optimisation result class.
        """
        self.result = self._run()

        # Store the optimised parameters
        self.parameters.update(values=self.result.x_best)

        if self.verbose:
            print(self.result)

        return self.result

    def name(self):
        return (
            "Expectation Propagation with Bayesian Optimization for "
            "Likelihood-Free Inference"
        )
