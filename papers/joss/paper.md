---
title: 'PyBOP: A Python package for battery model optimisation and parameterisation'
tags:
  - python
  - battery
  - model
  - parameter
  - inference
  - design optimisation
authors:
  - name: Brady Planden
    orcid: 0000-0002-1082-9125
    affiliation: 1
  - name: Nicola E. Courtier
    affiliation: "1, 2"
    orcid: 0000-0002-5714-1096
  - name: Martin Robinson
    orcid: 0000-0002-1572-6782
    affiliation: 3
  - name: Agriya Khetarpal
    orcid: 0000-0002-1112-1786
    affiliation: 4
  - name: Ferran Brosa Planella
    affiliation: "2, 5"
    orcid: 0000-0001-6363-2812
  - name: David A. Howey
    corresponding: true
    affiliation: "1, 2"
    orcid: 0000-0002-0620-3955
affiliations:
 - name: Department of Engineering Science, University of Oxford, Oxford, UK
   index: 1
 - name: The Faraday Institution, Harwell Campus, Didcot, UK
   index: 2
 - name: Research Software Engineering Group, University of Oxford, Oxford, UK
   index: 3
 - name: Quansight PBC
   index: 4
 - name:  Mathematics Institute, University of Warwick, Coventry, UK
   index: 5
date: 19 December 2024
bibliography: paper.bib
repository: https://github.com/pybop-team/PyBOP
---

# Summary

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides methods for estimating and optimising battery model parameters using both deterministic and stochastic approaches with example workflows. `PyBOP` enables parameter identification from data for various battery models, including electrochemical and equivalent circuit models from the open-source `PyBaMM` package [@Sulzer:2021]. The same approaches enable design optimisation under user-defined operating conditions across various model structures and design goals. `PyBOP` facilitates optimisation with multiple methods, providing diagnostics for examining optimiser performance and convergence of cost functions and parameters. Identified parameters can be used for prediction, online estimation, control, and design optimisation, accelerating battery research and development.

# Statement of need

`PyBOP` provides a user-friendly, object-oriented interface for optimising battery model parameters. It leverages the open-source `PyBaMM` package [@Sulzer:2021] to formulate and solve battery models. Together, these tools serve students, engineers, and researchers in academia and industry, enabling advanced model use without specialised knowledge of battery modelling, parameter inference, and software development. `PyBOP` emphasises clear diagnostics and workflows to support users with varying domain expertise, providing access to numerous optimisation and sampling algorithms. These capabilities are enabled through interfaces to `PINTS` [@Clerx:2019], `SciPy` [@SciPy:2020], and `PyBOP`'s implementations of algorithms including Adaptive Moment Estimation with Weight Decay (AdamW) [@Loshchilov:2017], Gradient Descent [@Cauchy:1847], and Cuckoo Search [@Yang:2009].

`PyBOP` complements other lithium-ion battery modelling packages built around `PyBaMM`, including `liionpack` for battery pack simulation [@Tranter2022] and `pybamm-eis` for fast electrochemical impedance computation.

# Architecture

`PyBOP` formulates the inference process into four core architectural components: `Builder`, `Pipeline`, `Problem`, and `Optimiser`/`Sampler`, as shown in \autoref{fig:classes}. `Builder` classes construct optimisation problems using a fluent interface, `Pipeline` classes manage simulation execution, `Problem` classes coordinate cost evaluation, and `Optimiser`/`Sampler` classes perform parameter inference. Each component represents a base class with child classes providing specialised functionality for different workflows.

The enhanced builder pattern provides a robust interface for constructing optimisation problems. The `BaseBuilder` class defines a common interface with methods including `set_dataset()`, `add_parameter()`, and `add_cost()`, enabling method chaining. Specialised builders (`Pybamm`, `PybammEIS`, `Python`, `MultiFitting`) extend this base functionality for specific use cases. This structure ensures extensibility for new optimisation problems without refactoring PyBOP's core classes. Multiple costs can be added with automatic weighting, and the builder validates requirements before constructing the final problem instance. The syntax for building a `PyBaMM`-based parameter inference workflow is shown below.

```python
# Builder pattern with extendable interface
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_parameter(pybop.Parameter("Negative electrode thickness [m]"))
    .add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
)

# Build and run inference
problem = builder.build()
optim = pybop.CMAES(problem)
result = optim.run()
```

A key architectural enhancement is explicit `Pipeline` classes that encapsulate simulation logic separate from problem coordination. The `PybammPipeline` manages PyBaMM model execution, including model building, discretisation, parameter updates, and sensitivity calculations. This separation allows `Problem` classes to focus on cost evaluation and optimisation coordination while `Pipeline` classes handle simulation execution complexities. The pipeline architecture provides a consistent interface between PyBOP and underlying simulation engines, facilitating future extensions to other modelling frameworks.

The `Problem` classes follow a clean hierarchy with `Problem` as the base class providing `run()` and `run_with_sensitivities()` methods. `PybammProblem` coordinates between `PybammPipeline` instances and cost evaluation, supporting multiple weighted costs and automatic hyperparameter handling for Bayesian inference. `Problem` classes are agnostic to simulation details, handled by their associated pipeline instances. This architecture enables consistent interfaces across different simulation backends while maintaining flexibility for specialised optimisation workflows. The `Optimiser` and `Sampler` classes orchestrate parameter inference through optimisation algorithms or Monte Carlo sampling, interfacing with problem classes through standardised methods. 


For PyBaMM-based builders, PyBOP supports user-provided PyBaMM models for optimisation and parameter inference workflows. This allows users to employ both canonical PyBaMM models and custom formulations with PyBOP's optimisation methods. PyBOP applies minimal modifications to provided models to improve optimisation convergence and goodness-of-fit criteria. For example, spatial re-discretisation is required for standard PyBaMM mesh construction when optimising geometric parameters. `PyBOP` rebuilds the `PyBaMM` model only when necessary to limit performance impact. Beyond convergence information, identifiability metrics are provided with estimated parameter values through Hessian approximation and Sobol sampling from the `salib` package.

![The core `PyBOP` architecture with four main components: Builder, Pipeline, Problem, and Optimiser/Sampler. Each component provides a direct mapping to a step in the optimisation workflow, with clear separation of concerns between construction, simulation, coordination, and inference. \label{fig:classes}](figures/PyBOP_components.drawio.png){ width=100% }

The `Pipeline` object provides methods for obtaining sensitivities from predictions, enabling gradient-based optimisation. Forward predictions with corresponding sensitivities are provided to the problem class for processing and exception control. A standardised data structure is then provided to cost classes, which compute distance, design, or likelihood-based metrics for optimisation. The restructured cost system supports multiple costs with automatic weighting and metadata introspection. Cost classes can define hyperparameters automatically added to the optimisation problem, enabling seamless integration of likelihood-based methods with hyperparameter inference. Cost evaluation is cleanly separated from simulation execution, with costs computed from pipeline outputs rather than embedded in the simulation process.

For point-based optimisation, optimisers minimise the cost function or negative log-likelihood if a likelihood class is provided. Bayesian inference is provided by sampler classes, which accept the `LogPosterior` class and sample using `PINTS`-based Monte Carlo algorithms. In typical workflows, the classes in \autoref{fig:classes} are constructed sequentially from left to right.

Beyond the core architecture, `PyBOP` provides specialised inference and optimisation features. Parameter inference from electrochemical impedance spectroscopy (EIS) simulations is handled through the `PybammEISPipeline`, which discretises and linearises the EIS forward model into sparse mass matrix form with an auto-differentiated Jacobian. The `PybammEIS` builder constructs problems for impedance-based parameter identification, with the pipeline managing frequency-domain transformations and impedance calculations. This architecture enables geometric parameter inference from EIS simulations while maintaining the same consistent interface as time-domain problems. Currently implemented cost classes are listed in \autoref{tab:subclasses}. 

:List of default cost classes. \label{tab:subclasses}

| Error Measures / Likelihoods | Design Metrics |
|:----------------------------|:--------------------------|
| Sum-squared error           | Volumetric energy density |
| Root-mean-squared error     | Gravimetric energy density |
| Minkowski                   | |
| Sum-of-power                | |
| Gaussian log likelihood     | |
| Maximum a Posteriori        | |

Current optimisation algorithms are presented in \autoref{tab:optimisers}. Note that SciPy minimize includes several gradient-based and gradient-free methods. Hereafter, point-based parameterisation and design-optimisation tasks are referred to as optimisation tasks. This simplification is justified by comparing \autoref{eqn:parameterisation} and \autoref{eqn:design}; deterministic parameterisation is an optimisation task to minimise distance-based cost between model output and measured values.

: Currently supported optimisers classified by candidate solution type, including gradient information. \label{tab:optimisers}

| Gradient-based                                    | Evolutionary                          | (Meta)heuristic      |
|:--------------------------------------------------|:--------------------------------------|:---------------------|
| Weight decayed adaptive moment estimation (AdamW) | Covariance matrix adaptation (CMA-ES) | Particle swarm (PSO) |
| Gradient descent                                  | Exponential natural (xNES)            | Nelder-Mead          |
| SciPy minimize                                    | Separable natural (sNES)              | Cuckoo search        |
| Improved resilient backpropagation (iRProp-/+)    | SciPy differential evolution          | Simulated Annealing  |


Beyond deterministic optimisers (\autoref{tab:optimisers}), `PyBOP` provides Monte Carlo sampling routines to estimate parameter distributions within a Bayesian framework. These methods construct posterior parameter distributions for assessing uncertainty and practical identifiability. Individual sampler classes are composed within `PyBOP` from the `PINTS` library, with a base sampler class implemented for interoperability and direct integration with `PyBOP`'s model, problem, and likelihood classes. Currently supported samplers are listed in \autoref{tab:samplers}.

: Sampling methods supported by `PyBOP`, classified according to the candidate proposal method. \label{tab:samplers}

| Gradient-based    | Adaptive                   | Slicing        | Evolutionary           | Other                        |
|:------------------|:---------------------------|:---------------|:-----------------------|:-----------------------------|
| Monomial gamma    | Delayed rejection adaptive | Rank shrinking | Differential evolution | Metropolis random walk       |
| No-U-turn         | Haario Bardenet            | Doubling       |                        | Emcee hammer                 |
| Hamiltonian       | Haario                     | Stepout        |                        | Metropolis adjusted Langevin |
| Relativistic      | Rao Blackwell              |                |                        |                              |


# Background

## Battery models

In general, battery models (after spatial discretisation) can be written in the form of a differential-algebraic system of equations,
\begin{equation}
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = f(t,\mathbf{x},\mathbf{\theta}),
\label{dynamics}
\end{equation}
\begin{equation}
0 = g(t, \mathbf{x}, \mathbf{\theta}),
\label{algebraic}
\end{equation}
\begin{equation}
\mathbf{y}(t) = h(t, \mathbf{x}, \mathbf{\theta}),
\label{output}
\end{equation}
with initial conditions
\begin{equation}
\mathbf{x}(0) = \mathbf{x}_0(\mathbf{\theta}).
\label{initial_conditions}
\end{equation}

Here, $t$ is time, $\mathbf{x}(t)$ are the (spatially discretised) states, $\mathbf{y}(t)$ are the outputs (e.g., the terminal voltage) and $\mathbf{\theta}$ are the unknown parameters.

Common battery models include equivalent circuit models (e.g., the Thévenin model), the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory, and reduced-order variants including the single particle model (SPM) [@Planella:2022] and multi-species multi-reaction (MSMR) model [@Verbrugge:2017]. Simplified models retaining acceptable predictive accuracy at lower computational cost are widely used in battery management systems, while physics-based models are required to understand physical parameter impacts on performance. This complexity separation traditionally results in multiple parameterisations for a single battery type, depending on model structure.

# Examples

## Parameterisation

Battery model parameterisation is challenging due to the large number of parameters requiring identification compared to measurable outputs [@Miguel:2021; @Wang:2022; @Andersson:2022]. Complete parameterisation often requires stepwise identification of smaller parameter sets from various excitations and datasets [@Chu:2019; @Chen:2020; @Lu:2021; @Kirk:2022]. Parameter identifiability can be poor for given excitations and datasets, requiring improved experimental design and uncertainty-capable identification methods [@Aitio:2020].

A generic data-fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\hat{\mathbf{y}_i})}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function quantifying agreement between model output $\mathbf{y}(t)$ and observations $(\hat{\mathbf{y}_i})$ measured at times $t_i$. Within the `PyBOP` framework, the `FittingProblem` class packages model output with measured observations, passing both to cost classes for cost function computation. For gradient-based optimisers, the Jacobian $\partial \mathcal{L} / \partial \theta$ is computed for step-size and directional information.

We demonstrate fitting synthetic data with known model parameters. We use `PyBaMM`'s single particle model with added contact resistance submodel. The model is fully parameterised except for two parameters: lithium diffusivity of negative electrode active material particles ("negative particle diffusivity") and contact resistance, with true values [3.3e-14 $\text{m}^2/\text{s}$, 10 mΩ]. We generate synthetic time-domain data for a one-hour discharge from 100% to 0% state of charge (1C rate) followed by 30 minutes relaxation. This dataset is corrupted with zero-mean Gaussian noise of 2 mV amplitude, shown as blue dots in \autoref{fig:inference-time-landscape} (left). Initial states are assumed known, though this is not generally necessary. The `PyBOP` repository contains [example notebooks](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks) following similar inference processes. The underlying cost landscape explored by the optimiser is shown in \autoref{fig:inference-time-landscape} (right), with initial position and known true system parameters for this synthetic inference task. Generally, true parameters are unknown.

![The synthetic fitting dataset (left) and cost landscape (right) for an example time-series battery model parameterisation using a root-mean-squared error cost function. \label{fig:inference-time-landscape}](figures/joss/sim-landscape.pdf){ width=100% }

`PyBOP` can generate and fit electrochemical impedance data using `pybamm-eis` methods enabling fast impedance computation of battery models [@pybamm-eis]. Using the same model and parameters as the time-domain case, \autoref{fig:impedance-landscape} shows numerical impedance prediction available in `PyBOP` alongside the cost landscape for the corresponding inference task. At publication, gradient-based optimisation and sampling methods are unavailable for impedance workflows.

![The data and model fit (left) and cost landscape (right) for a frequency-domain impedance parameterisation  with a root-mean-squared error cost function, at 5% SOC. \label{fig:impedance-landscape}](figures/joss/impedance.pdf){ width=100% }

We continue with time-domain identification (\autoref{fig:inference-time-landscape}). Generally, time- and frequency-domain models and data may be combined for improved parameterisation. As gradient information is available for our time-domain example, distance-based cost function and optimiser choice is unconstrained. Due to magnitude differences between parameters, we apply logarithmic parameter transformation offered by `PyBOP`. This transforms the optimiser search space to allow common step sizes between parameters, improving convergence. Demonstrating `PyBOP`'s parameterisation capabilities, \autoref{fig:convergence-min-max} (left) shows convergence rates for distance-minimising cost functions, while \autoref{fig:convergence-min-max} (right) shows analogous results for likelihood maximisation. Optimisation uses SciPy minimize with the gradient-based L-BFGS-B method.

![Optimiser convergence using various cost (left) and likelihood (right) functions and the L-BFGS-B algorithm. \label{fig:convergence-min-max}](figures/joss/converge.pdf){ width=100% }

Using the same model and parameters, we compare convergence rates of various algorithms across categories: gradient-based methods in \autoref{fig:optimiser-inference1} (left), evolutionary strategies in \autoref{fig:optimiser-inference1} (middle), and (meta)heuristics in \autoref{fig:optimiser-inference1} (right) using mean-squared-error cost function. \autoref{fig:optimiser-inference2} shows cost function and optimiser iterations, with three rows showing gradient-based optimisers (top), evolution strategies (middle), and (meta)heuristics (bottom). Optimiser performance depends on cost landscape, initial guess or prior, and hyperparameters for each problem.

![Convergence in parameter values for several optimisation algorithms provided by `PyBOP`.  \label{fig:optimiser-inference1}](figures/joss/optimisers_parameters.pdf){ width=100% }

![Cost landscape contour plot with corresponding optimisation traces, for several optimisers. \label{fig:optimiser-inference2}](figures/joss/contour_subplot.pdf){ width=100% }

This parameterisation task can be approached from a Bayesian perspective using `PyBOP`'s sampler methods. First, we introduce Bayes' rule,

\begin{equation}
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)},
\label{eqn:bayes_theorem}
\end{equation}

where $P(\theta|D)$ is the posterior parameter distribution, $P(D|\theta)$ is the likelihood function, $P(\theta)$ is the prior parameter distribution, and $P(D)$ is the model evidence or marginal likelihood acting as a normalising constant. For maximum likelihood estimation or maximum a posteriori estimation, one maximises $P(D|\theta)$ or $P(\theta|D)$, respectively, formulated as an optimisation problem per \autoref{eqn:parameterisation}.

To estimate the full posterior parameter distribution, one must use sampling or other inference methods to reconstruct $P(\theta|D)$. The posterior distribution provides uncertainty information about identified parameters, e.g., by calculating variance or other moments. Monte Carlo methods sample from the posterior. Monte Carlo methods available in `PyBOP` include gradient-based methods like No-U-Turn [@NUTS:2011] and Hamiltonian [@Hamiltonian:2011], heuristic methods like differential evolution [@DiffEvolution:2006], and conventional methods based on random sampling with rejection criteria [@metropolis:1953]. `PyBOP` offers a sampler class providing the interface to samplers from the Probabilistic Inference on Noisy Time-series (`PINTS`) package. \autoref{fig:posteriors} shows sampled posteriors for the synthetic model using an adaptive covariance-based sampler called Haario Bardenet [@Haario:2001].

![Posterior distributions of model parameters alongside identified noise on the observations. Shaded areas denote the 95th percentile credible interval for each parameter. \label{fig:posteriors}](figures/joss/posteriors.pdf){ width=100% }

## Design optimisation

`PyBOP` supports design optimisation to guide device design development by identifying parameter sensitivities that unlock performance improvements. This problem is similar to parameterisation workflows but aims to maximise a design-objective cost function rather than minimise a distance-based cost function. `PyBOP` performs maximisation by minimising the negative cost function. In design problems, the cost metric is no longer distance between time series, but a metric evaluated on model predictions. For example, to maximise gravimetric energy (or power) density, the cost is the integral of discharge energy (or power) normalised by cell mass. Such metrics are typically quantified for operating conditions like 1C discharge at given temperature.

In general, design optimisation can be written as a constrained optimisation problem,
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ \mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions}),}
\label{eqn:design}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function quantifying design desirability and $\Omega$ is the set of allowable parameter values.

We consider maximising gravimetric energy density subject to constraints on two geometric electrode parameters [@Couto:2023]. We use the `PyBaMM` single particle model with electrolyte (SPMe) to investigate positive electrode thickness and active material volume fraction impacts on energy density. Since total volume fraction must sum to unity, positive electrode porosity for each optimisation iteration is defined relative to active material volume fraction. The 1C rate corresponding to theoretical capacity can be updated for each design iteration.

![Initial and optimised voltage profiles alongside the gravimetric energy density cost landscape.  \label{fig:design_gravimetric}](figures/joss/design.pdf){ width=100% }

\autoref{fig:design_gravimetric} (left) shows predicted improvement in discharge profile between initial and optimised parameter values for fixed-rate 1C discharge selected from the initial design and (right) Nelder-Mead search over parameter space.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to `PyBOP`. This work was supported by the Faraday Institution Multiscale Modelling project (FIRG059), UKRI's Horizon Europe Guarantee (10038031), and EU IntelLiGent project (101069765).

# References
