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

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides methods for estimating and optimising battery model parameters, offering both deterministic and stochastic approaches with example workflows to assist users. `PyBOP` enables parameter identification from data for various battery models, including the electrochemical and equivalent circuit models provided by the popular open-source `PyBaMM` package [@Sulzer:2021]. Using the same approaches, `PyBOP` can also be used for design optimisation under user-defined operating conditions across a variety of model structures and design goals. `PyBOP` facilitates optimisation with a range of methods, providing diagnostics for examining optimiser performance and convergence of both the cost function and corresponding parameters. Identified parameters can be used for prediction, online estimation and control, and design optimisation, accelerating battery research and development.

# Statement of need

`PyBOP` is a Python package that provides a user-friendly, object-oriented interface for optimising battery model parameters. `PyBOP` leverages the open-source `PyBaMM` package [@Sulzer:2021] to formulate and solve battery models. Together, these tools serve a broad audience including students, engineers, and researchers in academia and industry, enabling the use of advanced models where previously this was not possible without specialised knowledge of battery modelling, parameter inference, and software development. `PyBOP` emphasises clear and informative diagnostics and workflows to support users with varying levels of domain expertise, and provides access to a wide range of optimisation and sampling algorithms. These capabilities are enabled through interfaces to `PINTS` [@Clerx:2019], `SciPy` [@SciPy:2020], and `PyBOP`'s own implementations of algorithms such as Adaptive Moment Estimation with Weight Decay (AdamW) [@Loshchilov:2017], Gradient Descent [@Cauchy:1847], and Cuckoo Search [@Yang:2009].

`PyBOP` complements other lithium-ion battery modelling packages built around `PyBaMM`, such as `liionpack` for battery pack simulation [@Tranter2022] and `pybamm-eis` for fast numerical computation of the electrochemical impedance of any battery model.

# Architecture

`PyBOP` formulates the inference process into three core classes: `Problem`, `Optimiser`, and `Sampler`, as shown in \autoref{fig:classes}. Each of these objects represents a base class with child classes that construct specialised functionality for different workflows. Management of these different workflows is achieved through a builder pattern for the `Problem` class, aiming to provide a robust, flexible interface. The `Problem` object offers the required functionality to compute the forward model, generate residuals if needed, and compute the corresponding cost for parameter value candidates. Multiple `Problem` builders are provided to construct optimisation `Problem`s for both time-series and electrochemical impedance spectroscopy PyBaMM domains, alongside a pure Python problem for general optimisation. The builder structure allows for extensibility and flexibility when new optimisation problems are required without requiring refactoring of PyBOP's core classes. An example of this is the `MultiFittingProblem` and corresponding builder, which generalises the pure Python problem and builder for optimisation tasks where multiple objectives are minimised. One such common use case for the `MultiFittingProblem` is parameter identification workflows where the initial model state varies for each corresponding set of observations; however, many other use cases are available through this generalised interface. The syntax for building a `PyBaMM`-based parameter inference workflow is shown below.

```python
builder = (
    pybop.builders.Pybamm()
    .set_dataset(dataset)
    .set_simulation(model, parameter_values=parameter_values)
    .add_parameter(pybop.Parameter("Negative electrode thickness [m]"))
    .add_cost(pybop.costs.pybamm.SumSquaredError("Voltage [V]"))
)
problem = builder.build()
optim = pybop.CMAES(problem)
result = optim.run()
```

The `PyBaMMProblem` class interfaces with the `PyBaMM` forward solution via a composed `Pipeline` object that manages the `PyBaMM` model, including initial state calculation, discretising and meshing the model, and exception handling with mocks for improved convergence. This `Pipeline` provides a singular interface for `PyBOP` to manage the `PyBaMM` simulation, allowing the `Problem` and optimisation classes to be agnostic to the various contexts required to acquire the forward solution. In addition, the `Pipeline` also manages multiprocessing for the `PyBaMMProblem`, as this is completed within the numerical C++ solver. The `Optimiser` and `Sampler` classes orchestrate the parameter inference process through either optimisation algorithms or Monte Carlo sampling. For the `PyBaMM`-based `Problem` classes, `PyBOP` constructs the cost functions as a PyBaMM expression, which is applied to the user-provided model after a defensive copy is performed. This implementation allows the cost to be computed alongside the forward solution, with gradient information available through `PyBaMM`'s automatic differentiation capabilities. Custom cost definitions are supported through a generalised `UserCost` subclass. 


Furthermore, for PyBaMM-based builders, PyBOP supports user-provided PyBaMM models for optimisation and parameter inference workflows. This allows users to use both canonical models offered by PyBaMM and custom formulations with PyBOP's optimisation methods. Under these conditions, PyBOP aims to apply the minimal number of modifications to the provided model in an effort to improve optimisation convergence, as well as corresponding goodness-of-fit criteria. One such example is spatial re-discretisation, which is required for the standard PyBaMM mesh construction for optimisation of geometric parameters. In this situation, `PyBOP` rebuilds the `PyBaMM` model only when necessary, aiming to limit the effect on workflow performance. In addition to the convergence information, identifiability metrics are provided with the correspondingly estimated parameter values through Hessian approximation, as well as Sobol sampling from the `salib` package.

![The core `PyBOP` architecture with base class interfaces. Each class provides a direct mapping to a step in the optimisation workflow. \label{fig:classes}](figures/PyBOP_components.drawio.png){ width=80% }

Alongside construction of the simulation process, the `Pipeline` object provides methods for obtaining sensitivities from the prediction, enabling gradient-based optimisation. A forward prediction, along with its corresponding sensitivities, is provided to the problem class for processing and exception control. A standardised data structure is then provided to the cost classes, which compute a distance, design, or likelihood-based metric for optimisation. For point-based optimisation, the optimisers minimise the cost function or the negative log-likelihood if a likelihood class is provided. Bayesian inference is provided by sampler classes, which accept the `LogPosterior` class and sample from it using `PINTS`-based Monte Carlo algorithms. In the typical workflow, the classes in \autoref{fig:classes} are constructed in sequence, from left to right in the figure.

In addition to the core architecture, `PyBOP` provides several specialised inference and optimisation features. One example is parameter inference from electrochemical impedance spectroscopy (EIS) simulations, where PyBOP discretises and linearises the EIS forward model into a sparse mass matrix form with an accompanying auto-differentiated Jacobian. This is then translated into the frequency domain, providing a direct solution to compute the input-output impedance. In this situation, the forward models are constructed within the spatial re-discretisation workflow, allowing for geometric parameter inference from EIS simulations and data. The currently implemented cost classes are listed in \autoref{tab:subclasses}. 

:List of default cost (or likelihood) classes. \label{tab:subclasses}

| Error Measures / Likelihoods | Design Metrics |
|:----------------------------|:--------------------------|
| Sum-squared error           | Volumetric energy density |
| Root-mean-squared error     | Gravimetric energy density |
| Minkowski                   | |
| Sum-of-power                | |
| Gaussian log likelihood     | |
| Maximum a Posteriori        | |

Similarly, the current optimisation algorithms are presented in \autoref{tab:optimisers}. It should be noted that SciPy minimize includes several gradient-based and gradient-free methods. From here on, the point-based parameterisation and design-optimisation tasks will simply be referred to as optimisation tasks. This simplification can be justified by comparing \autoref{eqn:parameterisation} and \autoref{eqn:design}; deterministic parameterisation is simply an optimisation task to minimise a distance-based cost between model output and measured values.

: Currently supported optimisers classified by candidate solution type, including gradient information. \label{tab:optimisers}

| Gradient-based                                    | Evolutionary                          | (Meta)heuristic           |
|:--------------------------------------------------|:--------------------------------------|:--------------------------|
| Weight decayed adaptive moment estimation (AdamW) | Covariance matrix adaptation (CMA-ES) | Particle swarm (PSO)      |
| Improved resilient backpropagation (iRProp-)      | Exponential natural (xNES)            | Nelder-Mead               |
| Gradient descent                                  | Separable natural (sNES)              | Cuckoo search             |
| SciPy minimize                                    | SciPy differential evolution          | Simulated Annealing       |


In addition to deterministic optimisers (\autoref{tab:optimisers}), `PyBOP` also provides Monte Carlo sampling routines to estimate distributions of parameters within a Bayesian framework. These methods construct a posterior parameter distribution that can be used to assess uncertainty and practical identifiability. The individual sampler classes are currently composed within `PyBOP` from the `PINTS` library, with a base sampler class implemented for interoperability and direct integration with `PyBOP`'s model, problem, and likelihood classes. The currently supported samplers are listed in \autoref{tab:samplers}.

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

Common battery models include various types of equivalent circuit models (e.g., the Thévenin model), the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory, and its reduced-order variants including the single particle model (SPM) [@Planella:2022] and the multi-species multi-reaction (MSMR) model [@Verbrugge:2017]. Simplified models that retain acceptable predictive accuracy at lower computational cost are widely used, for example, in battery management systems, while physics-based models are required to understand the impact of physical parameters on performance. This separation of complexity traditionally results in multiple parameterisations for a single battery type, depending on the model structure.

# Examples

## Parameterisation

The parameterisation of battery models is challenging due to the large number of parameters that need to be identified compared to the number of measurable outputs [@Miguel:2021; @Wang:2022; @Andersson:2022]. A complete parameterisation often requires stepwise identification of smaller sets of parameters from a variety of excitations and different datasets [@Chu:2019; @Chen:2020; @Lu:2021; @Kirk:2022]. Furthermore, parameter identifiability can be poor for a given set of excitations and datasets, requiring improved experimental design in addition to uncertainty-capable identification methods [@Aitio:2020].

A generic data-fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\hat{\mathbf{y}_i})}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the agreement between the model output $\mathbf{y}(t)$ and a sequence of observations $(\hat{\mathbf{y}_i})$ measured at times $t_i$. Within the `PyBOP` framework, the `FittingProblem` class packages the model output along with the measured observations, both of which are then passed to the cost classes for computation of the specific cost function. For gradient-based optimisers, the Jacobian of the cost function with respect to unknown parameters, $\partial \mathcal{L} / \partial \theta$, is computed for step-size and directional information.

Next, we demonstrate the fitting of synthetic data where the model parameters are known. Throughout this section, as an example, we use `PyBaMM`'s implementation of the single particle model with an added contact resistance submodel. We assume that the model is already fully parameterised apart from two parameters, namely, the lithium diffusivity of the negative electrode active material particles (denoted "negative particle diffusivity") and the contact resistance, with corresponding true values of [3.3e-14 $\text{m}^2/\text{s}$, 10 mΩ]. To start, we generate synthetic time-domain data corresponding to a one-hour discharge from 100% to 0% state of charge, denoted as 1C rate, followed by 30 minutes of relaxation. This dataset is then corrupted with zero-mean Gaussian noise of amplitude 2 mV, with the resulting signal shown by the blue dots in \autoref{fig:inference-time-landscape} (left). The initial states are assumed known, although this assumption is not generally necessary. The `PyBOP` repository contains several other [example notebooks](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks) that follow a similar inference process. The underlying cost landscape to be explored by the optimiser is shown in \autoref{fig:inference-time-landscape} (right), with the initial position denoted alongside the known true system parameters for this synthetic inference task. In general, the true parameters are not known.

![The synthetic fitting dataset (left) and cost landscape (right) for an example time-series battery model parameterisation using a root-mean-squared error cost function. \label{fig:inference-time-landscape}](figures/joss/sim-landscape.pdf){ width=100% }

We can also use `PyBOP` to generate and fit electrochemical impedance data using methods within `pybamm-eis` that enable fast impedance computation of battery models [@pybamm-eis]. Using the same model and parameters as in the time-domain case, \autoref{fig:impedance-landscape} shows the numerical impedance prediction available in `PyBOP` alongside the cost landscape for the corresponding inference task. At the time of publication, gradient-based optimisation and sampling methods are not available when using an impedance workflow.

![The data and model fit (left) and cost landscape (right) for a frequency-domain impedance parameterisation  with a root-mean-squared error cost function, at 5% SOC. \label{fig:impedance-landscape}](figures/joss/impedance.pdf){ width=100% }

To avoid confusion, in the remainder of this section, we continue with identification in the time domain (\autoref{fig:inference-time-landscape}). In general, however, time- and frequency-domain models and data may be combined for improved parameterisation. As gradient information is available for our time-domain example, the choice of distance-based cost function and optimiser is not constrained. Due to the difference in magnitude between the two parameters, we apply the logarithmic parameter transformation offered by `PyBOP`. This transforms the search space of the optimiser to allow for a common step size between the parameters, improving convergence in this particular case. As a demonstration of the parameterisation capabilities of `PyBOP`, \autoref{fig:convergence-min-max} (left) shows the rate of convergence for each of the distance-minimising cost functions, while \autoref{fig:convergence-min-max} (right) shows analogous results for maximising a likelihood. The optimisation is performed with SciPy minimize using the gradient-based L-BFGS-B method.

![Optimiser convergence using various cost (left) and likelihood (right) functions and the L-BFGS-B algorithm. \label{fig:convergence-min-max}](figures/joss/converge.pdf){ width=100% }

Using the same model and parameters, we compare example convergence rates of various algorithms across several categories: gradient-based methods in \autoref{fig:optimiser-inference1} (left), evolutionary strategies in \autoref{fig:optimiser-inference1} (middle) and (meta)heuristics in \autoref{fig:optimiser-inference1} (right) using a mean-squared-error cost function. We also show the cost function and optimiser iterations in \autoref{fig:optimiser-inference2}, with the three rows showing the gradient-based optimisers (top), evolution strategies (middle), and (meta)heuristics (bottom). Note that the performance of the optimiser depends on the cost landscape, the initial guess or prior, and the hyperparameters for each specific problem.

![Convergence in parameter values for several optimisation algorithms provided by `PyBOP`.  \label{fig:optimiser-inference1}](figures/joss/optimisers_parameters.pdf){ width=100% }

![Cost landscape contour plot with corresponding optimisation traces, for several optimisers. \label{fig:optimiser-inference2}](figures/joss/contour_subplot.pdf){ width=100% }

This example parameterisation task can also be approached from a Bayesian perspective, using `PyBOP`'s sampler methods. First, we introduce Bayes' rule,

\begin{equation}
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)},
\label{eqn:bayes_theorem}
\end{equation}

where $P(\theta|D)$ is the posterior parameter distribution, $P(D|\theta)$ is the likelihood function, $P(\theta)$ is the prior parameter distribution, and $P(D)$ is the model evidence, or marginal likelihood, which acts as a normalising constant. In the case of maximum likelihood estimation or maximum a posteriori estimation, one wishes to maximise $P(D|\theta)$ or $P(\theta|D)$, respectively, and this may be formulated as an optimisation problem as per \autoref{eqn:parameterisation}.

To estimate the full posterior parameter distribution, however, one must use sampling or other inference methods to reconstruct the function $P(\theta|D)$. The posterior distribution provides information about the uncertainty of the identified parameters, e.g., by calculating the variance or other moments. Monte Carlo methods are used here to sample from the posterior. The selection of Monte Carlo methods available in `PyBOP` includes gradient-based methods such as No-U-Turn [@NUTS:2011] and Hamiltonian [@Hamiltonian:2011], as well as heuristic methods such as differential evolution [@DiffEvolution:2006], and also conventional methods based on random sampling with rejection criteria [@metropolis:1953]. `PyBOP` offers a sampler class that provides the interface to samplers, the latter being provided by the Probabilistic Inference on Noisy Time-series (`PINTS`) package. \autoref{fig:posteriors} shows the sampled posteriors for the synthetic model described previously, using an adaptive covariance-based sampler called Haario Bardenet [@Haario:2001].

![Posterior distributions of model parameters alongside identified noise on the observations. Shaded areas denote the 95th percentile credible interval for each parameter. \label{fig:posteriors}](figures/joss/posteriors.pdf){ width=100% }

## Design optimisation

Design optimisation is supported in `PyBOP` to guide device design development by identifying parameter sensitivities that can unlock improvements in performance. This problem can be viewed in a similar way to the parameterisation workflows described previously, but with the aim of maximising a design-objective cost function rather than minimising a distance-based cost function. `PyBOP` performs maximisation by minimising the negative of the cost function. In design problems, the cost metric is no longer a distance between two time series, but a metric evaluated on a model prediction. For example, to maximise the gravimetric energy (or power) density, the cost is the integral of the discharge energy (or power) normalised by the cell mass. Such metrics are typically quantified for operating conditions such as a 1C discharge at a given temperature.

In general, design optimisation can be written as a constrained optimisation problem,
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ \mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions}),}
\label{eqn:design}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability of the design and $\Omega$ is the set of allowable parameter values.

As an example, we consider the challenge of maximising the gravimetric energy density, subject to constraints on two of the geometric electrode parameters [@Couto:2023]. In this case, we use the `PyBaMM` implementation of the single particle model with electrolyte (SPMe) to investigate the impact of the positive electrode thickness and the active material volume fraction on the energy density. Since the total volume fraction must sum to unity, the positive electrode porosity for each optimisation iteration is defined in relation to the active material volume fraction. It is also possible to update the 1C rate corresponding to the theoretical capacity for each iteration of the design.

![Initial and optimised voltage profiles alongside the gravimetric energy density cost landscape.  \label{fig:design_gravimetric}](figures/joss/design.pdf){ width=100% }

\autoref{fig:design_gravimetric} (left) shows the predicted improvement in the discharge profile between the initial and optimised parameter values for a fixed-rate 1C discharge selected from the initial design and (right) the Nelder-Mead search over the parameter space.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to `PyBOP`. This work was supported by the Faraday Institution Multiscale Modelling project (ref. FIRG059), UKRI's Horizon Europe Guarantee (ref. 10038031), and EU IntelLiGent project (ref. 101069765).

# References
