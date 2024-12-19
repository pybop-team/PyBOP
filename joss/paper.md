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
 - name: Quansight Labs
   index: 4
 - name:  Mathematics Institute, University of Warwick, Coventry, UK
   index: 5
date: 04 December 2024
bibliography: paper.bib
repository: https://github.com/pybop-team/PyBOP
---

# Summary

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides methods for estimating and optimising battery model parameters, offering both deterministic and stochastic approaches with example workflows to assist users. `PyBOP` enables parameter identification from data for various battery models, including the electrochemical and equivalent circuit models provided by the popular open-source `PyBaMM` package [@Sulzer:2021]. Using the same approaches, `PyBOP` can also be used for design optimisation under user-defined operating conditions across a variety of model structures and design goals. `PyBOP` facilitates optimisation with a range of methods, with diagnostics for examining optimiser performance and convergence of the cost and corresponding parameters. Identified parameters can be used for prediction, on-line estimation and control, and design optimisation, accelerating battery research and development.

# Statement of need

`PyBOP` is a Python package providing a user-friendly, object-oriented interface for optimising battery model parameters. `PyBOP` leverages the open-source `PyBaMM` package [@Sulzer:2021] to formulate and solve battery models. Together, these tools serve a broad audience including students, engineers, and researchers in academia and industry, enabling the use of advanced models where previously this was not possible without specialised knowledge of battery modelling, parameter inference, and software development. `PyBOP` emphasises clear and informative diagnostics and workflows to support users with varying levels of domain expertise, and provides access to a wide range of optimisation and sampling algorithms. These are enabled through interfaces to `PINTS` [@Clerx:2019], `SciPy` [@SciPy:2020], and `PyBOP`'s own implementations of algorithms such as adaptive moment estimation with weight decay (AdamW) [@Loshchilov:2017], gradient descent [@Cauchy:1847], and cuckoo search [@Yang:2009].

`PyBOP` supports the battery parameter exchange (BPX) standard [@BPX:2023] for sharing parameter sets. These are typically costly to obtain due to the specialised equipment and time required for characterisation experiments, the need for domain knowledge, and the computational cost of estimation. `PyBOP` reduces the requirements for the latter two by providing fast parameter estimation methods, standardised workflows, and parameter set interoperability (via BPX).

`PyBOP` complements other lithium-ion battery modelling packages built around `PyBaMM`, such as `liionpack` for battery pack simulation [@Tranter2022] and `pybamm-eis` for fast numerical computation of the electrochemical impedance of any battery model. Identified `PyBOP` parameters are easily exportable to other packages.

# Architecture

`PyBOP` has a layered structure enabling the necessary functionality to compute forward predictions, process  results, and run optimisation and sampling algorithms. The forward model is solved using the battery modelling software `PyBaMM`, with construction, parameterisation, and discretisation managed by `PyBOP`'s model interface to `PyBaMM`. This provides a robust object construction process with a consistent interface between forward models and optimisers. Furthermore, identifiability metrics are provided along with the estimated parameters (through Hessian approximation of the cost functions around the optimum point in frequentist workflows, and posterior distributions in Bayesian workflows).

![The core `PyBOP` architecture with base class interfaces. Each class provides a direct mapping to a step in the optimisation workflow. \label{fig:classes}](figures/PyBOP_components.drawio.png){ width=80% }

`PyBOP` formulates the inference process into four key classes: model, problem, cost (or likelihood), and optimiser (or sampler), as shown in \autoref{fig:classes}. Each of these objects represents a base class with child classes constructing specialised functionality for different workflows. The model class constructs a `PyBaMM` forward model with a specified set of equations, initial conditions, spatial discretisation, and numerical solver. By composing `PyBaMM` directly into `PyBOP`, specialised models can be constructed alongside the standard models that can also be modified for different inference tasks. One such example is spatial re-discretisation, which is required when one or more geometric parameters are being optimised. In this situation, `PyBOP` rebuilds the `PyBaMM` model only when necessary, reducing the total number of rebuilds, providing improved performance. Alongside construction of the forward model, `PyBOP`'s model class provides methods for obtaining sensitivities from the prediction, enabling gradient-based optimisation. A forward prediction, along with its corresponding sensitivities, is provided to the problem class for processing and exception control. A standardised data structure is then provided to the cost classes, which computes a distance, design, or likelihood-based metric for optimisation. For point-based optimisation, the optimisers minimise the cost function or the negative log-likelihood if a likelihood class is provided. Bayesian inference is provided by sampler classes, which accept the `LogPosterior` class and sample from it using `PINTS`-based Monte Carlo algorithms at the time of submission. In the typical workflow, the classes in \autoref{fig:classes} are constructed in sequence, from left to right in the figure.

In addition to the core architecture, `PyBOP` provides several specialised inference and optimisation features. One example is parameter inference from electrochemical impedance spectroscopy (EIS) simulations, where PyBOP discretises and linearises the EIS forward model into a sparse mass matrix form with accompanying auto-differentiated Jacobian. This is then translated into the frequency domain, giving a direct solution to compute the input-output impedance. In this situation, the forward models are constructed within the spatial re-discretisation workflow, allowing for geometric parameter inference from EIS simulations and data.

A second specialised feature is that `PyBOP` builds on the `JAX` [@jax:2018] numerical solvers used by `PyBaMM` by providing `JAX`-based cost functions for automatic forward model differentiation with respect to the parameters. This functionality provides a performance improvement and allows users to harness many other JAX-based inference packages to optimise cost functions, such as `Numpyro` [@numpyro:2019], `BlackJAX` [@blackjax:2024], and `Optax` [@optax:2020].

The currently implemented subclasses for the model, problem, and cost classes are listed in \autoref{tab:subclasses}. The model and optimiser classes can be selected in combination with any problem-cost pair.

:List of available model, problem and cost (or likelihood) classes. \label{tab:subclasses}

| Battery Models                      | Problem Types   | Cost / Likelihood Functions |
|:------------------------------------|:----------------|:----------------------------|
| Single-particle model (SPM)         | Fitting problem | Sum-squared error           |
| SPM with electrolyte (SPMe)         | Design problem  | Root-mean-squared error     |
| Doyle-Fuller-Newman (DFN)           | Observer        | Minkowski                   |
| Many-particle model (MPM)           |                 | Sum-of-power                |
| Multi-species multi-reaction (MSMR) |                 | Gaussian log likelihood     |
| Weppner Huggins                     |                 | Maximum a posteriori        |
| Equivalent circuit model (ECM)      |                 | Volumetric energy density   |
|                                     |                 | Gravimetric energy density  |

Similarly, the current algorithms available for optimisation are presented in \autoref{tab:optimisers}. It should be noted that SciPy minimize includes several gradient and non-gradient methods. From here on, the point-based parameterisation and design-optimisation tasks will simply be referred to as optimisation tasks. This simplification can be justified by comparing \autoref{eqn:parameterisation} and \autoref{eqn:design}; deterministic parameterisation is just an optimisation task to minimise a distance-based cost between model output and measured values.

: Currently supported optimisers classified by candidate solution type, including gradient information. \label{tab:optimisers}

| Gradient-based                                    | Evolutionary                          | (Meta)heuristic           |
|:--------------------------------------------------|:--------------------------------------|:--------------------------|
| Weight decayed adaptive moment estimation (AdamW) | Covariance matrix adaptation (CMA-ES) | Particle swarm (PSO)      |
| Improved resilient backpropagation (iRProp-)      | Exponential natural (xNES)            | Nelder-Mead               |
| Gradient descent                                  | Separable natural (sNES)              | Cuckoo search             |
| SciPy minimize                                    | SciPy differential evolution          |                           |


In addition to deterministic optimisers (\autoref{tab:optimisers}), `PyBOP` also provides Monte Carlo sampling routines to estimate distributions of parameters within a Bayesian framework. These methods construct a posterior parameter distribution that can be used to assess uncertainty and practical identifiability. The individual sampler classes are currently composed within `PyBOP` from the `PINTS` library, with a base sampler class implemented for interoperability and direct integration with `PyBOP`'s model, problem, and likelihood classes. The currently supported samplers are listed in \autoref{tab:samplers}.

: Sampling methods supported by `PyBOP`, classified according to the type of method. \label{tab:samplers}

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

Here, $t$ is time, $\mathbf{x}(t)$ are the (spatially discretised) states, $\mathbf{y}(t)$ are the outputs (e.g. the terminal voltage) and $\mathbf{\theta}$ are the unknown parameters.

Common battery models include various types of equivalent circuit models (e.g. the Thévenin model), the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory, and its reduced-order variants including the single particle model (SPM) [@Planella:2022] and the multi-species multi-reaction (MSMR) model [@Verbrugge:2017]. Simplified models that retain acceptable predictive accuracy at lower computational cost are widely used, for example in battery management systems, while physics-based models are required to understand the impact of physical parameters on performance. This separation of complexity traditionally results in multiple parameterisations for a single battery type, depending on the model structure.

# Examples

## Parameterisation

The parameterisation of battery models is challenging due to the large number of parameters that need to be identified compared to the number of measurable outputs [@Miguel:2021; @Wang:2022; @Andersson:2022]. A complete parameterisation often requires stepwise identification of smaller sets of parameters from a variety of excitations and different data sets [@Chu:2019; @Chen:2020; @Lu:2021; @Kirk:2022]. Furthermore, parameter identifiability can be poor for a given set of excitations and data sets, requiring improved experimental design in addition to uncertainty capable identification methods [@Aitio:2020].

A generic data-fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\hat{\mathbf{y}_i})}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the agreement between the model output $\mathbf{y}(t)$ and a sequence of observations $(\hat{\mathbf{y}_i})$ measured at times $t_i$. Within the `PyBOP` framework, the `FittingProblem` class packages the model output along with the measured observations, both of which are then passed to the cost classes for the computation of the specific cost function. For gradient-based optimisers, the Jacobian of the cost function with respect to unknown parameters, $\partial \mathcal{L} / \partial \theta$, is computed for step-size and directional information.

Next, we demonstrate the fitting of synthetic data where the model parameters are known. Throughout this section, as an example, we use `PyBaMM`'s implementation of the single particle model with an added contact resistance submodel. We assume that the model is already fully parameterised apart from two parameters, namely, the lithium diffusivity of the negative electrode active material particles (denoted "negative particle diffusivity") and the contact resistance with corresponding true values of [3.3e-14 $\text{m}^2/\text{s}$, 10 mOhm]. To start, we generate synthetic time-domain data corresponding to a one-hour discharge from 100% to 0% state of charge, denoted as 1C rate, followed by 30 minutes of relaxation. This dataset is then corrupted with zero-mean Gaussian noise of amplitude 2 mV, with the resulting signal shown by the blue dots in \autoref{fig:inference-time-landscape} (left). The initial states are assumed known, although this assumption is not generally necessary. The `PyBOP` repository contains several other [example notebooks](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks) that follow a similar inference process. The underlying cost landscape to be explored by the optimiser is shown in \autoref{fig:inference-time-landscape} (right), with the initial position denoted alongside the known true system parameters for this synthetic inference task. In general, the true parameters are not known.

![The fitted synthetic dataset (left) and cost landscape (right) for an example time-series battery model parameterisation using a root-mean-squared error cost function. \label{fig:inference-time-landscape}](figures/joss/sim-landscape.pdf){ width=100% }

We can also use `PyBOP` to generate and fit electrochemical impedance data using methods within `pybamm-eis` that enable fast impedance computation of battery models [@pybamm-eis]. Using the same model and parameters as in the time-domain case, \autoref{fig:impedance-landscape} shows the numerical impedance prediction available in `PyBOP` alongside the cost landscape for the corresponding inference task. At the time of publication, gradient-based optimisation and sampling methods are not available when using an impedance workflow.

![The data and model fit (left) and cost landscape (right) for a frequency-domain impedance parameterisation  with a root-mean-squared error cost function, at 5% SOC. \label{fig:impedance-landscape}](figures/joss/impedance.pdf){ width=100% }

To avoid confusion, in the remainder of this section, we continue with identification in the time domain (\autoref{fig:inference-time-landscape}). In general, however, time- and frequency-domain models and data may be combined for improved parameterisation. As gradient information is available for our time-domain example, the choice of distance-based cost function and optimiser is not constrained. Due to the difference in magnitude between the two parameters, we apply the logarithmic parameter transformation offered by `PyBOP`. This transforms the search space of the optimiser to allow for a common step size between the parameters, improving convergence in this particular case. As a demonstration of the parameterisation capabilities of `PyBOP`, \autoref{fig:convergence-min-max} (left) shows the rate of convergence for each of the distance-minimising cost functions, while \autoref{fig:convergence-min-max} (right) shows analogous results for maximising a likelihood. The optimisation is performed with SciPy Minimize using the gradient-based L-BFGS-B method.

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

To estimate the full posterior parameter distribution, however, one must use sampling or other inference methods to reconstruct the function $P(\theta|D)$. The posterior distribution provides information about the uncertainty of the identified parameters, e.g., by calculating the variance or other moments. Monte Carlo methods are used here to sample from the posterior. The selection of Monte Carlo methods available in `PyBOP` includes gradient-based methods such as no-u-turn [@NUTS:2011] and Hamiltonian [@Hamiltonian:2011], as well as heuristic methods such as differential evolution [@DiffEvolution:2006], and also conventional methods based on random sampling with rejection criteria [@metropolis:1953]. `PyBOP` offers a sampler class that provides the interface to samplers, the latter being provided by the probabilistic inference on noisy time-series (`PINTS`) package. \autoref{fig:posteriors} shows the sampled posteriors for the synthetic model described previously, using an adaptive covariance-based sampler called Haario Bardenet [@Haario:2001].

![Posterior distributions of model parameters alongside identified noise on the observations. Shaded areas denote the 95th percentile credible interval for each parameter.  \label{fig:posteriors}](figures/joss/posteriors.pdf){ width=100% }

## Design optimisation

Design optimisation is supported in `PyBOP` to guide device design development by identifying parameter sensitivities that can unlock improvements in performance. This problem can be viewed in a similar way to the parameterisation workflows described previously, but with the aim of maximising a design-objective cost function rather than minimising a distance-based cost function. `PyBOP` performs maximisation by minimising the negative of the cost function. In design problems, the cost metric is no longer a distance between two time series, but a metric evaluated on a model prediction. For example, to maximise the gravimetric energy (or power) density, the cost is the integral of the discharge energy (or power) normalised by the cell mass. Such metrics are typically quantified for operating conditions such as a 1C discharge, at a given temperature.

In general, design optimisation can be written as a constrained optimisation problem,
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ \mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions}),}
\label{eqn:design}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability of the design and $\Omega$ is the set of allowable parameter values.

As an example, we consider the challenge of maximising the gravimetric energy density, subject to constraints on two of the geometric electrode parameters [@Couto:2023]. In this case we use the `PyBaMM` implementation of the single particle model with electrolyte (SPMe) to investigate the impact of the positive electrode thickness and the active material volume fraction on the energy density. Since the total volume fraction must sum to unity, the positive electrode porosity for each optimisation iteration is defined in relation to the active material volume fraction. It is also possible to update the 1C rate corresponding to the theoretical capacity for each iteration of the design.

![Gravimetric energy density cost landscape alongside the initial and optimised voltage profiles, for a fixed-rate (nominally 1C) discharge. \label{fig:design_gravimetric}](figures/joss/design.pdf){ width=100% }

\autoref{fig:design_gravimetric} (left) shows the predicted improvement in the discharge profile between the initial and optimised parameter values and (left) the Nelder-Mead search over the parameter space.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to `PyBOP`. This work was supported by the Faraday Institution Multiscale Modelling project (ref. FIRG059), UKRI's Horizon Europe Guarantee (ref. 10038031), and EU IntelLiGent project (ref. 101069765).

[//]: # (# Open Discussion Points)

[//]: # (- Performance discussion &#40;multiprocessing / JAX&#41;)

[//]: # (- Feasibility checks on identified parameters)

# References
