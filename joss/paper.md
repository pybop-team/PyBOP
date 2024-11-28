---
title: 'PyBOP: A Python package for battery model optimisation and parameterisation'
tags:
  - Python
  - batteries
  - battery models
  - parameterisation
  - parameter inference
  - design optimisation
authors:
  - name: Brady Planden
    corresponding: true
    orcid: 0000-0002-1082-9125
    affiliation: 1
  - name: Nicola E. Courtier
    affiliation: "1, 2"
    orcid: 0000-0002-5714-1096
  - name: Martin Robinson
    orcid: 0000-0002-1572-6782
    affiliation: 3
  - name: Ferran Brosa Planella
    affiliation: "2, 4"
    orcid: 0000-0001-6363-2812
  - name: David A. Howey
    affiliation: "1, 2"
    orcid: 0000-0002-0620-3955
affiliations:
 - name: Department of Engineering Science, University of Oxford, Oxford, UK
   index: 1
 - name: The Faraday Institution, Harwell Campus, Didcot, UK
   index: 2
 - name: Research Software Engineering Group, University of Oxford, Oxford, UK
   index: 3
 - name:  Mathematics Institute, University of Warwick, Coventry, UK
   index: 4
date: 28 November 2024
bibliography: paper.bib
repository: https://github.com/pybop-team/PyBOP
---

# Summary

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides a set of methods for the parameterisation and optimisation of battery models, offering both Bayesian and frequentist approaches with example workflows to assist the user. `PyBOP` has been developed to enable parameter identification of various battery models, including the electrochemical and equivalent circuit models provided by the popular open-source package `PyBaMM` [@Sulzer:2021].

Similarly, `PyBOP` can be used for parameter design optimisation under user-defined operating conditions across a variety of model structures. `PyBOP` enables battery model parameterisation across a range of methods with diagnostics on the performance and convergence of the identified or optimised parameters. The identified parameters can be used for prediction, on-line control and design optimisation, all of which support improved battery utilisation and development.

# Statement of need

`PyBOP` is a Python package designed to provide a user-friendly, object-oriented interface for the optimisation of battery models. `PyBOP` leverages the open-source `PyBaMM` [@Sulzer:2021] package for formulation and solving of these battery models. `PyBOP` is intended to serve a broad audience of students, engineers, and researchers in both academia and the battery industry, by enabling usage of predictive battery models where not previously possible. `PyBOP` prioritises clear and informative diagnostics and workflows for users of varying expertise, by providing advanced optimisation and sampling algorithms. These methods are provided through interfaces to `PINTS` [@Clerx:2019], `SciPy` [@SciPy:2020], in addition to the PyBOP constructed algorithms such as Adaptive Moment Estimation with Weight Decay (AdamW), and Cuckoo search.

`PyBOP` supports the Battery Parameter eXchange (BPX) standard [@BPX:2023] for sharing battery parameter sets. As these parameter sets are costly to obtain due to: the equipment and time spent on characterisation experiments, the requirement of battery domain knowledge, and the computational cost of parameter estimation. `PyBOP` reduces these costs by enabling fast computational estimation with parameter set interoperability.

This package complements other tools in the field of lithium-ion battery modelling built around `PyBaMM`, such as `liionpack` for simulating battery packs [@Tranter2022] and `pybamm-eis` for numerical impedance spectroscopy as the identified parameters are easily exportable from `PyBOP` into these packages.

# Architecture

`PyBOP` has a tiered data structure aimed at computing and processing the forward model predictions and packaging the required information to the optimisation and sampling algorithms. The forward model is solved using the popular battery modelling package, `PyBaMM`, with construction, parameterisation, and discretisation managed through PyBOP's model interface to PyBaMM. This approach enables a robust object construction process with consistent interfacing between the models and optimisers. The statistical methods and optimisation algorithms are then constructed to interface cleanly with the forward model predictions. Furthermore, identifiability metrics are provided alongside the estimated parameters through Hessian approximation of the cost functions in the frequentist workflows and posterior moments in the Bayesian workflows.

![PyBOP's interface to supporting funding agencies, alongside a visualisation of the general workflow for parameterisation and optimisation \label{fig:high-level}](figures/PyBOP-high-level.pdf){width=80%}

`PyBOP` formulates the inference process into four key classes, namely the model, problem, cost, and optimiser/sampler, as shown in \autoref{fig:classes}. Each of these objects represent a base class with child classes constructing specialised functionality for inference or optimisation workflows. The model class constructs a `PyBaMM` forward model for a given set of model equations provided from `PyBaMM`, initial conditions, spatial discretisation, and numerical solver. By composing `PyBaMM` directly into `PyBOP`, specialised models can be constructed alongside the default models which can be modified, and optimally constructed for the inference tasks. One such example of this, is the spatial rediscretisation that is performed when geometric parameters are optimised. In this situation, `PyBOP` minimally rediscretises the `PyBaMM` model while maintaining the problem, cost, and optimiser objects, providing improved performance benefits to users. Alongside construction of the forward model, `PyBOP`'s model class provides methods for acquiring sensitivities from the prediction, enabling gradient based optimisation algorithms. This prediction alongside it's corresponding sensitivities are provided to the problem class for processing and exception control. A standardised data structure is then provided to the cost classes, which provides a distance, design, or likelihood based metric for optimisation. For deterministic optimisation, the optimisers minimise the corresponding cost function or the negative log-likelihood if a likelihood class is provided. Bayesian inference is provided through Monte Carlo sampling classes, which accept the child cost class, LogPosterior and samples from it using Pints' based Monte Carlo algorithms at the time of submission. In the typical workflow, the classes in \autoref{fig:classes} are constructed in sequence.

In addition to the core architecture, `PyBOP` offers multiple specialised inference and optimisation processes. One such instance is numerical electrochemical impedance spectroscopy predictions by discretising the forward model into sparese mass matrix form with accompanying auto-differentiation generated jacobian. These objects are then translated into the frequency domain with a linear solve used to compute the battery model impedance. In this situation, the forward models are constructed within the spatial rediscretisation workflow, allowing for geometric parameter inference from EIS forward model predictions. Furthermore, `PyBOP` builds upon the JAX [@jax:2018] numerical solvers provided by `PyBaMM` by offering JAX-based cost function for auto-differentiation of the forward model with respect to the parameters. This functionality provides a performance improvement alongside an interface to JAX-based inference packages, such as Numpyro [@numpyro:2019], BlackJAX [@blackjax:2024], and Optax [@optax:2020].

![The core `PyBOP` architecture, showcasing the base class interfaces. Each class provide direct mapping to a classical step in the optimisation workflow. \label{fig:classes}](figures/PyBOP_components.drawio.png){ width=80% }

The currently implemented subclasses for the model, problem, and cost classes are listed in \autoref{tab:subclasses}. The cost functions in \autoref{tab:subclasses} are grouped by problem type, while the model and optimiser classes can be selected in combination with any problem-cost pair.

:List of available model, problem and cost (or likelihood) classes. \label{tab:subclasses}

| Battery Models                      | Problem Types   | Cost / Likelihood Functions |
|:------------------------------------|:----------------|:----------------------------|
| Single particle model (SPM)         | Fitting problem | Sum squared error           |
| SPM with electrolyte (SPMe)         |                 | Root mean squared error     |
| Doyle-Fuller-Newman (DFN)           |                 | Minkowski                   |
| Many particle model (MPM)           |                 | Sum of power                |
| Multi-species multi-reaction (MSMR) |                 | Gaussian log likelihood     |
| Weppner Huggins                     |                 | Maximum a posteriori        |
| Equivalent circuit model (ECM)      | Observer        | Unscented Kalman filter     |
|                                     | Design problem  | Gravimetric energy density  |
|                                     |                 | Volumetric energy density   |


Likewise, the current algorithms available for optimisation tasks are presented in \autoref{tab:optimisers}. From this stage onwards, the point-based parameterisation and design optimisation tasks will simply be referred to as optimisation tasks. This simplification can be justified by inspecting \autoref{eqn:parameterisation} and \autoref{eqn:design} and confirming that deteriminstic parameterisation can be viewed as an optimisation task to minimise a distance-based cost function.

: The currently supported optimisation algorithms classified by candidate solution type, inclusive of gradient information. (*) Scipy minimize has gradient and non-gradient methods. \label{tab:optimisers}

| Gradient-based                                       | Evolutionary Strategies               | (Meta)heuristic      |
|:-----------------------------------------------------|:--------------------------------------|:---------------------|
| Adaptive moment estimation with weight decay (AdamW) | Covariance matrix adaptation (CMA-ES) | Particle swarm (PSO) |
| Improved resilient backpropagation (iRProp-)         | Exponential natural (xNES)            | Nelder-Mead          |
| Gradient descent                                     | Separable natural (sNES)              | Cuckoo search        |
| SciPy minimize (*)                                   | SciPy differential evolution          |                      |
|                                                      |                                       |                      |

 As discussed above, `PyBOP` offers Bayesian inference methods such as Maximum a Posteriori (MAP) presented alongside the point-based methods in \autoref{tab:subclasses}; however, for a full Bayesian framework, Monte Carlo sampling is implemented within `PyBOP`. These methods construct a posterior distribution on the inference parameters which can used for uncertainty and practical identifiability. The individual sampler classes are currently composed within `PyBOP` from the `PINTS` library, with a base sampling class implemented for interoperability and direct integration to the `PyBOP` model, problem, and likelihood classes. The currently supported samplers are presented in \autoref{tab:samplers}.

: PyBOP's supported sampling methods separated based on candidate suggestion method. \label{tab:samplers}

| Hamiltonian-based | Adaptive                   | Slice Sampling | Evolutionary           | Other                        |
|:------------------|:---------------------------|:---------------|:-----------------------|:-----------------------------|
| Monomial Gamma    | Delayed Rejection Adaptive | Doubling       | Differential Evolution | Metropolis Random Walk       |
| No-U-Turn         | Haario Bardenet            | Rank Shrinking |                        | Emcee Hammer                 |
| Hamiltonian       | Haario                     | Stepout        |                        | Metropolis Adjusted Langevin |
| Relativistic      | Rao Blackwell              |                |                        |                              |

# Background

## Battery models

In general, battery models can be written in the form of a differential-algebraic system of equations:
\begin{equation}
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = f(t,\mathbf{x},\mathbf{y},\mathbf{u}(t),\mathbf{\theta}),
\label{dynamics}
\end{equation}
\begin{equation}
\mathbf{y}(t) = g(t,\mathbf{x},\mathbf{y},\mathbf{u}(t),\mathbf{\theta}),
\label{output}
\end{equation}
with initial conditions
\begin{equation}
\mathbf{x}(0) = \mathbf{x}_0(\mathbf{\theta}).
\label{initial_conditions}
\end{equation}

Here, $t$ is time, $\mathbf{x}(t)$ are the (spatially discretised) states, $\mathbf{y}(t)$ are the outputs (for example the terminal voltage), $\mathbf{u}(t)$ are the inputs (e.g. the applied current) and $\mathbf{\theta}$ are the unknown parameters.

Common battery models include various types of equivalent circuit models (e.g. the Thévenin model), the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory and its reduced-order variants including the single particle model (SPM) [@Planella:2022], as well as the multi-species, multi-reaction (MSMR) model [@Verbrugge:2017]. Simplified models that retain acceptable prediction capabilities at a lower computational cost are widely used, for example within battery management systems, while physics-based models are required to understand the impact of physical parameters on battery performance. This separation of complexity conventionally results in multiple parameterisations for a single battery type, dependent on the model structure.

# Examples

## Parameterisation

Battery model parameterisation is challenging due to the high number of parameters required to identify compared to measurable outputs [@Miguel:2021; @Wang:2022; @Andersson:2022]. A complete parameterisation often requires a step-by-step identification of smaller groups of parameters from a variety of different datasets [@Chu:2019; @Chen:2020; @Kirk:2022] and excitations.

A generic data fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\mathbf{y}_i)}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}

in which $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost (or likelihood) function that quantifies the agreement between the model and a sequence of observations $(\mathbf{y}_i)$ measured at times $t_i$. For gradient-based optimisers, the Jacobian of the cost function with respect to the unknown parameters, $(\frac{\partial \mathcal{L}}{\partial \theta})$ is computed for step size and directional information.

We next demonstrate the fitting of synthetic data of which the system parameters are known. In this example problem, we employ `PyBaMM`'s implementation of the single particle model (SPM) with an added contact resistance submodel. We assume that the battery model is already parameterised except for two dynamic parameters, namely the lithium diffusivity of the negative electrode active material particle (denoted "negative particle diffusivity") and the contact resistance. We generate synthetic data from a one-hour discharge from 100% state of charge, to 0% (denoted as 1C rate), followed by 30 minutes of relaxation. This data is then corrupted with zero mean gaussian noise of amplitude 2mV, shown by the dots in \autoref{fig:inference-time-landscape} (left). The initial states are assumed known, although such an assumption is not necessary in general. The underlying cost landscape explored by the optimiser is shown in \autoref{fig:inference-time-landscape} (right).

![The cost landscape for the parameterisation problem using the root mean square error. \label{fig:inference-time-landscape}](figures/joss/sim-landscape.png){ width=100% }

 As gradient information is available for this problem, the choice of distance-based cost function and optimiser is not constrained. Due to the vastly different magnitudes of the two parameters, we apply two of the parameter transformations offered by `PyBOP`, namely the log transformation for the negative particle diffusivity and the scaled transformation (with a coefficient of 100) for the contact resistance. This application transforms the optimisers search space, enabling a shared step-size between the parameters; however, in general is not required. As a demonstration of `PyBOP`'s parameterisation capabilities, \autoref{fig:convergence-min-max} (left) shows the rate of convergence for each of the distance-minimising cost functions, while \autoref{fig:convergence-min-max} (right) displays analogous results for maximising a likelihood. Here, the optimisation is performed with SciPy Minimize using the gradient-based L-BFGS-B method.

![Convergence in the likelihood functions obtained using various likelihood functions and L-BFGS-B algorithm. \label{fig:convergence-min-max}](figures/joss/converge.png){ width=100% }

Furthermore, we can also compare the performance of the various optimisation algorithms divided by category: gradient-based in \autoref{fig:optimiser-inference} (left), evolution strategies in \autoref{fig:optimiser-inference} (middle) and (meta)heuristics in \autoref{fig:optimiser-inference} (right) for a sum squared error cost function. Note that optimiser performance depends on the cost landscape, prior information, and corresponding hyperparameters for each specific problem.

![Convergence in the parameter values obtained for the various optimisation algorithms provided by `PyBOP`.  \label{fig:optimiser-inference}](figures/joss/optimisers_parameters.png){ width=100% }

![Cost landscape contour plot with corresponding optimisation traces. The top row represents the gradient-based optimisers, the middle is the evolution-based, and the bottom is the (meta)heuristics. The order left to right aligns with the entries in \autoref{tab:optimisers}.  \label{fig:optimiser-inference}](figures/joss/contour_total.png){ width=100% }

This parameterisation task can also be approached from the Bayesian perspective, which we will present below using PyBOP's sampler methods. The optimisation equation presented in equation \autoref{eqn:parameterisation} does not represent the Bayesian parameter identification task, and as such we introduce Bayes Theorem as,

\begin{equation}
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
\label{eqn:bayes_theorem}
\end{equation}

where, $P(\theta|D)$ is the posterior and represents the probability density function of the parameter. $P(D|\theta)$ is the likelihood function and assesses the parameter values alongside a noise model. $P(\theta)$ encapsulates the prior knowledge on the parameters, and finally $P(D)$ is the model evidence and acts as a normalising constant such that the final posterior is a correctly scaled density function.
Our aim in parameter inference is to identify the parameter values with the highest probability, which can be presented from point-based metric or as the posterior distribution, which provides additional information on the uncertainty of the identified parameters. To acquire this posterior distribution, we provide Monte-Carlo sampling methods. These methods sample from the posterior through a variety of methods, including gradient-based such as No-U-Turn [@NUTS:2011] and Hamiltonian [@Hamiltonian:2011] as well as heuristic methods such as Differential Evolution [@DiffEvolution:2006], and finally conventional methods based on random sampling with rejection criteria [@metropolis:1953]. PyBOP offers a sampling class which provides an interface for these samplers, which are supported from the Probabilistic Inference of Noise Time-Series (PINTS) package. \autoref{fig:posteriors} below presents the sampled posterior for the synthetic workflow described above, using an adaptive covariance based sampler, Haario Bardenet [@Haario:2001].

![Posterior distributions for model parameters alongside identified noise on the observations. Shaded area denotes confidence bounds for each parameter.  \label{fig:posteriors}](figures/joss/posteriors.png){ width=100% }

## Design optimisation

Design optimisation is supported within `PyBOP` to guide future development of battery design by identifying parameter sensitivities which may unlock improvements in battery performance. This problem can be viewed similarly to the parameterisation workflows described above; however, with the aim of maximising a distance metric instead of minimising it. In the case of design optimisation for maximising gravimetric energy density, `PyBOP` minimises the negative of the cost function, where the cost metric is no longer a distance between two time-series vectors, but instead is the integrated energy from the vector normalised with the corresponding cell mass. This is typically quantified for operational conditions such as a 1C discharge, at a given temperature.

Design optimisation can be written in the form of a constrained optimisation problem as:
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ \mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:design}
\end{equation}
in which $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability
of the design and $\Omega$ is the set of allowable parameter values.

As an example, let us consider the problem of maximising gravimetric energy density subject to constraints on two of the geometric electrode parameters [@Couto:2023]. For this example, we use `PyBaMM`'s implementation of the single particle model with electrolyte (SPMe) to investigate the impact of the positive electrode thickness and active material volume fraction on the target cost. As the active material volume fraction is linked to the electrode porosity, the porosity is defined with a driven reference from the volume fraction. In this problem, we estimate the 1C rate from the theoretical capacity for each iteration of the design. For this example, we employ the particle swarm optimisation (PSO) algorithm.

![The gravimetric landscape alongside the corresponding initial and optimised voltage profiles for a 1C discharge. \label{fig:design_gravimetric}](figures/joss/design.png){ width=100% }

\autoref{fig:design_gravimetric} (left) shows the optimiser's search across the gravimetric energy density parameter space. The predicted improvement in the discharge profile between the initial and optimised parameter values (right) for their respective applied 1C current.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to this package. This work was supported by the Faraday Institution Multiscale Modelling (MSM) project (ref. FIRG059), UKRI's Horizon Europe Guarantee (ref. 10038031), and EU IntelLiGent project (ref. 101069765).

[//]: # (# Discussion Points)

[//]: # (EIS numerical identification)

[//]: # (- Performance discussion &#40;multiprocessing / JAX&#41;)

[//]: # (- Feasibility checks on identified parameters)

# References
