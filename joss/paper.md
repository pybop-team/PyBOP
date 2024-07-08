---
title: 'PyBOP: A Python package for battery model optimisation and parameterisation'
tags:
  - Python
  - batteries
  - battery models
  - parameterisation
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
    affiliation: 4
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
date: 30 June 2024
bibliography: paper.bib
---

# Summary

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides a set of methods for the parameterisation and optimisation of battery models, offering both Bayesian and frequentist approaches with example workflows to assist the user. `PyBOP` can be used for parameter identification of various models, including the electrochemical and equivalent circuit models provided by the popular open-source package `PyBaMM` [@Sulzer:2021]. Similarly, `PyBOP` can be used for design optimisation under user-defined operating conditions for a given parameter. `PyBOP` allows the user to parameterise battery models using a variety of methods and provides diagnostics on the performance and convergence of the optimisation. The identified parameters can be used for prediction, on-line control and design optimisation, all of which support improved battery utilisation and development.

# Statement of need

`PyBOP` is designed to provide a user-friendly, object-oriented interface for the optimisation of battery models which have been implemented in existing battery modelling software, e.g. `PyBaMM` [@Sulzer:2021]. `PyBOP` is intended to serve a broad audience of students, engineers, and researchers in both academia and the battery industry. `PyBOP` prioritises clear and informative diagnostics and workflows for both new and experienced users, while also leveraging advanced optimisation algorithms provided by `SciPy` [@SciPy:2020], `PINTS` [@Clerx:2019], and internal implementations such as the adaptive moment estimation with weight decay (AdamW), as well as Cuckoo search.

`PyBOP` supports the Battery Parameter eXchange (BPX) standard [@BPX:2023] for sharing battery parameter sets. These parameter sets are costly to obtain due to a number of factors: the equipment and time spent on characterisation experiments, the requirement of battery domain knowledge, and the computational cost of parameter estimation. `PyBOP` reduces the barrier to entry and ongoing costs by providing an accessible workflows' that efficiently connects battery models with numerical optimisers, as well as explanatory examples of battery parameterisaton and design optimisation.

This package complements other tools in the field of lithium-ion battery modelling built around `PyBaMM`, such as `liionpack` for simulating battery packs [@Tranter2022] as the identified parameters are easily exportable from `PyBOP` into packages aimed at predictive forward modelling.

# Architecture

`PyBOP` is a Python package packaged through PyPI, which currently supports Python versions 3.9 — 3.12. The package composes the popular battery modelling package `PyBaMM` for forward modelling, while providing classes for parameterisation and optimisation. As shown in \autoref{fig:high-level}, `PyBOP` composes the battery modelling package `PyBaMM`, enabling a consistent interface and robust objection construction process. With the forward model interface construction for parameter identification and optimisation, `PyBOP` provides statistical methods and optimisation algorithms to interface cleanly with the forward model predictions. Furthermore, identifiablity metrics are provided for the estimated parameters through Hessian approximation of the cost/likelihood functions in the frequentist workflows and posterior moments in the Bayesian workflows.

![PyBOP's interface to supporting funding agencies, alongside a visualisation of the general workflow for parameterisation and optimisation \label{fig:high-level}](../assets/PyBOP-high-level.pdf){width=80%}

`PyBOP` formulates the optimisation workflow through four main classes, namely the model, problem, cost, and optimiser or sampler, as shown in \autoref{fig:classes}. Each of these objects represent a base class with children classes constructing differing functionality for specialised parameterisation or optimisation workflows. For example, the model class offers children classes for differing physics-based battery models, as well as emperical models. This allows for the underlying `PyBaMM` model to be constructed and validated for the different requirements between the physics-based models and the emperical. For a given set of model equations provided from `PyBaMM`, initial conditions, spatialy discretisation, and numerical solver initialisation is completed. By composing `PyBaMM` models directly into `PyBOP`, the underlying model structure can be modified, and optimally constructed for the optimisation tasks. One such example of this, is the spatially rediscretiation that is performed with geometric parameters are optimised. In this situation, `PyBOP` aims to minimally reconstruct the `PyBaMM` model while maintaining the problem, cost, and optimisation objects, providing improved performance benefits to end-users. In the typical optimisation workflow, the classes in \autoref{fig:classes} are constructed in sequence.

![The core `PyBOP` architecture, showcasing the base class interfaces. Each class provide direct mapping to a classical step in the optimisation workflow. \label{fig:classes}](PyBOP_components.drawio.png){ width=80% }

The currently implemented subclasses for the model, problem, and cost classes are listed in \autoref{tab:subclasses}. From this point onwards, the parameterisation and optimisation tasks will simply be referred to as an optimisation task. This simplification can be justified by inspecting \autoref{eqn:parameterisation} and \autoref{eqn:design} and confirming parameterisation can be viewed as an minimising optimisation task for a distance-based cost function.

: List of available model, problem and cost classes. \label{tab:subclasses}

| Battery Models                      | Problem Types   | Cost / Likelihood Functions         |
| :---------------------------------- | :-------------- | :----------------------------- |
| Single particle model (SPM)         | Fitting problem | Sum of squared error    |
| SPM with electrolyte (SPMe)         | Design problem  | Root mean squared error  |
| Doyle-Fuller-Newman (DFN)           | Observer        | Gaussian log likelihood            |
| Many particle model (MPM)           |                 | Maximum a posteriori     |
| Multi-species multi-reaction (MSMR) |                 | Unscented Kalman filter   |
| Weppner Huggins                     |                 | Gravimetric energy density   |
| Equivalent circuit model (ECM)      |                 | Volumetric energy density     |


Likewise, the current optimisation algorithms available for usage in optimisation tasks in presented in \autoref{tab:optimisers}. The cost functions in \autoref{tab:subclasses} are grouped by problem type, while the model and optimiser classes can be selected in combination with any problem-cost pair.

: List of available optimisers. (*) Scipy Minimize provides gradient and non-gradient methods. \label{tab:optimisers}

| Gradient-based       | Non-gradient-based          |
| :------------------------------------------- | :------------------------------------------------------- |
| Adaptive moment estimation with weight decay (AdamW) | Covariance matrix adaptation evolution strategy (CMA-ES) |
| Improved resilient backpropagation (iRProp-) | Exponential natural evolution strategy (xNES)    |
| Gradient descent                             | Separable natural evolution strategy (sNES)                       |
| SciPy minimize (*)                           | Particle swarm optimization (PSO)            |
|                                              | SciPy differential evolution     |
|                                              | Nelder-Mead  |
|                                              | Cuckoo search         |

 As previously discussed, PyBOP offers Bayesian inference methods such as Maximum a Posteriori (MAP) presented alongside the frequentist methods in \autoref{tab:optimisers}. A full Bayesian framework is available from a Markov-chain Monte Carlo implemented within PyBOP, capable of providing uncertainty on the inferred parameters. These samplers are currently composed within PyBOP from the Pints' library, with a base class implemented for interopability and direct application to the PyBOP model, problem, and likelihood classes. The currently support MCMC samplers is shown in Table \autoref{tab:samplers}.

: List of available Monte Carlo samplers. \label{tab:samplers}

| Gradient-based       | Non-gradient-based              |
| :------------------------------------------- | :------------------------------------------------------- |
| Adaptive Moment Estimation with Weight Decay (AdamW) | Covariance Matrix Adaptation Evolution Strategy (CMA-ES) |
| Improved Resilient Backpropagation (iRProp-) | Exponential Natural Evolution Strategy (xNES)    |
| Gradient Descent                             | Separable Natural Evolution Strategy (sNES)                       |
| SciPy Minimize (*)                           | Particle Swarm Optimization (PSO)            |
|                                              | SciPy Differential Evolution     |
|                                              | Nelder-Mead  |
|                                              | Cuckoo Search         |

- Performance (multiprocessing)
- Construction of PyBaMM models (geometric and non-geometric identification)
- Feasability checks on identified parameters
- Spatial identification methods?
- Documentation supported at X
- Benchmarks provided at X
- Plotting available via Plotly (cost landscapes, gradient landscapes)
- Test suite provided by pytest (~98% coverage)
- Standalone implementations (Bring your own model)

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

Here, $t$ is time, $\mathbf{x}(t)$ are the spatially (discretised) states, $\mathbf{y}(t)$ are the outputs (for example the
terminal voltage), $\mathbf{u}(t)$ are the inputs (e.g. the applied current) and $\mathbf{\theta}$ are the unknown parameters.

Common battery models include various types of equivalent circuit model (e.g. the Thévenin model),
the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory and its reduced-order
variants including the single particle model (SPM) [@Planella:2022], as well as the multi-scale, multi-reaction
(MSMR) model [@Verbrugge:2017].

Simplified models that retain good prediction capabilities at a lower computational cost are widely used, for
example within battery management systems, while physics-based models are required to understand the impact of
design parameters on battery performance.

# Examples

## Parameterisation

Battery model parameterisation is difficult due to the high ratio of the number of parameters to
measurable outputs [@Miguel:2021; @Wang:2022; @Andersson:2022]. A complete parameterisation often requires
a step-by-step identification of smaller groups of parameters from a variety of different datasets
[@Chu:2019; @Chen:2020; @Kirk:2022].

A generic data fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\mathbf{y}_i)}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}
in which $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost (or likelihood) function that quantifies the agreement between the model and a sequence of data points $(\mathbf{y}_i)$ measured at times $t_i$. For gradient-based optimisers, the Jacobian of the cost function with respect to the unknown parameters, $(\frac{\partial L}{\partial \theta})$ is used as a directional metric for the algorithm when exploring the parameter space.

By way of example, we next demonstrate the fitting of some synthetic data for which we know the
true parameter values.

## Design optimisation

Design optimisation is supported within `PyBOP` to
guide future development of the battery design by identifying parameter variations which may unlock
improvements in battery performance. This optimisation task can be viewed similarly to the parameterisation workflows described above, however, with the aim of increasing the distance metric instead of minimising it. In the case of design optimisation for maximising gravimetric energy density, `PyBOP` minimises the negative of the cost function, where the cost metric is no longer a distance between two time-series vectors, instead it is the integrated energy from the vector normalised with the corresponding cell mass. This is typically quantified for operational conditions such as a 1C (the applied current required to discharge the cell in one hour) capacity.

Design optimisation can be written in the form of a constrained optimisation problem as:
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ \mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:design}
\end{equation}
in which $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability
of the design and $\Omega$ is the set of allowable parameter values.

As an example, let us consider the target of maximising gravimetric energy density subject to
constraints on the geometric electrode parameters [@Couto:2023].

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to this
package. This work was supported by the Faraday Institution Multiscale Modelling (MSM)
project (ref. FIRG059), UKRI's Horizon Europe Guarantee (ref. 10038031), and EU IntelLiGent project (ref. 101069765).

# References
