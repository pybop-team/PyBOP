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
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Nicola E. Courtier
    affiliation: "1, 2"
    orcid: 0000-0002-5714-1096
  - name: Martin Robinson
    orcid: 0000-0002-1572-6782
    affiliation: 3
  - name: David A. Howey
    affiliation: "1, 2"
    orcid: 0000-0002-0620-3955
affiliations:
 - name: Department of Engineering Science, University of Oxford, Oxford, UK
   index: 1
 - name: The Faraday Institution, Harwell Campus, Didcot, UK
   index: 2
 - name: Head of Research Software Engineering, University of Oxford, Oxford, UK
   index: 3
date: 30 June 2024
bibliography: paper.bib
---

# Summary

`PyBOP` provides a range of tools for the parameterisation and optimisation of battery models, offering
both Bayesian and frequentist approaches with example workflows to assist the user. `PyBOP` can be
used to parameterise various battery models, including the electrochemical and equivalent circuit
models provided by the popular open-source package `PyBaMM` [@Sulzer:2021]. Likewise, PyBOP can be used for design optimisation for a given parameter set under predefined operating conditions. PyBOP allows users to parameterise battery models through a variety of methods, providing diagnostics into the convergence of the optimisation task. These identified parameter sets can then be used for design optimisation to support development of improved battery configurations.

- PyBOP incorporates a PDE solver, and parameterisation/optimisation workflows into a single package.
- PyBOP provide identifiablity estimates for the identified parameter set (Hessian approximation, fisher information?, Posterior variance, CI upper/lower)

# Statement of need

`PyBOP` is designed to provide a user-friendly, object-oriented interface for the optimisation of
battery models which have been implemented in existing battery modelling software, e.g. `PyBaMM` [@Sulzer:2021].
This software package is intended to serve a broad audience of students, engineers, and researchers in both
academia and the battery industry. `PyBOP` prioritises clear and informative diagnostics for both
new and experienced users, while also leveraging advanced optimisation algorithms provided by `SciPy`
[@SciPy:2020], `PINTS` [@Clerx:2019], and internal implementations.

`PyBOP` supports the Battery Parameter eXchange (BPX) standard [@BPX:2023] for sharing battery
parameter sets. These parameter sets are costly to obtain due to a number of factors: the equipment
cost and time spent on characterisation experiments, the requirement of battery domain knowledge
and the computational cost of parameter estimation. `PyBOP` reduces the barrier to entry and ongoing
costs by providing an accessible workflow that efficiently connects battery models with numerical
optimisers, as well as explanatory examples of battery parameterisaton and design optimisation.

This package complements other tools in the field of lithium-ion battery modelling built around
`PyBaMM` such as `liionpack` for simulating battery packs [@Tranter2022].

# Architecture

PyBOP is a Python package provided through PyPI, currently available for Python versions 3.9 to 3.12. The package composes the popular battery modelling package, PyBaMM for battery model numerical solutions, while providing the parameterisation and optimisation workflows. These workflows are constructed through a mixture of internal algorithms, as well as popular optimisation packages such as Pints and SciPy.
The PyBOP framework consists of 4 main classes of Python object, namely the Model, Problem, Cost,
and Optimiser classes, as shown in \autoref{fig:objects}. Each of these objects has a base class
and example subclasses that combine to form a flexible and extensible codebase. The typical workflow
would be to define an optimisation problem by constructing the objects in sequence.

![The core PyBOP classes and how they interact.\label{fig:objects}](PyBOP_components.drawio.png){ width=100% }

The current instances for each class are listed in \autoref{tab:subclasses} and \autoref{tab:optimisers}.

: List of preset subclasses for the model, problem and cost classes. \label{tab:subclasses}

| Battery Models                      | Problem Types   | Cost Functions                 |
| :---------------------------------- | :-------------- | :----------------------------- |
| Single Particle Model (SPM)         | Fitting Problem | Sum of Squared Errors (SSE)    |
| SPM with Electrolyte (SPMe)         | Observer        | Root Mean Squared Error (RMSE) |
| Doyle-Fuller-Newman (DFN)           | Design Problem  | Gaussian Log Likelihood            |
| Many Particle Model (MPM)           |                 | Maximum a Posteriori (MAP)     |
| Multi-Species Multi-Reaction (MSMR) |                 | Unscented Kalman Filter (UKF)  |
| Equivalent Circuit Models (ECM)     |                 | Gravimetric Energy Density     |
|                                     |                 | Volumetric Energy Density      |

: List of available optimisers. (*) Note that Scipy Minimize provides both gradient and non-gradient-based methods. \label{tab:optimisers}

| Gradient-based algorithms       | Non-gradient-based algorithms               |
| :------------------------------------------- | :------------------------------------------------------- |
| Adaptive Moment Estimation with Weigth Decay (AdamW) | Covariance Matrix Adaptation Evolution Strategy (CMA-ES) |
| Improved Resilient Backpropagation (iRProp-) | Exponential Natural Evolution Strategy (xNES)            |
| Gradient Descent                             | Nelder-Mead                                              |
| SciPy Minimize (*)                           | Particle Swarm Optimization (PSO)                        |
|                                              | SciPy Differential Evolution                             |
|                                              | Separable Natural Evolution Strategy (sNES)              |
|                                              | (pending) Cuckoo Search                                  |

The cost functions are grouped by problem type, while each of the models and optimisers may be selected in combination with
any problem-cost pair.

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

Here, $t$ is time, $\mathbf{x}(t)$ are the (discretised) states, $\mathbf{y}(t)$ are the outputs (for example the
terminal voltage), $\mathbf{u}(t)$ are the inputs (e.g. the applied current) and $\mathbf{\theta}$ are the
uncertain parameters.

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
\min_{\mathbf{\theta}} ~ L_{(\mathbf{\hat{y}}_i)}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\end{equation}
in which $L : \mathbf{\theta} \mapsto [0,\infty)$ is a cost (or likelihood) function that quantifies the
agreement between the model and a sequence of data points $(\mathbf{\hat{y}}_i)$ measured at times $t_i$.
For gradient-based optimisers, the gradient refers to the Jacobian of the cost function with respect to the
uncertain parameters, $\mathbf{\theta}$.

By way of example, we next demonstrate the fitting of some synthetic data for which we know the
true parameter values.

## Design optimisation

Once a battery model has been parameterised, design optimisation can be performed in order to
guide future development of the battery design by identifying parameter variations which may unlock
improvements in battery performance. Battery performance is typically quantified via metrics such as
a 1C discharge capacity.

Design optimisation can be written in the form of a constrained optimisation problem as:
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ L(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\end{equation}
in which $L : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability
of the design and $\Omega$ is the set of allowable parameter values.

As an example, let us consider the target of maximising gravimetric energy density subject to
constraints on the geometric electrode parameters [@Couto:2023].

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP?tab=readme-ov-file#contributors-) to this
package. This work was supported by the Faraday Institution Multiscale Modelling (MSM)
project (grant number FIRG059) and the EU IntelLiGent project.

# References
