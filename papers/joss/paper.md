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

The Python Battery Optimisation and Parameterisation (`PyBOP`) package provides methods for estimating and optimising battery model parameters, offering both deterministic and stochastic approaches with example workflows. `PyBOP` enables parameter identification from data for various battery models, including electrochemical and equivalent circuit models from the open-source `PyBaMM` package [@Sulzer:2021]. The same approaches enable design optimisation under user-defined operating conditions across various model structures and design goals. `PyBOP` facilitates optimisation and provides diagnostics to examine optimiser performance and convergence of the cost and parameters. Identified parameters can be used for prediction, online estimation and control, and design optimisation, accelerating battery research and development.

# Statement of need

`PyBOP` provides a user-friendly, object-oriented interface for optimising battery model parameters. It leverages the open-source `PyBaMM` package [@Sulzer:2021] to formulate and solve battery models. Together, these tools serve a broad audience including students, engineers, and researchers in academia and industry, enabling advanced applications without specialised knowledge of battery modelling, parameter inference, or software development. `PyBOP` emphasises clear diagnostics and workflows to support users with varying domain expertise, and provides access to numerous optimisation and sampling algorithms. These capabilities are enabled through interfaces to `PINTS` [@Clerx:2019], `SciPy` [@SciPy:2020], and `PyBOP`'s implementations of algorithms including adaptive moment estimation with weight decay (AdamW) [@Loshchilov:2017], gradient descent [@Cauchy:1847], and cuckoo search [@Yang:2009].

`PyBOP` complements other lithium-ion battery modelling packages built around `PyBaMM`, such as `liionpack` for battery pack simulation [@Tranter2022] and `pybamm-eis` for fast numerical computation of the electrochemical impedance of any battery model, as well as the battery parameter exchange (BPX) standard [@BPX:2023]. Identified `PyBOP` parameters are easily exported to other packages.

# Architecture

`PyBOP` is structured around four core components: a `Simulator`, `Cost`, `Problem`, and `Optimiser`/`Sampler`, as shown in \autoref{fig:classes}. The purpose of the `Simulator` is to generate model predictions. For example, `pybop.pybamm.Simulator` interfaces with `PyBaMM` to efficiently construct, discretise and numerically solve a `PyBaMM` model for candidate parameter values. Custom or built-in `Cost` classes evaluate an error measure, likelihood or design metric for the candidate parameter values and simulation result. Multiple costs can be summed with optional weighting. The `Problem` class coordinates simulator and cost evaluation, and the `Optimiser`/`Sampler` classes perform parameter inference through optimisation algorithms or Monte Carlo sampling. This structure ensures extensibility for new optimisation problems with a consistent interface between models and optimisers.

![The core `PyBOP` architecture with base class interfaces. Each class provides a direct mapping to a step in the optimisation workflow. Note that 'inputs' means candidate parameter values being optimised, following `PyBaMM` conventions. \label{fig:classes}](figures/PyBOP_components.drawio.pdf){ width=90% }

The `pybamm.Simulator` object returns a solution with corresponding sensitivities, where possible, to enable gradient-based optimisation. Bayesian inference is provided by sampler classes, with Monte Carlo algorithms provided by `PINTS`. In the typical workflow, the classes in \autoref{fig:classes} are constructed in sequence, from left to right. The optimisation result includes a log of the candidate parameters and corresponding cost values. Beyond convergence information, identifiability metrics are provided through Hessian approximation and Sobol sampling from the `SAlib` package.

Beyond the core architecture, `PyBOP` provides specialised inference and optimisation features. Parameter inference from electrochemical impedance spectroscopy (EIS) simulations is handled through  `pybop.pybamm.EISSimulator`, which discretises and linearises the EIS forward model into sparse mass matrix form with an auto-differentiated Jacobian. The result is returned in the frequency domain and is compatible with the same cost classes as in the time-domain simulations.

The currently available optimisation algorithms are presented in \autoref{tab:optimisers}. Note that `SciPy` minimize includes several gradient-based and gradient-free methods. Hereafter, point-based parameterisation and design-optimisation tasks are referred to as optimisation tasks. This simplification can be justified by comparing \autoref{eqn:parameterisation} and \autoref{eqn:design}; deterministic parameterisation is an optimisation task to minimise distance-based cost between model output and measured values.

: Currently supported optimisers classified by optimisation type. \label{tab:optimisers}

| Gradient-based                                    | Evolutionary                          | (Meta)heuristic      |
|:--------------------------------------------------|:--------------------------------------|:---------------------|
| Weight decayed adaptive moment estimation (AdamW) | Covariance matrix adaptation (CMA-ES) | Particle swarm (PSO) |
| Gradient descent                                  | Exponential natural (xNES)            | Nelder-Mead          |
| `SciPy` minimize                                  | Separable natural (sNES)              | Cuckoo search        |
| Improved resilient backpropagation (iRProp-/+)    | `SciPy` differential evolution        | Simulated annealing  |


Beyond deterministic optimisers (\autoref{tab:optimisers}), `PyBOP` provides Monte Carlo sampling methods to estimate parameter distributions within a Bayesian framework. These methods estimate posterior parameter distributions that can be used to assess uncertainty and practical identifiability. Individual sampler classes are composed from the `PINTS` library, with a base sampler class implemented for interoperability and direct integration with the `Problem` class. Currently supported samplers are listed in \autoref{tab:samplers}.

: Sampling methods supported by `PyBOP`, classified according to candidate proposal method. \label{tab:samplers}

| Gradient-based    | Adaptive                   | Slicing        | Other                        |
|:------------------|:---------------------------|:---------------|:-----------------------------|
| Monomial gamma    | Delayed rejection adaptive | Rank shrinking | Metropolis adjusted Langevin |
| No-U-turn         | Haario Bardenet            | Doubling       | Emcee hammer                 |
| Hamiltonian       | Haario                     | Stepout        | Metropolis random walk       |
| Relativistic      | Rao Blackwell              |                | Differential evolution       |


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

Here, $t$ is time, $\mathbf{x}(t)$ are the (spatially discretised) states, $\mathbf{y}(t)$ are the outputs (e.g., the terminal voltage) and $\mathbf{\theta}$ are the parameters. Here the model input(s) (such as current) are implicitly part of the state vector.

Common battery models include equivalent circuits (e.g., the Thévenin model), the Doyle–Fuller–Newman (DFN) model [@Doyle:1993; @Fuller:1994] based on porous electrode theory, and its reduced-order variants including the single particle model (SPM) [@Planella:2022] and the multi-species multi-reaction (MSMR) model [@Verbrugge:2017]. Simplified models that retain acceptable predictive accuracy at lower computational cost are widely used, for example in battery management systems, while physics-based models are required to understand the impact of physical parameters on performance. However, different model structures will lead to different parameter estimates from the same dataset for parameters-in-common, such as diffusion time or series resistance.

# Examples

## Parameterisation

Battery model parameterisation is challenging due to the large number of parameters compared to the number of possible measurements [@Miguel:2021; @Wang:2022; @Andersson:2022]. A complete parameterisation often requires stepwise identification of parameter subsets from a variety of excitations and datasets [@Chu:2019; @Chen:2020; @Lu:2021; @Kirk:2022]. Parameter identifiability can be poor for some excitations and datasets, requiring improved experimental design and uncertainty-capable identification methods [@Aitio:2020].

A generic data-fitting optimisation problem may be formulated as:
\begin{equation}
\min_{\mathbf{\theta}} ~ \mathcal{L}_{(\hat{\mathbf{y}_i})}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions})}
\label{eqn:parameterisation}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the agreement between the model output $\mathbf{y}(t)$ and a sequence of observations $(\hat{\mathbf{y}_i})$ measured at times $t_i$. For gradient-based optimisers, the Jacobian of the cost function with respect to unknown parameters, $\partial \mathcal{L} / \partial \theta$, is computed for step-size and directional information.

We demonstrate the fitting of synthetic data where the model parameters are known, using `PyBaMM`'s SPM with contact resistance. We target two parameters: the lithium diffusivity in the negative electrode active material particles ("negative particle diffusivity") and the contact resistance, with true values [3.3e-14 $\text{m}^2/\text{s}$, 10 m$\text{\Omega}$]. We generate time-domain data for a one-hour discharge from 100% to 0% state of charge (1C rate) followed by 30 minutes relaxation. The output voltage is corrupted with zero-mean Gaussian noise of amplitude 2 mV (blue dots in \autoref{fig:inference-time-landscape} (left)). Initial states are assumed known, although this is not generally necessary. The `PyBOP` repository contains [example notebooks](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks) illustrating similar inference processes. The underlying cost landscape to be explored by the optimiser is shown in \autoref{fig:inference-time-landscape} (right), with the initial position and true values marked. In general, the true values are unknown.

![A synthetic dataset (left) and cost landscape (right) depicting a time-series parameterisation problem using the root-mean-squared error cost function. \label{fig:inference-time-landscape}](figures/combined/sim-landscape.pdf){ width=100% }

`PyBOP` can generate and fit EIS data using methods from `pybamm-eis` [@pybamm-eis]. Using `PyBaMM`'s SPM with double-layer capacitance and contact resistance, \autoref{fig:impedance-landscape} shows numerical EIS predictions alongside the cost landscape for the corresponding inference task. At the time of publication, gradient-based optimisation and sampling methods are not available for EIS simulators.

![Data and model fit (left) and cost landscape (right) for a frequency-domain EIS parameterisation, at 5% SOC, using the root-mean-squared error cost function. \label{fig:impedance-landscape}](figures/combined/impedance.pdf){ width=100% }

We continue here with time-domain identification (\autoref{fig:inference-time-landscape}), however time- and frequency-domain problems may be combined for improved parameterisation. As gradient information is available for our time-domain example, the choice of distance-based cost function and optimiser is unconstrained. Due to the difference in magnitude between the two parameters, we apply a logarithmic transformation that transforms the search space to allow for a common step size, improving convergence. As a demonstration of `PyBOP`'s parameterisation capabilities, \autoref{fig:convergence-min-max} (left) shows convergence rates for distance-minimising cost functions, while \autoref{fig:convergence-min-max} (right) shows analogous results for likelihood maximisation. Optimisation is performed using `SciPy` minimize with the gradient-based BFGS method.

![Convergence of the BFGS method for various cost (left) and likelihood (right) functions. \label{fig:convergence-min-max}](figures/combined/converge.pdf){ width=100% }

Using the same model and parameters, we compare example convergence rates of various algorithms across several categories: gradient-based methods in \autoref{fig:optimiser-inference1} (left), evolutionary strategies in \autoref{fig:optimiser-inference1} (middle) and (meta)heuristics in \autoref{fig:optimiser-inference1} (right) using a mean-squared-error cost. \autoref{fig:optimiser-inference2} shows the optimiser's exploration of the cost landscape, with the three rows showing the gradient-based optimisers (top), evolution strategies (middle), and (meta)heuristics (bottom). Optimiser performance depends on the cost landscape, initial guess or prior for each parameter, and the hyperparameters for each problem.

![Convergence in the parameter values for optimisation algorithms available in `PyBOP`. \label{fig:optimiser-inference1}](figures/combined/optimisers_parameters.pdf){ width=100% }

![Cost landscape plots showing the optimisation traces of 12 different optimisers. \label{fig:optimiser-inference2}](figures/combined/contour_subplot.pdf){ width=100% }

This example parameterisation task can also be approached from a Bayesian perspective, using `PyBOP`'s sampler methods. First, we introduce Bayes' rule,

\begin{equation}
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)},
\label{eqn:bayes_theorem}
\end{equation}

where $P(\theta|D)$ is the posterior parameter distribution, $P(D|\theta)$ is the likelihood function, $P(\theta)$ is the prior parameter distribution, and $P(D)$ is the model evidence, or marginal likelihood, which acts as a normalising constant. For maximum likelihood estimation or maximum a posteriori estimation, one wishes to maximise $P(D|\theta)$ or $P(\theta|D)$, respectively, formulated as an optimisation problem as per \autoref{eqn:parameterisation}.

To estimate the full posterior parameter distribution, however, one must use sampling or other inference methods to reconstruct $P(\theta|D)$. The posterior distribution provides information about the uncertainty of the identified parameters, e.g., by calculating the variance or other moments. Monte Carlo methods available from the probabilistic inference on noisy time-series (`PINTS`) package include gradient-based methods such as no-u-turn [@NUTS:2011] and Hamiltonian [@Hamiltonian:2011], heuristic methods such as differential evolution [@DiffEvolution:2006], and conventional methods based on random sampling with rejection criteria [@metropolis:1953]. \autoref{fig:posteriors} shows sampled posteriors for the synthetic model using an adaptive covariance-based sampler called Haario Bardenet [@Haario:2001].

![Posterior distributions of model parameters and observation noise; shaded areas show the 95th percentile credible interval. \label{fig:posteriors}](figures/combined/posteriors.pdf){ width=100% }

## Design optimisation

`PyBOP` supports design optimisation to guide device design development by identifying parameter sensitivities that can unlock improvements in performance. Design workflows are similar to parameterisation workflows, but the aim is to maximise a design metric rather than minimise a distance-based cost function. `PyBOP` performs maximisation by minimising the negative cost. An example design metric is the gravimetric energy (or power) density given by the integral of the discharge energy (or power) normalised by the cell mass. Such metrics are typically quantified for operating conditions such as a 1C discharge, at a given temperature.

In general, design optimisation can be written as a constrained optimisation problem,
\begin{equation}
\min_{\mathbf{\theta} \in \Omega} ~ -\mathcal{L}(\mathbf{\theta}) ~~~
\textrm{subject to equations (\ref{dynamics})\textrm{-}(\ref{initial_conditions}),}
\label{eqn:design}
\end{equation}

where $\mathcal{L} : \mathbf{\theta} \mapsto [0,\infty)$ is a cost function that quantifies the desirability of the design and $\Omega$ is the set of allowable parameter values.

We consider maximising gravimetric energy density subject to constraints on two of the geometric electrode parameters [@Couto:2023]. We use the `PyBaMM` single particle model with electrolyte (SPMe) to investigate the impact of positive electrode thickness and active material volume fraction on energy density. Since the total volume fraction must sum to unity, the positive electrode porosity is defined relative to the active material volume fraction. The 1C rate can also be optimised (via the nominal capacity parameter) or defined as a function of the parameters for each design.

![Initial and optimised voltage profiles alongside the gravimetric energy density cost landscape. \label{fig:design_gravimetric}](figures/combined/design.pdf){ width=100% }

\autoref{fig:design_gravimetric} (left) shows the predicted improvement in the discharge profile between the initial and optimised parameter values for a fixed-rate 1C discharge selected from the initial design and (right) the Nelder-Mead search over the parameter space.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP/graphs/contributors) to `PyBOP`. This work was supported by the Faraday Institution Multiscale Modelling project (FIRG059), UKRI's Horizon Europe Guarantee (10038031), and EU IntelLiGent project (101069765).

# References
