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
date: 19 April 2024
bibliography: paper.bib
---

# Summary

`PyBOP` offers a range of tools for the parameterisation and optimisation of battery models, using
both Bayesian and frequentist approaches with example workflows to assist the user. `PyBOP` can be
used to parameterise various battery models, including the electrochemical and equivalent circuit
models provided by the complementary open-source package `PyBaMM` [@Sulzer:2021].

# Statement of need

`PyBOP` is designed to provide a user-friendly, object-oriented interface for the optimisation of
battery models implemented in existing battery modelling software, e.g. `PyBaMM` [@Sulzer:2021].
This software package is intended to serve a broad audience of students and researchers in both
academia and the battery industry. `PyBOP` prioritises clear and informative diagnostics for both
new and experienced users, while also leveraging advanced optimisation algorithms provided by `SciPy`
[@SciPy:2020] and `PINTS` [@Clerx:2019].

`PyBOP` supports the Battery Parameter eXchange (BPX) standard [@BPX:2023] for sharing battery 
parameter sets. These parameter sets are costly to obtain due to a number of factors: the equipment
cost and time spent on characterisation experiments, the requirement of battery domain knowledge
and the computational cost of parameter estimation. `PyBOP` reduces the entry barrier and ongoing
costs by providing an accessible workflow that efficiently connects battery models with numerical
optimisers, as well as explanatory examples of battery parameterisaton and design optimisation.

<!-- # Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png){ width=80% }
and referenced from text using \autoref{fig:example}. -->

# Mathematical background

## Battery models

In general, battery models can be written in the form of a differential-algebraic system of equations:

\begin{equation}
\frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t} = f(t,\mathbf{x},\mathbf{y},\mathbf{u}(t),\mathbf{\theta})
\end{equation}
\begin{equation}
\mathbf{y}(t) = g(t,\mathbf{x},\mathbf{y},\mathbf{u}(t),\theta)
\end{equation}

Here, $t$ is time, $x(t)$ are the (discretised) states, $y(t)$ are the outputs (for example the
terminal voltage), $u(t)$ are the inputs (for example the applied current) and $\theta$ are the
parameters.

## Parameterisation

Battery model parameterisation is difficult due to the high ratio of the number of parameters to
measurable outputs. A complete parameterisation often requires a step-by-step identification of
smaller groups of parameters from a variety of different datasets.

## Design optimisation

Once a battery model has been parameterised, design optimisation can be performed in order to
guide future development of the battery design by identifying parameter variations which may unlock
improvements in battery performance. Battery performance is typically quantified via metrics such as
a 1C discharge capacity.

# Acknowledgements

We gratefully acknowledge all [contributors](https://github.com/pybop-team/PyBOP) to this
package. This work was supported by the Faraday Institution Multiscale Modelling (MSM)
project (grant number FIRG059) and the EU IntelLiGent project.

# References
