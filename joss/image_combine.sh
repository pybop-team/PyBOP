convert +append figures/design_gravimetric.png figures/design_prediction.png figures/joss/design.png

convert +append figures/gradient_parameters.png figures/evolution_parameters.png figures/heuristic_parameters.png figures/joss/optimisers_parameters.png

convert +append figures/convergence_minimising.png figures/convergence_maximising.png figures/joss/converge.png

convert +append figures/simulation.png figures/landscape.png figures/joss/sim-landscape.png

convert +append figures/contour_gradient_0.png figures/contour_gradient_1.png figures/contour_gradient_2.png figures/contour_gradient_3.png figures/joss/countour_gradient.png

convert +append figures/contour_evolution_0.png figures/contour_evolution_1.png figures/contour_evolution_2.png figures/contour_evolution_3.png figures/joss/countour_evolution.png

convert +append figures/contour_heuristic_0.png figures/contour_heuristic_1.png figures/contour_heuristic_2.png figures/joss/countour_heuristic.png

convert -append figures/joss/countour_gradient.png figures/joss/countour_evolution.png figures/joss/countour_heuristic.png figures/joss/countour_total.png
