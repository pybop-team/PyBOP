pdfjam --nup 2x1 figures/design_prediction.pdf figures/design_gravimetric.pdf --landscape --outfile figures/joss/design.pdf --papersize '{450px,900px}'

pdfjam --nup 3x1 figures/gradient_parameters.pdf figures/evolution_parameters.pdf figures/heuristic_parameters.pdf --landscape --outfile figures/joss/optimisers_parameters.pdf --papersize '{768px,1296px}'

pdfjam --nup 2x1 figures/convergence_minimising.pdf figures/convergence_maximising.pdf --landscape --outfile figures/joss/converge.pdf --papersize '{450px,900px}'

pdfjam --nup 2x1 figures/simulation.pdf figures/landscape.pdf --landscape --outfile figures/joss/sim-landscape.pdf --papersize '{450px,900px}'

pdfjam --nup 2x1 figures/impedance_spectrum.pdf figures/impedance_contour.pdf --landscape --outfile figures/joss/impedance.pdf --papersize '{450px,900px}'
