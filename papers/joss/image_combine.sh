pdfjam --nup 2x1 figures/individual/design_prediction.pdf figures/individual/design_gravimetric.pdf --landscape --outfile figures/combined/design.pdf --papersize '{420px,900px}'

pdfjam --nup 3x1 figures/individual/gradient_parameters.pdf figures/individual/evolution_parameters.pdf figures/individual/heuristic_parameters.pdf --landscape --outfile figures/combined/optimisers_parameters.pdf --papersize '{900px,1440px}'

pdfjam --nup 2x1 figures/individual/convergence_minimising.pdf figures/individual/convergence_maximising.pdf --landscape --outfile figures/combined/converge.pdf --papersize '{400px,900px}' --trim '0cm 0cm 0cm 1.5cm' --clip true

pdfjam --nup 2x1 figures/individual/simulation.pdf figures/individual/landscape.pdf --landscape --outfile figures/combined/sim-landscape.pdf --papersize '{420px,900px}'

pdfjam --nup 2x1 figures/individual/impedance_spectrum.pdf figures/individual/impedance_contour.pdf --landscape --outfile figures/combined/impedance.pdf --papersize '{420px,900px}'
