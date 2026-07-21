# Plotting backend default
DEFAULT_BACKEND = 'matplotlib'
current_backend=DEFAULT_BACKEND

from .util import (
    use_backend,
    get_backend,
    get_backend_from_figure,
    parse_data,
    remove_brackets,
    wrap_text
)

#
# Import plots
#
from .trajectories import trajectories
from .contour import contour
from .dataset import dataset
from .convergence import convergence
from .parameters import parameters
from .problem import problem
from .nyquist import nyquist
from .voronoi import surface
from .samples import trace, chains, posterior, summary_table
from .predictive import predictive
from .distribution import distribution

# Import backend specific plotting functions
from . import  backends
