#
# Root of the pybop module.
# Provides access to all shared functionality (models, solvers, etc.).
#
# The code in this file is adapted from pybamm
# (see https://github.com/pybamm-team/pybamm)
#

import sys
import os


#
# Version info
#
from pybop.version import __version__


#
# Constants
#
# Float format: a float can be converted to a 17 digit decimal and back without
# loss of information
FLOAT_FORMAT = "{: .17e}"
# Absolute path to the PyBOP repo
script_path = os.path.abspath(__file__)

#
# Model Classes
#
from .models.base_model import BaseModel
from .models.spm import BaseSPM

# 
# Simulation class
#
from .simulation import Simulation

#
# Remove any imported modules, so we don't expose them as part of pybop
#
del sys