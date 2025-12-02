import sys

import pytest


class TestImport:
    pytestmark = pytest.mark.unit

    def unload_pybop(self):
        """
        Unload pybop and its sub-modules. Credit PyBaMM team:
        https://github.com/pybamm-team/PyBaMM/blob/90c1c357a97dfd5c8c6a9092a70dddf0dac978db/tests/unit/test_util.py
        """
        # Unload pybop and its sub-modules
        for module_name in list(sys.modules.keys()):
            base_module_name = module_name.split(".")[0]
            if base_module_name == "pybop":
                sys.modules.pop(module_name)
