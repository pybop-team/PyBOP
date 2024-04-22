import importlib
import sys
from unittest.mock import patch

import pytest


class TestImport:
    @pytest.mark.unit
    def test_multiprocessing_init_non_win32(self, monkeypatch):
        """Test multiprocessing init on non-Windows platforms"""
        monkeypatch.setattr(sys, "platform", "linux")
        # Unload pybop and its sub-modules
        self.unload_pybop()
        with patch("multiprocessing.set_start_method") as mock_set_start_method:
            importlib.import_module("pybop")
            mock_set_start_method.assert_called_once_with("fork")

    @pytest.mark.unit
    def test_multiprocessing_init_win32(self, monkeypatch):
        """Test multiprocessing init on Windows"""
        monkeypatch.setattr(sys, "platform", "win32")
        self.unload_pybop()
        with patch("multiprocessing.set_start_method") as mock_set_start_method:
            importlib.import_module("pybop")
            mock_set_start_method.assert_called_once_with("spawn")

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
