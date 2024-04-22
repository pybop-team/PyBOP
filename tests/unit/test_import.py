import importlib
import sys
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_multiprocessing_init_non_win32(monkeypatch):
    """Test multiprocessing init on non-Windows platforms"""
    monkeypatch.setattr(sys, "platform", "linux")
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        del sys.modules["pybop"]
        importlib.import_module("pybop")
        mock_set_start_method.assert_called_once_with("fork")


@pytest.mark.unit
def test_multiprocessing_init_win32(monkeypatch):
    """Test multiprocessing init on Windows"""
    monkeypatch.setattr(sys, "platform", "win32")
    del sys.modules["pybop"]
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        importlib.import_module("pybop")
        mock_set_start_method.assert_called_once_with("spawn")
