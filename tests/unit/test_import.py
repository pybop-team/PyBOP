import importlib
import sys
from unittest.mock import patch

import pytest

import pybop


@pytest.mark.unit
def test_multiprocessing_init_non_win32(monkeypatch):
    """Test multiprocessing init on non-Windows platforms"""
    monkeypatch.setattr(sys, "platform", "linux")
    print(sys.platform)
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        importlib.reload(pybop)
        mock_set_start_method.assert_called_once_with("fork")


@pytest.mark.unit
def test_multiprocessing_init_win32(monkeypatch):
    """Test multiprocessing init on Windows"""
    monkeypatch.setattr(sys, "platform", "win32")
    with patch("multiprocessing.set_start_method") as mock_set_start_method:
        importlib.reload(pybop)
        mock_set_start_method.assert_called_once_with("spawn")
