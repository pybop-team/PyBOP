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
        # import pybop.__init__ as init
        mock_set_start_method.assert_called_once_with("spawn")


def test_multiprocessing_import_error(capsys):
    with patch.dict(sys.modules, {"multiprocessing": None}):
        with pytest.raises(ImportError) as excinfo:
            importlib.reload(pybop)

        assert "multiprocessing" in str(excinfo.value)

        captured = capsys.readouterr()
        assert "Warning: multiprocessing module not available:" in captured.out
