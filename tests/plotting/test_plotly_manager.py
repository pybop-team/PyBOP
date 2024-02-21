from importlib.metadata import distributions
from distutils.spawn import find_executable
from pybop import PlotlyManager
import subprocess
import plotly
import pytest

# Find the Python executable
python_executable = find_executable("python")


@pytest.fixture(scope="session")
def plotly_installed():
    """A session-level fixture that ensures Plotly is installed after tests."""
    # Check if Plotly is initially installed
    initially_installed = is_package_installed("plotly")

    # If Plotly is not installed initially, install it
    if not initially_installed:
        subprocess.check_call([python_executable, "-m", "pip", "install", "plotly"])

    # Yield control back to the tests
    yield

    # After tests, if Plotly was not installed initially, uninstall it
    if not initially_installed:
        subprocess.check_call(
            [python_executable, "-m", "pip", "uninstall", "plotly", "-y"]
        )


@pytest.fixture(scope="function")
def uninstall_plotly_if_installed():
    """A fixture to uninstall Plotly if it's installed before a test and reinstall it afterwards."""
    # Check if Plotly is installed before the test
    was_installed = is_package_installed("plotly")

    # If Plotly is installed, uninstall it
    if was_installed:
        subprocess.check_call(
            [python_executable, "-m", "pip", "uninstall", "plotly", "-y"]
        )

    # Yield control back to the test
    yield

    # If Plotly was uninstalled for the test, reinstall it afterwards
    if was_installed:
        subprocess.check_call([python_executable, "-m", "pip", "install", "plotly"])

    # Reset the default renderer for tests
    plotly.io.renderers.default = None


@pytest.mark.plots
def test_initialization_with_plotly_installed(plotly_installed):
    """Test initialization when Plotly is installed."""
    assert is_package_installed("plotly")
    plotly_manager = PlotlyManager()

    import plotly.graph_objs as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    assert plotly_manager.go == go
    assert plotly_manager.pio == pio
    assert plotly_manager.make_subplots == make_subplots


@pytest.mark.plots
def test_prompt_for_plotly_installation(mocker, uninstall_plotly_if_installed):
    """Test prompt for Plotly installation when not installed."""
    assert not is_package_installed("plotly")
    mocker.patch("builtins.input", return_value="y")
    plotly_manager = PlotlyManager()

    import plotly.graph_objs as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    assert plotly_manager.go == go
    assert plotly_manager.pio == pio
    assert plotly_manager.make_subplots == make_subplots


@pytest.mark.plots
def test_cancel_installation(mocker, uninstall_plotly_if_installed):
    """Test exit if Plotly installation is canceled."""
    assert not is_package_installed("plotly")
    mocker.patch("builtins.input", return_value="n")
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        PlotlyManager().prompt_for_plotly_installation()

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1
    assert not is_package_installed("plotly")


@pytest.mark.plots
def test_post_install_setup(plotly_installed):
    """Test post-install setup."""
    plotly_manager = PlotlyManager()
    plotly_manager.post_install_setup()

    assert plotly_manager.pio.renderers.default == "browser"

    # Reset the default renderer for tests
    plotly.io.renderers.default = None


def is_package_installed(package_name):
    """Check if a package is installed without raising an exception."""
    return any(d.metadata["Name"] == package_name for d in distributions())
