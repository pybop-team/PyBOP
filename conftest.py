import pytest
import matplotlib
import plotly

plotly.io.renderers.default = None
matplotlib.use("Template")


def pytest_addoption(parser):
    parser.addoption(
        "--unit", action="store_true", default=False, help="run unit tests"
    )
    parser.addoption(
        "--examples", action="store_true", default=False, help="run examples tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "examples: mark test as an example")


def pytest_collection_modifyitems(config, items):
    def skip_marker(marker_name, reason):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker_name in item.keywords:
                item.add_marker(skip)

    if config.getoption("--unit"):
        skip_marker("examples", "need --examples option to run")
        return

    if config.getoption("--examples"):
        skip_marker("unit", "need --unit option to run")
        return

    skip_marker("unit", "need --unit option to run")
