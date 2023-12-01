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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add additional section to terminal summary reporting."""
    total_time = sum([x.duration for x in terminalreporter.stats.get("passed", [])])
    num_tests = len(terminalreporter.stats.get("passed", []))
    print(f"\nTotal number of tests completed: {num_tests}")
    print(f"Total time taken: {total_time:.2f} seconds")


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "examples: mark test as an example")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit") and not config.getoption("--examples"):
        skip_examples = pytest.mark.skip(
            reason="need --examples option to run examples tests"
        )
        for item in items:
            if "examples" in item.keywords:
                item.add_marker(skip_examples)

    if config.getoption("--examples") and not config.getoption("--unit"):
        skip_unit = pytest.mark.skip(reason="need --unit option to run unit tests")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)
