import pytest
import matplotlib
import plotly

plotly.io.renderers.default = None
matplotlib.use("Template")

# Ignore the pybamm/ folder and any subdirectories from test discovery
collect_ignore = ["pybamm/"]


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
    unit_option = config.getoption("--unit")
    examples_option = config.getoption("--examples")

    if not unit_option and not examples_option:
        skip_all = pytest.mark.skip(reason="need --unit or --examples option to run")
        for item in items:
            item.add_marker(skip_all)

    elif unit_option and not examples_option:
        skip_examples = pytest.mark.skip(
            reason="need --examples option to run examples tests"
        )
        for item in items:
            if "examples" in item.keywords:
                item.add_marker(skip_examples)

    if examples_option and not unit_option:
        skip_unit = pytest.mark.skip(reason="need --unit option to run unit tests")
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit)
