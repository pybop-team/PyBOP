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
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
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
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "examples: mark test as an example")


def pytest_collection_modifyitems(config, items):
    unit = config.getoption("--unit")
    integration = config.getoption("--integration")
    examples = config.getoption("--examples")

    if not unit and not examples and not integration:
        skip_all = pytest.mark.skip(
            reason="need --unit or --examples or --integration option to run"
        )
        for item in items:
            item.add_marker(skip_all)

    elif unit and not examples and not integration:
        skip_examples_integration = pytest.mark.skip(
            reason="need --examples option to run examples tests, or --integration option to run integration tests"
        )
        for item in items:
            if "examples" in item.keywords:
                item.add_marker(skip_examples_integration)
            if "integration" in item.keywords:
                item.add_marker(skip_examples_integration)

    elif examples and not unit and not integration:
        skip_unit_integration = pytest.mark.skip(
            reason="need --unit option to run unit tests or --integration option to run integration tests"
        )
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit_integration)
            if "integration" in item.keywords:
                item.add_marker(skip_unit_integration)

    elif integration and not unit and not examples:
        skip_unit_examples = pytest.mark.skip(
            reason="need --unit option to run unit tests or --examples option to run examples tests"
        )
        for item in items:
            if "unit" in item.keywords:
                item.add_marker(skip_unit_examples)
            if "examples" in item.keywords:
                item.add_marker(skip_unit_examples)
