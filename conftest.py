import time

import matplotlib
import plotly
import pytest

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
    parser.addoption(
        "--plots", action="store_true", default=False, help="run plot tests"
    )
    parser.addoption(
        "--notebooks", action="store_true", default=False, help="run notebook tests"
    )
    parser.addoption("--docs", action="store_true", default=False, help="run doc tests")


def pytest_sessionstart(session):
    start_time = time.time()
    session.config.cache.set("session_start_time", start_time)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add additional section to terminal summary reporting."""
    start_time = config.cache.get("session_start_time", None)
    if start_time is None:
        print("Warning: Session start time not found in cache.")
        return

    # Collect the durations of all tests from different outcomes
    total_time = 0
    num_tests = 0

    # Loop through all test outcomes (including skipped)
    for _outcome, reports in terminalreporter.stats.items():
        for report in reports:
            if hasattr(report, "duration") and report.duration is not None:
                total_time += report.duration
                num_tests += 1

    print(f"\nTotal number of tests completed: {num_tests}")
    print(f"Total summed time taken: {total_time:.2f} seconds")
    print(f"Total wall clock time: {(time.time() - start_time):.2f} seconds")


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "examples: mark test as an example")
    config.addinivalue_line("markers", "plots: mark test as a plot test")
    config.addinivalue_line("markers", "notebook: mark test as a notebook test")
    config.addinivalue_line("markers", "docs: mark test as a doc test")


def pytest_collection_modifyitems(config, items):
    options = {
        "unit": "unit",
        "examples": "examples",
        "integration": "integration",
        "plots": "plots",
        "notebooks": "notebooks",
        "docs": "docs",
    }
    selected_markers = [
        marker for option, marker in options.items() if config.getoption(option)
    ]

    if (
        "notebooks" in selected_markers
    ):  # Notebooks are meant to be run as an individual session
        return

    # If no options were passed, skip all tests
    if not selected_markers:
        skip_all = pytest.mark.skip(
            reason="Need at least one of --unit, --examples, --integration, --docs, or --plots option to run"
        )
        for item in items:
            item.add_marker(skip_all)
        return

    # Skip tests that don't match any of the selected markers
    for item in items:
        item_markers = {
            mark.name for mark in item.iter_markers()
        }  # Gather markers of the test item
        if not item_markers.intersection(
            selected_markers
        ):  # Skip if there's no intersection with selected markers
            skip_this = pytest.mark.skip(
                reason=f"Test does not match the selected options: {', '.join(selected_markers)}"
            )
            item.add_marker(skip_this)
