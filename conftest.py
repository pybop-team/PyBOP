import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--unit", action="store_true", default=False, help="run unit tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit"):
        # --unit given in cli: do not skip unit tests
        return
    skip_unit = pytest.mark.skip(reason="need --unit option to run")
    for item in items:
        if "unit" in item.keywords:
            item.add_marker(skip_unit)