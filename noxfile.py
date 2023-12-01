import nox

# nox options
nox.options.reuse_existing_virtualenvs = True


@nox.session
def unit(session):
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "pytest-mock")
    session.run("pytest", "--unit", "-v", "--showlocals")


@nox.session
def coverage(session):
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "pytest-cov", "pytest-mock")
    session.run(
        "pytest",
        "--unit",
        "--examples",
        "-v",
        "--cov",
        "--cov-report=xml",
        "--showlocals",
    )


@nox.session
def notebooks(session):
    """Run the examples tests for Jupyter notebooks."""
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "nbmake")
    session.run("pytest", "--nbmake", "examples/", external=True)
