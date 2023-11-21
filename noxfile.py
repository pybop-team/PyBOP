import nox

# nox options
nox.options.reuse_existing_virtualenvs = True


@nox.session
def unit(session):
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest")
    session.run("pytest", "--unit", "-v")


@nox.session
def coverage(session):
    session.run_always("pip", "install", "-e", ".")
    session.install("pytest-cov")
    session.run("pytest", "--unit", "-v", "--cov", "--cov-report=xml")


@nox.session
def notebooks(session):
    """Run the examples tests for Jupyter notebooks."""
    session.run_always("pip", "install", "-e", ".")
    session.install("pytest", "nbmake")
    session.run("pytest", "--nbmake", "examples/", external=True)
