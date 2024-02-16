import os
import nox


# nox options
nox.options.reuse_existing_virtualenvs = True
nox.options.venv_backend = "virtualenv"

# Environment variables to control CI behaviour for nox sessions
PYBOP_SCHEDULED = int(os.environ.get("PYBOP_SCHEDULED", 0))
PYBAMM_VERSION = os.environ.get("PYBAMM_VERSION", None)


@nox.session
def unit(session):
    session.install("-e", ".[all]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.install("pytest", "pytest-mock", silent=False)
    session.run("pytest", "--unit")


@nox.session
def coverage(session):
    session.install("-e", ".[all]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.install("pytest", "pytest-cov", "pytest-mock", silent=False)
    session.run(
        "pytest",
        "--unit",
        "--examples",
        "--cov",
        "--cov-report=xml",
    )


@nox.session
def notebooks(session):
    """Run the examples tests for Jupyter notebooks."""
    session.install("-e", ".[all]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.install("pytest", "nbmake", silent=False)
    session.run("pytest", "--nbmake", "--examples", "examples/", external=True)


@nox.session
def docs(session):
    """
    Build the documentation and load it in a browser tab, rebuilding on changes.
    Credit: PyBaMM Team
    """
    envbindir = session.bin
    session.install("-e", ".[all,docs]", silent=False)
    session.chdir("docs")
    # Local development
    if session.interactive:
        session.run(
            "sphinx-autobuild",
            "-j",
            "auto",
            "--open-browser",
            "-qT",
            ".",
            f"{envbindir}/../tmp/html",
        )
    # Runs in CI only, treating warnings as errors
    else:
        session.run(
            "sphinx-build",
            "-j",
            "auto",
            "-b",
            "html",
            "--keep-going",
            ".",
            "_build/html",
        )
