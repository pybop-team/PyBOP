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
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run("pytest", "--unit", "-n", "auto")


@nox.session
def coverage(session):
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run(
        "pytest", "--unit", "--cov", "--cov-append", "--cov-report=xml", "-n", "auto"
    )
    session.run(
        "pytest",
        "--integration",
        "--cov",
        "--cov-append",
        "--cov-report=xml",
        "-n",
        "auto",
    )
    session.run("pytest", "--plots", "--cov", "--cov-append", "--cov-report=xml")


@nox.session
def integration(session):
    session.install("-e", ".[all,dev]", silent=False)
    session.install("pytest", "pytest-mock")
    session.run("pytest", "--integration", "-n", "auto")


@nox.session
def examples(session):
    session.install("-e", ".[all,dev]", silent=False)
    session.run("pytest", "--examples", "-n", "auto")


@nox.session
def notebooks(session):
    """Run the examples tests for Jupyter notebooks."""
    session.install("-e", ".[all,dev]", silent=False)
    if PYBOP_SCHEDULED:
        session.run("pip", "install", f"pybamm=={PYBAMM_VERSION}", silent=False)
    session.run(
        "pytest",
        "--notebooks",
        "--nbmake",
        "examples/",
        "-n",
        "auto",
    )


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
