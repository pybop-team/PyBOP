import nox

# nox options
nox.options.reuse_existing_virtualenvs = True


@nox.session
def unit(session):
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "pytest-mock")
    session.run("pytest", "--unit")


@nox.session
def coverage(session):
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "pytest-cov", "pytest-mock")
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
    session.run_always("pip", "install", "-e", ".[all]")
    session.install("pytest", "nbmake")
    session.run("pytest", "--nbmake", "examples/", external=True)


@nox.session
def docs(session):
    """
    Build the documentation and load it in a browser tab, rebuilding on changes.
    Credit: PyBaMM Team
    """
    envbindir = session.bin
    session.install("-e", ".[all,docs]")
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
