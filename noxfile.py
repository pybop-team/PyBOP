import nox

# nox options
nox.options.reuse_existing_virtualenvs = True

# @nox.session
# def lint(session):
#     session.install('flake8')
#     session.run('flake8', 'example.py')


@nox.session
def unit(session):
    session.run_always("pip", "install", "-e", ".")
    session.install("pytest")
    session.run("pytest", "--unit")


@nox.session
def coverage(session):
    session.run_always("pip", "install", "-e", ".")
    session.install("pytest-cov")
    session.run("pytest", "--cov", "--cov-report=xml")
