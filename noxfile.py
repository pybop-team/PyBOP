import nox

# nox options
nox.options.reuse_existing_virtualenvs = True

# @nox.session
# def lint(session):
#     session.install('flake8')
#     session.run('flake8', 'example.py')


@nox.session
def tests(session):
    session.run_always('pip', 'install', '-e', '.')
    session.install('pytest')
    session.run('pytest')