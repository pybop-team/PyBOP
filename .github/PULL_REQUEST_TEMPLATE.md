# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

Fixes # (issue)

## Type of change

Please add a line in the relevant section of [CHANGELOG.md](https://github.com/pybop-team/PyBOP/blob/develop/CHANGELOG.md) to document the change (include PR #).

# Important checks:

Please confirm the following before marking the PR as ready for review:
- No style issues: `$ pre-commit run` or `$ nox -s pre-commit` (see [CONTRIBUTING.md](https://github.com/pybop-team/PyBOP/blob/develop/CONTRIBUTING.md#installing-and-using-pre-commit) for how to set this up to run automatically when committing locally, in just two lines of code)
- All tests pass: `nox -s tests`
- The documentation builds: `nox -s doctest`
- Code is commented for hard-to-understand areas
- Tests added that prove fix is effective or that feature works
