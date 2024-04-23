# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

## Issue reference
Fixes # (issue-number)

## Review
Before you mark your PR as ready for review, please ensure that you've considered the following:
- Updated the [CHANGELOG.md](https://github.com/pybop-team/PyBOP/blob/develop/CHANGELOG.md) in reverse chronological order (newest at the top) with a concise description of the changes, including the PR number.
- Noted any breaking changes, including details on how it might impact existing functionality.

## Type of change
- [ ] New Feature: A non-breaking change that adds new functionality.
- [ ] Optimization: A code change that improves performance.
- [ ] Examples: A change to existing or additional examples.
- [ ] Bug Fix: A non-breaking change that addresses an issue.
- [ ] Documentation: Updates to documentation or new documentation for new features.
- [ ] Refactoring: Non-functional changes that improve the codebase.
- [ ] Style: Non-functional changes related to code style (formatting, naming, etc).
- [ ] Testing: Additional tests to improve coverage or confirm functionality.
- [ ] Other: (Insert description of change)

# Key checklist:

- [ ] No style issues: `$ pre-commit run` (or `$ nox -s pre-commit`) (see [CONTRIBUTING.md](https://github.com/pybop-team/PyBOP/blob/develop/CONTRIBUTING.md#installing-and-using-pre-commit) for how to set this up to run automatically when committing locally, in just two lines of code)
- [ ] All unit tests pass: `$ nox -s tests`
- [ ] The documentation builds: `$ nox -s doctest`

You can run integration tests, unit tests, and doctests together at once, using `$ nox -s quick`.

## Further checks:
- [ ] Code is well-commented, especially in complex or unclear areas.
- [ ] Added tests that prove my fix is effective or that my feature works.
- [ ] Checked that coverage remains or improves, and added tests if necessary to maintain or increase coverage.

Thank you for contributing to our project! Your efforts help us to deliver great software.
