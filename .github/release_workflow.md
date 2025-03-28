# Release Workflow

This document outlines the release workflow for publishing to PyPI and TestPyPI using GitHub Actions.

## Creating a New Release

To create a new release, follow these steps:

1. **Prepare the Release:**
   -  Create a new branch for the release (i.e. `24.XX`) from `develop`.
   -  Increment the following:
         -  The version number in the `pyproject.toml` and `CITATION.cff` files following CalVer versioning.
         -  The `CHANGELOG.md` version with the changes for the new version.
         -  Add a new entry for the documentation site version switcher located at `docs/_static/switcher.json`.
   -  Open a PR to the `main` branch. Once the PR is merged, proceed to the next step.

2. **Create a GitHub Release:**
   -  Go to the "Releases" section on GitHub.
   -  Click "Draft a new release".
   -  Select the `main` branch as the release target.
   -  Create a release tag:
      - For a full release, use the following CalVer format: `vYY.M` (i.e. `v24.3`).
      - For a release candidate, use: `vYY.Mrc.X` (i.e. `v24.3rc.1`).
      - Note: GitHub provides the option to create this tag on publish if you start typing a new tag
   -  Fill in the release title and description. Add any major changes and link to the `CHANGELOG.md` for a list of total changes.
   -  If it's a pre-release (release candidate), check the "This is a pre-release" checkbox.
   -  Click "Publish release" to create the release.

3. **Monitor the Workflow:**
   -  Go to the "Actions" tab of the repository and monitor the workflow's progress.
   -  The workflow will build the distribution packages and then publish them to PyPI or TestPyPI, depending on whether the release is a full release or a pre-release.

4. **Verify the Release:**
   -  Check PyPI or TestPyPI to ensure that your package is available and has been updated to the new version.
   -  Test installing the package using `pip` to ensure everything works as expected (run an example as confirmation).

5. **Merge Main into Develop**
   -  Finally, open a PR from `main` to `develop` to synchronise the release changes. This ensures `develop` is always ahead of `main`, reducing the work required for future releases.
