# Contributing

If you'd like to contribute to PyBOP, please have a look at the guidelines below.

## Developer-Installation

To install PyBOP for development purposes, which includes the plotting dependencies, use the `[all]` and the `[dev]` flags as demonstrated below:

For `zsh`:

```sh
pip install -e '.[all,dev]'
```

For `bash`:
```sh
pip install -e .[all,dev]
```

## Pre-commit checks

Before you commit any code, please perform the following checks using [Nox](https://nox.thea.codes/en/stable/index.html):

- [All tests pass](#testing): `$ nox -s unit`

### Installing and using pre-commit

`PyBOP` uses a set of `pre-commit` hooks and the `pre-commit` bot to format and prettify the codebase. The hooks can be installed locally using -

```bash
nox -s pre-commit
```

alternatively, without nox:

```bash
pip install pre-commit
pre-commit install
```

This would run the checks every time a commit is created locally. The checks will only run on the files modified by that commit, but the checks can be triggered for all the files using,

```bash
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

## Workflow

We use [Git](https://en.wikipedia.org/wiki/Git) and [GitHub](https://en.wikipedia.org/wiki/GitHub) to coordinate our work. When making any kind of update, we try to follow the procedure below.

### A. Before you begin

1. Create an [issue](https://guides.github.com/features/issues/) where new proposals can be discussed before any coding is done.
2. Create a [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/) of this repo (ideally on your own [fork](https://help.github.com/articles/fork-a-repo/)), where all changes will be made
3. Download the source code onto your local system, by [cloning](https://help.github.com/articles/cloning-a-repository/) the repository (or your fork of the repository).
4. [Install](#developer-installation) PyBOP with the developer options.
5. [Test](#testing) if your installation worked: `nox -s unit` or `$ pytest --unit -v`.

You now have everything you need to start making changes!

### B. Writing your code

6. PyBOP is developed in [Python](https://en.wikipedia.org/wiki/Python_(programming_language)), and makes heavy use of [NumPy](https://en.wikipedia.org/wiki/NumPy) (see also [NumPy for MatLab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html) and [Python for R users](https://rebeccabarter.com/blog/2023-09-11-from_r_to_python)).
7. Make sure to follow our [coding style guidelines](#coding-style-guidelines).
8. Commit your changes to your branch with [useful, descriptive commit messages](https://chris.beams.io/posts/git-commit/): Remember these are publicly visible and should still make sense a few months ahead in time. While developing, you can keep using the GitHub issue you're working on as a place for discussion. Refer to your commits when discussing specific lines of code. This is achieved by referencing the SHA-hash in the comment. An example of this looks like: `the commit 3e5c1e6 solved the issue...`
9. If you want to add a dependency on another library, or re-use code you found somewhere else, have a look at [these guidelines](#dependencies-and-reusing-code).

### C. Merging your changes with PyBOP

10. [Test your code!](#testing)
12. If you added a major new feature, perhaps it should be showcased in an [example notebook](https://github.com/pybop-team/PyBOP/tree/develop/examples/notebooks).
13. If you've added new functionality, please add additional tests to ensure ample code coverage in PyBOP.
13. When you feel your code is finished, or at least warrants serious discussion, create a [pull request](https://help.github.com/articles/about-pull-requests/) (PR) on [PyBOP's GitHub page](https://github.com/pybop-team/PyBOP).
14. Once a PR has been created, it will be reviewed by any member of the community. Changes might be suggested which you can make by simply adding new commits to the branch. When everything's finished, someone with the right GitHub permissions will merge your changes into PyBOP main repository.

Finally, if you really, really, _really_ love developing PyBOP, have a look at the current [project infrastructure](#infrastructure).

## Coding style guidelines

PyBOP follows the [PEP8 recommendations](https://www.python.org/dev/peps/pep-0008/) for coding style. These are very common guidelines, and community tools have been developed to check how well projects implement them.

### Ruff

We use [ruff](https://github.com/charliermarsh/ruff) to check our PEP8 adherence. To try this on your system, navigate to the PyBOP directory in a console and type

```bash
python -m pip install pre-commit
pre-commit run ruff
```

ruff is configured inside the file `pre-commit-config.yaml`, allowing us to ignore some errors. If you think this should be added or removed, please submit an [issue](https://guides.github.com/features/issues/).

When you commit your changes they will be checked against ruff automatically (see [Pre-commit checks](#pre-commit-checks)).

### Naming

Naming is hard. In general, we aim for descriptive class, method, and argument names. Avoid abbreviations when possible without making names overly long, so `mean` is better than `mu`, but a class name like `MyClass` is fine.

Class names are CamelCase, and start with an upper case letter, for example `MyOtherClass`. Method and variable names are lower-case, and use underscores for word separation, for example, `x` or `iteration_count`.

## Dependencies and reusing code

While it's a bad idea for developers to "reinvent the wheel", it's important for users to get a _reasonably sized download and an easy install_. In addition, external libraries can sometimes cease to be supported, and when they contain bugs it might take a while before fixes become available as automatic downloads to PyBOP users.
For these reasons, all dependencies in PyBOP should be thought about carefully and discussed on GitHub.

Direct inclusion of code from other packages is possible, as long as their license permits it and is compatible with ours, but again should be considered carefully and discussed in the group. Snippets from blogs and stackoverflow can often be included but must include attribution to the original by commenting with a link in the source code.

### Separating dependencies

On the other hand... We _do_ want to compare several tools, to generate documentation, and speed up development. For this reason, the dependency structure is split into 4 parts:

1. Core PyBOP: A minimal set, including things like NumPy, SciPy, etc. All infrastructure should run against this set of dependencies, as well as any numerical methods we implement ourselves.
2. Extras: Other inference packages and their dependencies. Methods we don't want to implement ourselves, but do want to provide an interface to can have their dependencies added here.
3. Documentation generating code: Everything you need to generate and work on the docs. This is managed by the `[docs]` set of extras.
4. Development code: Everything you need to do PyBOP development (so all of the above packages, plus ruff and other testing tools). This is managed by the `[dev]` set of extras.

Only 'core pybop' is installed by default. The others have to be specified explicitly when running the installation command.

### Plotly

We use Plotly in PyBOP, but with two caveats:

First, Plotly should only be used in plotting methods, and these should _never_ be called by other PyBOP methods. So users who don't like Plotly will not be forced to use it in any way. Use in notebooks is OK and encouraged.

Second, Plotly should never be imported at the module level, but always inside methods. For example:

```python
def plot_great_things(self, x, y, z):
   go = pybop.PlotlyManager().go
    ...
```

This allows people to (1) use PyBOP without ever importing Plotly and (2) configure Plotly's settings in their scripts, which _must_ be done before e.g. `graph_objects` is first imported.

### Building documentation

We use [Sphinx](http://www.sphinx-doc.org/en/stable/) to build our documentation. A [Nox](https://nox.thea.codes/en/stable/index.html) session has been created to reduce the overhead when building the documentation locally. To run this session, type

```bash
nox -s docs
```

This will build the docs using sphinx-autobuild and render them in your browser. Likewise, to test the docs build, the following nox session is available:

```bash
nox -s doctests
```

## Testing

All code requires testing. We use the [pytest](https://docs.pytest.org/en/) package for our tests. (These tests typically just check that the code runs without error, and so, are more _debugging_ than _testing_ in a strict sense. Nevertheless, they are very useful to have!)

If you have nox installed, to run unit tests, type

```bash
nox -s unit
```

Alternatively, to run tests standalone with pytest, run,

```bash
pytest --unit -v
```

To run individual test files, you can use

```bash
pytest tests/unit/path/to/test --unit -v
```

And for individual tests,

```bash
pytest tests/unit/path/to/test.py::TestClass:test_name --unit -v
```
where `--unit` is a flag to run only unit tests and `-v` is a flag to display verbose output. Furthermore, to run all the standard tests, type

```bash
nox -s tests
```
Additionally, to run the standard and docs tests, type

```bash
nox -s quick
```

### Writing tests

Every new feature should have its own test. To create ones, have a look at the `test` directory and see if there's a test for a similar method. Copy-pasting is a good way to start.

Next, add some simple (and speedy!) tests of your main features. If these run without exceptions that's a good start! Next, check the output of your methods using any of these [functions](https://docs.pytest.org/en/7.4.x/reference/reference.html#functions).

### Debugging

Often, the code you write won't pass the tests straight away, at which stage it will become necessary to debug.
The key to successful debugging is to isolate the problem by finding the smallest possible example that causes the bug.
In practice, there are a few tricks to help you do this, which we give below.
Once you've isolated the issue, it's a good idea to add a unit test that replicates this issue, so that you can easily check whether it's been fixed, and make sure that it's easily picked up if it crops up again.
This also means that, if you can't fix the bug yourself, it will be much easier to ask for help (by opening a [bug-report issue](https://github.com/pybop-team/PyBOP/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml&title=%5BBug%5D%3A+)).

1. Run individual test scripts instead of the whole test suite:

   ```bash
   pytest tests/unit/path/to/test --unit -v
   ```

   You can also run an individual test from a particular script, e.g.

   ```bash
   pytest tests/unit/path/to/test.py::TestClass:test_name --unit -v
   ```
   where `--unit` is a flag to run only unit tests and `-v` is a flag to display verbose output.

2. Set break-points, either in your IDE or using the Python debugging module. To use the latter, add the following line where you want to set the break point

   ```python
   import ipdb

   ipdb.set_trace()
   ```

   This will start the [Python interactive debugger](https://gist.github.com/mono0926/6326015). If you want to be able to use magic commands from `ipython`, such as `%timeit`, then set

   ```python
   from IPython import embed

   embed()
   import ipdb

   ipdb.set_trace()
   ```

   at the break point instead.
   Figuring out where to start the debugger is the real challenge. Some good ways to set debugging break points are:

   1. Try-except blocks. Suppose the line `do_something_complicated()` is raising a `ValueError`. Then you can put a try-except block around that line as:

      ```python
      try:
          do_something_complicated()
      except ValueError:
          import ipdb

          ipdb.set_trace()
      ```

      This will start the debugger at the point where the `ValueError` was raised, and allow you to investigate further. Sometimes, it is more informative to put the try-except block further up the call stack than exactly where the error is raised.
   2. Warnings. If functions are raising warnings instead of errors, it can be hard to pinpoint where this is coming from. Here, you can use the `warnings` module to convert warnings to errors:

      ```python
      import warnings

      warnings.simplefilter("error")
      ```

      Then you can use a try-except block, as in a., but with, for example, `RuntimeWarning` instead of `ValueError`.

3. To isolate whether a bug is in a model, its Jacobian or its simplified version, you can set the `use_jacobian` and/or `use_simplify` attributes of the model to `False` (they are both `True` by default for most models).
4. If a model isn't giving the answer you expect, you can try comparing it to other models. For example, you can investigate parameter limits in which two models should give the same answer by setting some parameters to be small or zero. The `StandardOutputComparison` class can be used to compare some standard outputs from battery models.
5. To get more information about what is going on under the hood, and hence understand what is causing the bug, you can set the [logging](https://realpython.com/python-logging/) level to `DEBUG` by adding the following line to your test or script:

   ```python3
   pybop.set_logging_level("DEBUG")
   ```

### Profiling

Sometimes, a bit of code will take much longer than you expect to run. In this case, you can set

```python
from IPython import embed

embed()
import ipdb

ipdb.set_trace()
```

as above, and then use some of the profiling tools. In order of increasing detail:

1. Simple timer. In ipython, the command

   ```
   %time command_to_time()
   ```

   tells you how long the line `command_to_time()` takes. You can use `%timeit` instead to run the command several times and obtain more accurate timings.
2. Simple profiler. Using `%prun` instead of `%time` will give a brief profiling report 3. Detailed profiler. You can install the detailed profiler `snakeviz` through pip:

   ```bash
   pip install snakeviz
   ```

   and then, in ipython, run

   ```
   %load_ext snakeviz
   %snakeviz command_to_time()
   ```

   This will open a window in your browser with detailed profiling information.

## Infrastructure

### Installation via `pip`

Installation of PyBOP and its dependencies is handled via [`pip`](https://pip.pypa.io/) through the [setuptools](http://setuptools.readthedocs.io/) build-backend.

Configuration files:

```
pyproject.toml
```

### Continuous Integration using GitHub actions

Each change pushed to the PyBOP GitHub repository will trigger the test and benchmark suites to be run, using [GitHub actions](https://github.com/features/actions).

Tests are run for different operating systems, and for all Python versions officially supported by PyBOP. If you opened a Pull Request, feedback is directly available on the corresponding page. If all tests pass, a green tick will be displayed next to the corresponding test run. If one or more test(s) fail, a red cross will be displayed instead.

Similarly, the benchmark suite is automatically run for the most recently pushed commit. Benchmark results are compared to the results available for the latest commit on the `develop` branch. Should any significant performance regression be found, a red cross will be displayed next to the benchmark run.

In all cases, more details can be obtained by clicking on a specific run.

Configuration files for various GitHub actions workflow can be found in `.github/workflows`.

### Codecov

Code coverage (how much of our code is seen by the (Linux) unit tests) is tested using [Codecov](https://docs.codecov.io/), a report is visible on https://codecov.io/gh/pybop-team/PyBOP.


### GitHub

GitHub does some magic with particular filenames. In particular:

- The first page people see when they go to [our GitHub page](https://github.com/pybop-team/PyBOP) displays the contents of [README.md](https://github.com/pybop-team/PyBOP/blob/develop/README.md), which is written in the [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) format. Some guidelines can be found [here](https://help.github.com/articles/about-readmes/).
- The license for using PyBOP is stored in [LICENSE](https://github.com/pybop-team/PyBOP/blob/develop/LICENSE), and [automatically](https://help.github.com/articles/adding-a-license-to-a-repository/) linked to by GitHub.
- This file, [CONTRIBUTING.md](https://github.com/pybop-team/PyBOP/blob/develop/CONTRIBUTING.md) is recognised as the contribution guidelines and a link is [automatically](https://github.com/blog/1184-contributing-guidelines) displayed when new issues or pull requests are created.

## Acknowledgements

This CONTRIBUTING.md file, along with large sections of the code infrastructure, was copied from the excellent [Pints repo](https://github.com/pints-team/pints), and [PyBaMM repo](https://github.com/pybamm-team/PyBaMM)
