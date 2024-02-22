import sys
import pytest
import shutil
import subprocess


class TestDocs:
    """
    A class to test the pybop documentation
    """

    @pytest.mark.docs
    def test_docs(self):
        """
        Checks if the documentation can be built, runs any doctests (currently not
        used).

        Credit: PyBaMM Team
        """
        print("Checking if docs can be built.")
        try:
            subprocess.run(
                [
                    "sphinx-build",
                    "-j",
                    "auto",
                    "-b",
                    "html",
                    "--keep-going",
                    ".",
                    "_build/html",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"FAILED with exit code {e.returncode}")
            sys.exit(e.returncode)
        finally:
            # Regardless of whether the doctests pass or fail, attempt to remove the built files.
            print("Deleting built files.")
            try:
                shutil.rmtree("docs/_build/html/.doctrees/")
            except Exception as e:
                print(f"Error deleting built files: {e}")
