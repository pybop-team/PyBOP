import shutil
import subprocess
import sys
from pathlib import Path

import pytest


class TestDocs:
    """A class to test the PyBOP documentation."""

    pytestmark = pytest.mark.docs

    def test_docs(self):
        """
        Check if the documentation can be built and run any doctests (currently not used).

        Credit: PyBaMM Team
        """
        print("Checking if docs can be built.")
        docs_path = Path("docs")
        build_path = docs_path / "_build" / "html"

        try:
            subprocess.run(
                [
                    "sphinx-build",
                    "-j",
                    "auto",
                    "-b",
                    "html",
                    str(docs_path),
                    str(build_path),
                    "--keep-going",
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"FAILED with exit code {e.returncode}")
            print(f"stdout: {e.stdout.decode()}")
            print(f"stderr: {e.stderr.decode()}")
            sys.exit(e.returncode)
        finally:
            # Regardless of whether the doctests pass or fail, attempt to remove the built files.
            print("Deleting built files.")
            try:
                shutil.rmtree(build_path)
            except Exception as e:
                print(f"Error deleting built files: {e}")
