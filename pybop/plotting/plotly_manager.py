import subprocess
import webbrowser
import sys


class PlotlyManager:
    """
    Manages the installation and configuration of Plotly for generating visualisations.

    This class checks if Plotly is installed and, if not, prompts the user to install it.
    It also ensures that the Plotly renderer and browser settings are properly configured
    to display plots.

    Methods:
        ``ensure_plotly_installed``: Verifies if Plotly is installed and installs it if necessary.
        ``prompt_for_plotly_installation``: Prompts the user for permission to install Plotly.
        ``install_plotly_package``: Installs the Plotly package using pip.
        ``post_install_setup``: Sets up Plotly default renderer after installation.
        ``check_renderer_settings``: Verifies that the Plotly renderer is correctly set.
        ``check_browser_availability``: Checks if a web browser is available for rendering plots.

    Usage:
        Instantiate the PlotlyManager class to automatically ensure Plotly is installed
        and configured correctly when creating an instance.
        Example:
            plotly_manager = PlotlyManager()
    """

    def __init__(self):
        self.go = None
        self.pio = None
        self.make_subplots = None
        self.ensure_plotly_installed()
        self.check_renderer_settings()
        self.check_browser_availability()

    def ensure_plotly_installed(self):
        """Verifies if Plotly is installed, importing necessary modules and prompting for installation if missing."""
        try:
            import plotly.graph_objs as go
            import plotly.io as pio
            from plotly.subplots import make_subplots

            self.go = go
            self.pio = pio
            self.make_subplots = make_subplots
        except ImportError:
            self.prompt_for_plotly_installation()

    def prompt_for_plotly_installation(self):
        """Prompts the user for permission to install Plotly and proceeds with installation if consented."""
        user_input = (
            input(
                "Plotly is not installed. To proceed, we need to install plotly. (Y/n)? "
            )
            .strip()
            .lower()
        )
        if user_input == "y":
            self.install_plotly()
            self.post_install_setup()
        else:
            print("Installation cancelled by user.")
            sys.exit(1)  # Exit if user cancels installation

    def install_plotly(self):
        """Attempts to install the Plotly package using pip and exits if installation fails."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        except subprocess.CalledProcessError as e:
            print(f"Error installing plotly: {e}")
            sys.exit(1)  # Exit if installation fails

    def post_install_setup(self):
        """After successful installation, imports Plotly and sets the default renderer if necessary."""
        import plotly.graph_objs as go
        import plotly.io as pio
        from plotly.subplots import make_subplots

        self.go = go
        self.pio = pio
        self.make_subplots = make_subplots
        if pio.renderers.default == "":
            pio.renderers.default = "browser"
            print(
                'Set default renderer to "browser" as it was empty after installation.'
            )

    def check_renderer_settings(self):
        """Checks if the Plotly renderer is set and provides information on how to set it if empty."""
        if self.pio and self.pio.renderers.default == "":
            print(
                "The Plotly renderer is an empty string. To set the renderer, use:\n"
                "    pio.renderers\n"
                '    pio.renderers.default = "browser"\n'
                "For more information, see: https://plotly.com/python/renderers/#setting-the-default-renderer"
            )

    def check_browser_availability(self):
        """Ensures a web browser is available for rendering plots with the 'browser' renderer and provides guidance if not."""
        if self.pio and self.pio.renderers.default == "browser":
            try:
                webbrowser.get()
            except webbrowser.Error:
                raise Exception(
                    "\n **Browser Not Found** \nFor Windows users, in order to view figures in the browser using Plotly, "
                    "you need to set the environment variable BROWSER equal to the "
                    "path to your chosen browser. To do this, please enter a command like "
                    "the following to add this to your virtual environment activation file:\n\n"
                    "echo 'export BROWSER=\"/mnt/c/Program Files/Mozilla Firefox/firefox.exe\"' >> your-env/bin/activate"
                    "\n\nThen reactivate your virtual environment. Alternatively, you can use a "
                    "different Plotly renderer. For more information see: "
                    "https://plotly.com/python/renderers/#setting-the-default-renderer"
                )
