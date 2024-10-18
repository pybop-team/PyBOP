import subprocess
import sys
import webbrowser


class PlotlyManager:
    """
    Manages the installation and configuration of Plotly for generating visualizations.

    This class ensures that Plotly is installed and properly configured to display
    plots in a web browser.

    Upon instantiation, it checks for Plotly's presence, installs it if missing,
    and configures the default renderer and browser settings.

    Attributes
    ----------
    go : module
        The Plotly graph_objects module for creating figures.
    pio : module
        The Plotly input/output module for configuring the renderer.
    make_subplots : function
        The function from Plotly for creating subplot figures.

    Examples
    --------
    >>> plotly_manager = PlotlyManager()
    """

    def __init__(self):
        """
        Initialize the PlotlyManager, ensuring Plotly is installed and configured.
        """
        self.go = None
        self.pio = None
        self.make_subplots = None
        self.ensure_plotly_installed()
        self.check_renderer_settings()
        self.check_browser_availability()

    def ensure_plotly_installed(self):
        """
        Check if Plotly is installed and import necessary modules; prompt for installation if missing.
        """
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
        """
        Prompt the user for Plotly installation and install it upon agreement.
        """
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

    @staticmethod
    def install_plotly():
        """
        Install the Plotly package using pip. Exit if installation fails.
        """
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        except subprocess.CalledProcessError as e:
            print(f"Error installing plotly: {e}")
            sys.exit(1)  # Exit if installation fails

    def post_install_setup(self):
        """
        Import Plotly modules and set the default renderer after installation.
        """
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
        """
        Check and provide information on setting the Plotly renderer if it's not already set.
        """
        if self.pio and self.pio.renderers.default == "":
            print(
                "The Plotly renderer is an empty string. To set the renderer, use:\n"
                "    pio.renderers\n"
                '    pio.renderers.default = "browser"\n'
                "For more information, see: https://plotly.com/python/renderers/#setting-the-default-renderer"
            )

    def check_browser_availability(self):
        """
        Confirm a web browser is available for Plotly's 'browser' renderer; provide guidance if not.
        """
        if self.pio and self.pio.renderers.default == "browser":
            try:
                webbrowser.get()
            except webbrowser.Error as e:
                raise Exception(
                    "\n **Browser Not Found** \nFor Windows users, in order to view figures in the browser using Plotly, "
                    "you need to set the environment variable BROWSER equal to the "
                    "path to your chosen browser. To do this, please enter a command like "
                    "the following to add this to your virtual environment activation file:\n\n"
                    "echo 'export BROWSER=\"/mnt/c/Program Files/Mozilla Firefox/firefox.exe\"' >> your-env/bin/activate"
                    "\n\nThen reactivate your virtual environment. Alternatively, you can use a "
                    "different Plotly renderer. For more information see: "
                    "https://plotly.com/python/renderers/#setting-the-default-renderer"
                ) from e
