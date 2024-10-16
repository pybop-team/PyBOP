import numpy as np

from pybop import PlotlyManager


class PosteriorSummary:
    def __init__(self, chains: np.ndarray, significant_digits: int = 4):
        """
        Initialize with chains of posterior samples.

        Parameters:
        chains (np.ndarray): List where each element is a NumPy array representing
                                     a chain of posterior samples for a parameter.
        significant_digits (int): Number of significant digits to display for summary statistics.
        """
        self.chains = chains
        self.all_samples = np.concatenate(chains, axis=0)
        self.num_parameters = self.chains.shape[2]
        self.sig_digits = significant_digits
        self.get_summary_statistics()
        self.go = PlotlyManager().go

    def signif(self, x, p: int):
        """
        Rounds array `x` to `p` significant digits.
        """
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    def _calculate_statistics(self, fun, attr_name, *args, **kwargs):
        """
        Calculate statistics from callable `fun`.
        """
        stat = fun(self.all_samples, *args, **kwargs)
        setattr(self, attr_name, stat)
        return self.signif(stat, self.sig_digits)

    def get_summary_statistics(self):
        """
        Calculate summary statistics for the posterior samples.

        Returns:
        dict: Summary statistics including mean, median, standard deviation, and 95% credible interval.
        """
        summary_funs = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
            "min": np.min,
            "std": np.std,
            "ci_lower": lambda x, axis: np.percentile(x, 2.5, axis=axis),
            "ci_upper": lambda x, axis: np.percentile(x, 97.5, axis=axis),
        }

        return {
            key: self._calculate_statistics(func, key, axis=0)
            for key, func in summary_funs.items()
        }

    def plot_trace(self):
        """
        Plot trace plots for the posterior samples.
        """

        for i in range(self.num_parameters):
            fig = self.go.Figure()

            for j, chain in enumerate(self.chains):
                fig.add_trace(
                    self.go.Scatter(y=chain[:, i], mode="lines", name=f"Chain {j}")
                )

            fig.update_layout(
                title=f"Parameter {i} Trace Plot",
                xaxis_title="Sample Index",
                yaxis_title="Value",
            )
            fig.show()

    def plot_chains(self):
        """
        Plot posterior distributions for each chain.
        """
        fig = self.go.Figure()

        for i, chain in enumerate(self.chains):
            for j in range(chain.shape[1]):
                fig.add_trace(
                    self.go.Histogram(
                        x=chain[:, j],
                        name=f"Chain {i} - Parameter {j}",
                        opacity=0.75,
                    )
                )

                fig.add_shape(
                    type="line",
                    x0=self.mean[j],
                    y0=0,
                    x1=self.mean[j],
                    y1=self.max[j],
                    name=f"Mean - Parameter {j}",
                    line=dict(color="Black", width=1.5, dash="dash"),
                )

        fig.update_layout(
            barmode="overlay",
            title="Posterior Distribution",
            xaxis_title="Value",
            yaxis_title="Density",
        )
        fig.show()

    def plot_posterior(self):
        """
        Plot the summed posterior distribution across chains.
        """
        fig = self.go.Figure()

        for j in range(self.all_samples.shape[1]):
            histogram = self.go.Histogram(
                x=self.all_samples[:, j],
                name=f"Parameter {j}",
                opacity=0.75,
            )
            fig.add_trace(histogram)
            fig.add_vline(
                x=self.mean[j], line_width=3, line_dash="dash", line_color="black"
            )

        fig.update_layout(
            barmode="overlay",
            title="Posterior Distribution",
            xaxis_title="Value",
            yaxis_title="Density",
        )
        fig.show()

    def summary_table(self):
        """
        Display summary statistics in a table.
        """
        summary_stats = self.get_summary_statistics()

        header = ["Statistic", "Value"]
        values = [
            ["Mean", summary_stats["mean"]],
            ["Median", summary_stats["median"]],
            ["Standard Deviation", summary_stats["std"]],
            ["95% CI Lower", summary_stats["ci_lower"]],
            ["95% CI Upper", summary_stats["ci_upper"]],
        ]

        fig = self.go.Figure(
            data=[
                self.go.Table(
                    header=dict(values=header),
                    cells=dict(
                        values=[[row[0] for row in values], [row[1] for row in values]]
                    ),
                )
            ]
        )

        fig.update_layout(title="Summary Statistics")
        fig.show()

    def autocorrelation(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the autocorrelation (Pearson correlation coefficient)
        of a numpy array representing samples.
        """
        x = (x - x.mean()) / (x.std() * np.sqrt(len(x)))
        cor = np.correlate(x, x, mode="full")
        return cor[len(x) : -1]

    def _autocorrelate_negative(self, autocorrelation):
        """
        Returns the index of the first negative entry in ``autocorrelation``, or
        ``len(autocorrelation)`` if a negative entry is not found.
        """
        negative_indices = np.where(autocorrelation < 0)[0]
        return (
            negative_indices[0] if negative_indices.size > 0 else len(autocorrelation)
        )

    def effective_sample_size(self):
        """
        Computes the effective sample size for each parameter in each chain.
        """

        if not isinstance(self.all_samples, np.ndarray) or self.all_samples.ndim != 2:
            raise ValueError("Samples must be of type np.ndarray with dims == 2")
        if self.all_samples.shape[0] < 2:
            raise ValueError("At least two samples must be given.")

        ess = []
        for _, chain in enumerate(self.chains):
            for j in range(self.num_parameters):
                rho = self.autocorrelation(chain[:, j])
                T = self._autocorrelate_negative(rho)
                ess.append(len(chain[:, j]) / (1 + 2 * rho[:T].sum()))

        return ess
