import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def correlation(
    fig,
    ax,
    correlation,
    names=None,
    title=None,
    cmap=None,
    entry_color="w",
):
    """
    Produces a heatmap of a correlation matrix.

    :param fig:
        The ``matplotlib.Figure`` object for plotting.
    :param ax:
        The ``matplotlib.Axes`` object for plotting.
    :param correlation:
        A two-dimensional (NumPy) array that is the correlation matrix.
    :param names:
        A list of strings that are names of the variables corresponding
        to each row or column in the correlation matrix.
    :param title:
        The title of the heatmap.
    :param cmap:
        The matplotlib colormap for the heatmap.
    :param entry_color:
        The colour of the correlation matrix entries.
    """
    if cmap is None:
        cmap = plt.get_cmap("BrBG")

    # This one line produces the heatmap.
    ax.imshow(correlation, cmap=cmap, norm=matplotlib.colors.Normalize(-1, 1))

    # Define the coordinates of the ticks.
    ax.set_xticks(np.arange(len(correlation)))
    ax.set_yticks(np.arange(len(correlation)))

    # Display the names alongside the rows and columns.
    if names is not None:
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        # Rotate the labels at the x-axis for better readability.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Plot the correlation matrix entries on the heatmap.
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            if i == j:
                color = "w"
            else:
                color = entry_color
            ax.text(
                j,
                i,
                f"{correlation[i][j]:3.2f}",
                ha="center",
                va="center",
                color=color,
                in_layout=False,
            )

    ax.set_title(title or "Correlation matrix")
    fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-1, 1), cmap=cmap
        ),
        ax=ax,
        label="correlation",
    )
    fig.tight_layout()
