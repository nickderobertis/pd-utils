from typing import List, Optional

import pandas as pd
from pandas.plotting._matplotlib.style import _get_standard_colors
import matplotlib.pyplot as plt


def plot_multi_axis(df: pd.DataFrame, cols: Optional[List[str]] = None, spacing: float = .1, **kwargs) -> plt.Axes:
    """
    Plot multiple series with different y-axes

    Adapted from https://stackoverflow.com/a/50655786

    :param df: Data to be plotted
    :param cols: subset of columns to plot
    :param spacing: Amount of space between y-axes beyond the two which are on the sides of the box
    :param kwargs: df.plot kwargs
    :return:
    """
    if cols is None:
        cols = df.columns
    if len(cols) == 0:
        raise ValueError('if cols are passed, must not be an empty list')

    # Get default color style from pandas - can be changed to any other color list
    colors = _get_standard_colors(num_colors=len(cols))

    # First axis
    ax = df.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        df.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax
