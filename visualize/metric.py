from collections import defaultdict
from typing import Iterable, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_performance(performance: Iterable[Dict[str, float]], metrics_selected: Iterable[str] = None, *,
                     ax: Axes = None):
    """
    A generic plotting function for list of metric evaluations.

    :param performance:
    :param metrics_selected: a subset of metrics to be plotted, `None` means using all metrics existing.
    :param ax:
    :return:
    """
    def default_array_factory(length: int):
        arr = np.empty(length)
        arr.fill(np.nan)
        return arr

    N = len(performance)
    metrics = defaultdict(lambda: default_array_factory(N))

    for i, d in enumerate(performance):
        for k, v in d.items():
            if (metrics_selected is None) or (k in metrics_selected):
                metrics[k][i] = v

    if ax is None:
        ax = plt.gca()
    for k, arr in metrics.items():
        ax.plot(range(N), arr, label=k)
    ax.legend(loc='best')
    ax.set(xlabel='index of fold', ylabel='metric')
