from typing import Iterable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pandas import DataFrame


class Performance:
    def __init__(self):
        self._df = DataFrame()

    def __setitem__(self, key, value):
        try:
            index, metric_name = key
        except:
            raise ValueError('Expecting the key to be iterable. \n'
                             'Hint: this object shall be called as `performance[index, metric_name]=metric_value`.')

        self._df.loc[index, metric_name] = value

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        self._df.sort_index(inplace=True)
        return self._df.columns

    def items(self):
        return ((column, self._df[column]) for column in self._df.columns)

    def __getitem__(self, item):
        self._df.sort_index(inplace=True)
        if isinstance(item, tuple):
            raise NotImplementedError('`__getitem__` on performance object is not supported yet.')
        return self._df[item]


def plot_performance(performance: Performance, metrics_selected: Iterable[str] = None, *,
                     ax: Axes = None):
    """
    A generic plotting function for list of metric evaluations.

    :param performance:
    :param metrics_selected: a subset of metrics to be plotted, `None` means using all metrics existing.
    :param ax:
    :return:
    """

    if metrics_selected is not None:
        for name, values in performance.items():
            if name in metrics_selected:
                print(f'The ending score for metric {name} is: {values.iloc[-1]:.4e}')

    if ax is None:
        ax = plt.gca()
    for name, values in performance.items():
        if metrics_selected is None or name in metrics_selected:
            ax.plot(values.index, values, label=name)
    ax.legend(loc='best')
    ax.set(xlabel='index of fold', ylabel='metric', ylim=[-0.05, 0.1])
