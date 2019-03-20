import datetime
from collections import namedtuple
from typing import Tuple, Dict, List, Union, Optional

import numpy as np


class Context(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        source_stack.append(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        source_stack.pop()


Metric = namedtuple("Metric", "name x_label y_label")
MetricValue = Tuple[List[Context], Optional[Union[float, str]], float, datetime.datetime]
metrics: Dict[Metric, MetricValue] = {}
tracked_metrics = set()

source_stack = []


def append_metric(metric: Metric, metric_value: Tuple[Optional[Union[float, str]], float]):
    if metric not in tracked_metrics:
        return

    if metric not in metrics:
        metrics[metric] = []

    metrics[metric].append((source_stack.copy(),) + metric_value + (datetime.datetime.now(),))


def is_tracked(metric: Metric):
    return metric in tracked_metrics


def track_metric(metric: Metric):
    tracked_metrics.add(metric)


def _plot_data(x: np.ndarray, y: np.ndarray, obj):
    if np.issubdtype(x.dtype, str):
        obj.bar(np.arange(len(x)), y, tick_label=x)
        obj.xticks(rotation=55)
    else:
        obj.plot(x, y)


def plot_metrics():
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set()
    for metric in metrics:
        source_stacks, x, y, time = zip(*metrics[metric])
        source = np.array([' -> '.join(stack) for stack in source_stacks])

        for i, s in enumerate(np.unique(source)):
            indices = source == s
            x_s, y_s = (np.array([v for i, v in enumerate(x) if indices[i]]),
                        np.array([v for i, v in enumerate(y) if indices[i]]))

            # FIXME: Handle this more nicely
            if x_s[0] is None:
                x_s = np.arange(len(y_s))
            if metric.x_label != 'dist':
                _plot_data(x_s, y_s, plt)
                plt.xlabel(metric.x_label)
                plt.ylabel(metric.y_label)
            else:
                sns.distplot(y_s)

            plt.title(f'{metric.name} (From {s})')
            plt.show()


def save_metrics(path: str):
    import numpy as np
    np.save(path, metrics)


def reset_metrics():
    global metrics
    metrics = {}
