# standard libraries
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from matplotlib.axes import Axes

# third party libraries
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from matplotlib import ticker

# local libraries
from utils.paths import add_suffix, get_safe_filename


__all__ = [
    'ConsolidatedData',
    'InOutData',
    'PlottingData',
    'save_figure',
    'scatter',
]


@dataclass(frozen=True)
class InOutData:
    input: List[ArrayLike]
    output: List[ArrayLike] = None


@dataclass(frozen=True)
class ConsolidatedData:
    train: InOutData
    all: InOutData = None

@dataclass(frozen=True)
class ManifoldData:
    """Data that is assumed to be the underlying manifold and which was used to generate the training data.
    """
    train: ArrayLike
    all: ArrayLike = None


@dataclass(frozen=True)
class PlottingData:
    """Dictionaries with 'inputs' and 'outputs' as keys for the corresponding nets and inputs
    will be stored in pickle-file which then can later be easiliy read and plotted

    Args:
        _data_manifold (ManifoldData): Underlying manifold, that was used to generate the data
        _data_enc (ConsolidatedData): Input data and output of an encoder
        _data_dec (ConsolidatedData): Input data and output of a decoder
        _data_ae (ConsolidatedData): Forward pass with data from training interval
    
    Note: This can be probably also used for any type of data, not just for a circle.
    """
    _data_manifold: ManifoldData
    _data_enc: ConsolidatedData
    _data_dec: ConsolidatedData
    _data_ae: ConsolidatedData

    def get_data(self, model: str, interval: str, in_out: str) -> List[ArrayLike]:
        interval = interval.lower()
        model = model.lower()
        in_out = in_out.lower()

        assert model in ['enc', 'dec', 'ae'], f'Unkown model, only enc/dec/ae allowed, not {model}'
        assert interval in ['train', 'all'], f'Unkown interval, only train/all allowed, not {interval}'
        assert in_out in ['in', 'out'], f'Only in/out allowed, not {in_out}'

        data = self._data_enc if model == 'enc' else (self._data_dec if model == 'dec' else self._data_ae)
        data = data.train if interval == 'train' else data.all
        return data.input if in_out == 'in' else data.output
    
    def get_manifold_data(self, interval: str) -> ArrayLike:
        interval = interval.lower()
        assert interval in ['train', 'all']
        return self._data_manifold.train if interval == 'train' else self._data_manifold.all


def scatter(ax: Axes, x1: ArrayLike, x2: ArrayLike, c: ArrayLike, x_lim: Tuple[float, float] = None,
    y_lim: Tuple[float, float] = None, s: float = 1
    ) -> None:
    ax.scatter(x1, x2, s=s, c=c)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)


def save_figure(folder: Path, name: str, tight_layout: bool = True) -> None:
    """Wrapper to save a figure in mutliple formats, but always as pdf.

    Args:
        folder (Path): Output folder in which the image(s) will be stored.
        name (str): Save figure with given name.
        tight_layout (bool): If True, set `plt.tight_layout`
    """
    if tight_layout:
        plt.tight_layout()
    
    formats = ['pdf', 'png', 'svg']
    for format in formats:
        path = folder / format
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(path / f'{name}.{format}')


"""
Code for plots taken from:
https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
"""

def plot_3d(points: ArrayLike, points_color: ArrayLike, title: str, output_folder: Path):
    x, y, z = points.T
    fig, ax = plt.subplots(
        figsize=(15, 15),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title.replace('_', ' '), size=16)
    col = ax.scatter(x, y, z, s=10, c=points_color, alpha=1)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    figname = add_suffix(get_safe_filename(output_folder / title), '.pdf')
    plt.savefig(figname)


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(15, 15), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2dscatter(ax, points, points_color)
    plt.show()


def add_2dscatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())