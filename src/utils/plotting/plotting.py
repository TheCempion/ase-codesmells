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
    'PlottingData',
    'save_figure',
    'scatter',
    # 'add_2dscatter',
    # 'plot_2d',
    # 'plot_3d',
]


@dataclass(frozen=True)
class PlottingData:
    """Dictionaries with 'inputs' and 'outputs' as keys for the corresponding nets and inputs
    will be stored in pickle-file which then can later be easiliy read and plotted

    Args:
        inputs_all_generated_from (AraryLike): Entire interval of the latent space from which the circle
                generated.
    	inputs_train_generated_from (ArrayLike): Interval of latent space, used to create section of circle that is
                used as training data.
        encoder_all (Dict[str, List[ArrayLike]]): Entire circle created from `inputs_all_generated_from` (input for
                encoder).
        encoder_train (Dict[str, List[ArrayLike]]): Training data for AE to learn latent representation for section of
                circle. Generated from `inputs_train_generated_from` (input for encoder)
        decoder_all (Dict[str, List[ArrayLike]]): Latent representation of entire interval that can be used to
                interpolate the latent space (input for decoder).
        decoder_train (Dict[str, List[ArrayLike]]): Latent representation used to interpolate LS on which the AE was
                                trained (input for decoder)
        autoencoder_train (Dict[str, List[ArrayLike]]): Forward pass with data from training interval
        autoencoder_all (Dict[str, List[ArrayLike]]): Forward pass with data from entire interval
    
    Note: This can be probably also used for any type of data, not just for a circle.
    """
    inputs_all_generated_from: ArrayLike
    inputs_train_generated_from: ArrayLike
    encoder_all: Dict[str, List[ArrayLike]]
    encoder_train: Dict[str, List[ArrayLike]]
    decoder_all: Dict[str, List[ArrayLike]]
    decoder_train: Dict[str, List[ArrayLike]]
    autoencoder_all: Dict[str, List[ArrayLike]]
    autoencoder_train: Dict[str, List[ArrayLike]]

    @staticmethod
    def format_as_field(inputs: List[ArrayLike], outputs: List[ArrayLike]) -> Dict[str, List[ArrayLike]]:
        """Formats an vector of input(vectors) and ouput(vectors) into a dictionary, s.t. it can 

        Args:
            inputs (List[ArrayLike]): For each input
            outputs (List[ArrayLike]): _description_

        Returns:
            Dict[str, List[ArrayLike]]: _description_
        """
        formatted = {
            'inputs': inputs,
            'outputs': outputs
        }
        return formatted


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