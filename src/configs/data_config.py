# standard libraries
from dataclasses import dataclass

# third party libraries
from numpy.typing import ArrayLike

# local libraries
from configs.sample_config import SampleIntervals1DConfig

Example = ArrayLike
Label = ArrayLike


@dataclass
class ColorConfig:
    all: ArrayLike
    train: ArrayLike
    test: ArrayLike


@dataclass
class Data:
    examples: Example
    labels: Label


@dataclass(frozen=True)
class DataConfig:
    train: Data
    test: Data
    name: str = None
    colors: ColorConfig = None
    sample_config: SampleIntervals1DConfig = None
