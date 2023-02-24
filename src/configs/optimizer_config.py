# standard libraries
from dataclasses import dataclass
from abc import ABC

# third party libraries
from keras.optimizers import Adam, SGD
# from tensorflow import keras

# local libraries


__all__ = [
    'AdamConfig',
    'OptimizerConfig',
    'SGDConfig'
]


@dataclass
class OptimizerConfig(ABC):
    pass


@dataclass
class AdamConfig(OptimizerConfig):
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999,
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        self.optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon
            )
        self.name = 'Adam'

@dataclass
class SGDConfig(OptimizerConfig):
    learning_rate: float = 0.01 
    momentum: float = 0.9,  # tf-default is 0.0
    nesterov: bool = False

    def __post_init__(self) -> None:
        self.optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov
            )
        self.name = 'SGD'
