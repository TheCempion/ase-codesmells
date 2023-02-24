# standard libraries
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Optional
from enum import Enum
from abc import ABC

# third party libraries
from tensorflow import keras

# local libraries
from configs.general_settings import SEED


__all__ = [
    'AvgPool',
    'BottleneckDense',
    'Conv',
    'Deconv',
    'Dense',
    'Dropout',
    'Flatten',
    'InputLayer',
    'Layer',
    'LayerType',
    'MaxPool',
    'OutputLayer',
    'Reshape',
    'Upsampling',
]


class LayerType(Enum):
    INPUT = 0
    DENSE = 1
    CONV = 2
    DECONV = 3
    BOTTLENECK_DENSE = 4
    OUTPUT = 5
    AVG_POOL = 6
    MAX_POOL = 7
    UPSAMPLING = 8
    FLATTEN = 9
    RESHAPE = 10
    DROPOUT = 11


initializer = keras.initializers.GlorotUniform

class Layer(ABC):
    '''Abstract class that has the sole purpose of defining a Layer-type.

    All methods must define a `self.layer` and `self.type`. When the model is created, the input
    will be through the network by calling the layer's layer-function, which can either be a _normal_
    layer like `Dense` or a _special_ layer like `Dropout` or an `Activation` layer.

    A model then comprises of
    - an `InputLayer`
    - an _encoder_ that is nothing else than a list of `Layer`
    - a Latentspace layer that is a `StandardLayer` which can already contain an activation function
    - a _decoder_ that  is nothing else than a list of `Layer` (similar to the _encoder_) and
    - an output layer, that is also just a standard ('predefined') layer that will be passed as argument.
    '''
    pass


@dataclass
class InputLayer:
    shape: Tuple[int]
    batch_size: int = None
    name: str = None

    def __post_init__(self):
        self.type = LayerType.INPUT
        self.layer = keras.layers.Input(self.shape, self.batch_size, self.name)


@dataclass
class Dense(Layer):
    units: int
    name: str = None
    activation: Any = None

    def __post_init__(self):
        self.type = LayerType.DENSE
        self.layer = keras.layers.Dense(
            self.units,
            activation=self.activation,
            name=self.name,
            kernel_initializer=initializer(seed=SEED)
            )


@dataclass
class Conv(Layer):
    filters: int    # num_kernel
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = 'valid'
    data_format: Optional[str] = None
    name: str = None
    activation: Any = None

    def __post_init__(self):
        self.type = LayerType.CONV
        self.layer = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name=self.name,
            kernel_initializer=initializer(seed=SEED),
            activation=self.activation
        )


@dataclass
class Deconv(Layer):
    filters: int    # num_kernel
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]] = (1, 1)
    padding: str = 'valid'
    data_format: Optional[str] = None
    output_padding: Union[int, Tuple[int], List[int], None] = None
    name: str = None
    activation: Any = None
    
    def __post_init__(self) -> None:
        self.type = LayerType.DECONV
        self.layer = keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            output_padding=self.output_padding,
            name=self.name,
            kernel_initializer=initializer(seed=SEED),
            activation=self.activation
            )


@dataclass
class BottleneckDense:
    latent_dim: int

    def __post_init__(self):
        self.type = LayerType.BOTTLENECK_DENSE
        self.layer = keras.layers.Dense(
            self.latent_dim,
            activation='linear',
            name='Latent_space',
            kernel_initializer=initializer(seed=SEED)
            )


@dataclass
class OutputLayer:
    output_dim: int
    name: str = None
    activation: Any = 'sigmoid'

    def __post_init__(self) -> None:
        self.type = LayerType.OUTPUT
        self.layer = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
            name=self.name,
            kernel_initializer=initializer(seed=SEED)
        )
        self.units = self.output_dim


@dataclass
class AvgPool(Layer):
    pool_size: Union[int, Tuple[int, int]] = (2, 2)
    strides: Union[int, Tuple[int, int]] = None
    padding: str = 'valid'
    data_format: Optional[str] = None
    name: str = None

    def __post_init__(self) -> None:
        self.type = LayerType.AVG_POOL
        self.layer = keras.layers.AvgPool2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding, 
            data_format=self.data_format,
            name=self.name
        )


@dataclass
class MaxPool(Layer):
    pool_size: Union[int, Tuple[int, int]] = (2, 2)
    strides: Union[int, Tuple[int, int]] = None
    padding: str = 'valid'
    data_format: Optional[str] = None
    name: str = None

    def __post_init__(self) -> None:
        self.type = LayerType.MAX_POOL
        self.layer = keras.layers.MaxPool2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding, 
            data_format=self.data_format,
            name=self.name
        )


@dataclass
class Upsampling(Layer):
    size: Union[int, Tuple[int, int]] = (2, 2)
    data_format: Optional[str] = None
    interpolation: str = 'nearest'
    name: str = None

    def __post_init__(self) -> None:
        self.type = LayerType.UPSAMPLING
        self.layer = keras.layers.UpSampling2D(self.size, self.data_format, self.interpolation, name=self.name)


@dataclass
class Flatten(Layer):
    data_format: Optional[str] = None
    name: str = None

    def __post_init__(self):
        self.type = LayerType.FLATTEN
        self.layer = keras.layers.Flatten(name=self.name)


@dataclass
class Reshape(Layer):
    new_shape: Tuple[int]
    input_shape: Tuple[int] = None
    name: str = None

    def __post_init__(self) -> None:
        self.type = LayerType.RESHAPE
        if self.input_shape is None:
            self.layer = keras.layers.Reshape(self.new_shape, name=self.name)
        else:
            self.layer = keras.layers.Reshape(self.new_shape, input_shape=self.input_shape, name=self.name)


@dataclass
class Dropout(Layer):
    rate: float
    name: str = None

    def __post_init__(self) -> None:
        self.type = LayerType.DROPOUT
        self.layer = keras.layers.Dropout(self.rate, name=self.name)
