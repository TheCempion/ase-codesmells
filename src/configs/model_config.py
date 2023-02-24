# standard libraries
from dataclasses import dataclass
from typing import List

# third party libraries

# local libraries
from utils.Layers import InputLayer, Layer, BottleneckDense


__all__ = [
    'ModelConfig',
]


@dataclass(frozen=True)
class ModelConfig:
    input_layer: InputLayer
    encoder: List[Layer]
    bottleneck: BottleneckDense
    decoder: List[Layer]
    output_layer: Layer

    def get_list_of_layers(self) -> List[Layer]:
        return [self.input_layer] + self.encoder + [self.bottleneck] + self.decoder + [self.output_layer]
