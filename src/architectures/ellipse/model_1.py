# standard libraries

# third party libraries

# local libraries
from configs.model_config import ModelConfig
from utils.Layers import Dense, BottleneckDense, InputLayer, OutputLayer
from utils.custom_activations import Abs


def model_architecture_1() -> ModelConfig:
    input_layer = InputLayer((2,), name='Input_ellipse')
    encoder = [
        Dense(64, name='Dense_encoder1', activation=Abs),
        Dense(128, name='Dense_encoder2', activation=Abs),
        Dense(32, name='Dense_encoder3', activation=Abs),
        ]
    bottleneck  = BottleneckDense(1)
    decoder = [
        Dense(32, name='Dense_decoder1', activation=Abs),
        Dense(128, name='Dense_decoder2', activation=Abs),
        Dense(64, name='Dense_decoder3', activation=Abs),
        ]
    output_layer = OutputLayer(2, name='Outputlayer', activation='linear')

    return ModelConfig(
        input_layer=input_layer,
        encoder=encoder,
        bottleneck = bottleneck,
        decoder=decoder,
        output_layer=output_layer
    )
