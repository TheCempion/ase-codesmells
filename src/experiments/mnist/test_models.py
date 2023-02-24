# standard libraries

# third party libraries

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from models.mnist_models import MnistMLP
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from configs.optimizer_config import SGDConfig 
from utils.Layers import Dense, BottleneckDense, InputLayer, OutputLayer
from utils.custom_activations import leakyReLU, Abs


def test_custom_activation_functions(model_name: str) -> AE:
    output_path = 'output'

    # training config
    optimizer = SGDConfig(momentum=0.9, learning_rate=0.03)
    epochs = 1
    batch_size = 128
    train_config = TrainingConfig(optimizer, epochs, batch_size)

    # model structure / config
    input_layer = InputLayer((784,), name='Input_MNIST')
    encoder = [
        Dense(256, name='Dense_encoder', activation=leakyReLU),
        ]
    bottleneck  = BottleneckDense(2)
    decoder = [
        Dense(256, name='Dense_decoder', activation=Abs),
        ]
    output_layer = OutputLayer(784, name='Outputlayer')

    model_config = ModelConfig(
        input_layer=input_layer,
        encoder=encoder,
        bottleneck = bottleneck,
        decoder=decoder,
        output_layer=output_layer
    )
    autoencoder = MnistMLP(
        model_config, train_config, output_path, model_name
        )
    autoencoder.compile()
    # TODO: pass tensorboard as callback function
    autoencoder.fit()
    return autoencoder
