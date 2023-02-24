# standard libraries

# third party libraries

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from models.mnist_models import MnistMLP, MnistCNN
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from configs.optimizer_config import SGDConfig 
from utils.Layers import *
from utils.custom_activations import leakyReLU, Abs


# TODO: need to pass noise to constructor

def mnist_cnn_decoder_dense(model_name: str) -> AE:
    output_path = 'output'

    # training config
    optimizer = SGDConfig(momentum=0.9, learning_rate=0.03)
    epochs = 1
    batch_size = 128
    train_config = TrainingConfig(optimizer, epochs, batch_size)

    # model structure / config
    input_layer = InputLayer((28, 28, 1), name='Input_MNIST')
    encoder = [
        Conv(16, (3, 3), name='Conv_1', activation=leakyReLU),
        MaxPool(name='MaxPool_1'),
        Conv(8, (3, 3), name='Conv_2', activation=leakyReLU),
        MaxPool(name='MaxPool_2'),
        Flatten()
        ]
    bottleneck = BottleneckDense(2)
    decoder = [
        Dense(128, activation='relu'),
        Dense(512, activation='relu'),
        Dense(784, activation='sigmoid'),
    ]
    output_layer = Reshape((28, 28, 1))

    model_config = ModelConfig(
        input_layer=input_layer,
        encoder=encoder,
        bottleneck = bottleneck,
        decoder=decoder,
        output_layer=output_layer
    )
    autoencoder = MnistCNN(
        model_config, train_config, output_path, model_name
        )
    autoencoder.compile()
    # TODO: pass tensorboard as callback function
    autoencoder.fit()
    return autoencoder


def mnist_cnn_both_conv(model_name: str) -> AE:
    output_path = 'output'

    # training config
    optimizer = SGDConfig(momentum=0.9, learning_rate=0.03)
    epochs = 3
    batch_size = 128
    train_config = TrainingConfig(optimizer, epochs, batch_size)

    # model structure / config
    input_layer = InputLayer((28, 28, 1), name='Input_MNIST')
    encoder = [
        Conv(16, (3, 3), name='Conv_1', activation=leakyReLU, padding='same'),
        MaxPool(name='MaxPool_1'),
        Conv(8, (3, 3), name='Conv_2', activation=leakyReLU, padding='same'),
        MaxPool(name='MaxPool_2'),
        Flatten(),
        Dense(64, name='Encoder_dense_1', activation=leakyReLU)
        ]
    bottleneck = BottleneckDense(2)
    decoder = [
        Dense(64, name='Decoder_dense_1', activation=leakyReLU),
        Dense(392, name='Decoder_dense_2', activation=leakyReLU),
        Reshape((7, 7, 8)),     # Reshape((5, 5, 8), input_shape(2,)),
        Upsampling(name='Upsampling_1'),
        Deconv(8, (3, 3), name='Deconv_1', activation=leakyReLU, padding='same'),
        Upsampling(name='Upsampling_2'),
        
        ]
    output_layer = Deconv(1, (3, 3), name='Deconv_2', activation='sigmoid', padding='same')

    model_config = ModelConfig(
        input_layer=input_layer,
        encoder=encoder,
        bottleneck = bottleneck,
        decoder=decoder,
        output_layer=output_layer
    )
    autoencoder = MnistCNN(
        model_config, train_config, output_path, model_name
        )
    autoencoder.compile()
    # TODO: pass tensorboard as callback function
    autoencoder.fit()
    return autoencoder


def mnist_cnn_test_deconv(model_name: str) -> AE:
    output_path = 'output'

    # training config
    optimizer = SGDConfig(momentum=0.9, learning_rate=0.03)
    epochs = 1
    batch_size = 128
    train_config = TrainingConfig(optimizer, epochs, batch_size)

    # model structure / config
    input_layer = InputLayer((28, 28, 1), name='Input_MNIST')
    encoder = [
        Conv(16, (3, 3), name='Conv_1', activation=leakyReLU, padding='same'),
        MaxPool(name='MaxPool_1'),
        Conv(8, (3, 3), name='Conv_2', activation=leakyReLU, padding='same'),
        MaxPool(name='MaxPool_2'),
        Flatten(),
        Dense(64, name='Encoder_dense_1', activation=leakyReLU)
        ]
    bottleneck = BottleneckDense(2)
    decoder = [
        Dense(64, name='Decoder_dense_1', activation=leakyReLU),
        Dense(392, name='Decoder_dense_2', activation=leakyReLU),
        Dense(784, name='Decoder_dense_3', activation='relu'),
        Reshape((28, 28, 1)),     # Reshape((5, 5, 8), input_shape(2,)),
        ]
    output_layer = Conv(1, (1, 1), name='Deconv_2', activation='sigmoid', padding='same')

    model_config = ModelConfig(
        input_layer=input_layer,
        encoder=encoder,
        bottleneck = bottleneck,
        decoder=decoder,
        output_layer=output_layer
    )
    autoencoder = MnistCNN(
        model_config, train_config, output_path, model_name
        )
    autoencoder.compile()
    # TODO: pass tensorboard as callback function
    autoencoder.fit()
