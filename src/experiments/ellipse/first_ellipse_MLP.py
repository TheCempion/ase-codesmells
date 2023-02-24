# standard libraries

# third party libraries
import numpy as np
from numpy import pi

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from models.ellipse import EllipseMLP
from configs.sample_config import SampleIntervals1DConfig
from configs.training_config import TrainingConfig
from configs.optimizer_config import SGDConfig, AdamConfig
from utils.DataGenerator import SinCosData 
from architectures.ellipse.model_1 import model_architecture_1

  

def test_ellipse(model_name: str, epochs: int = 1) -> AE:
    output_path = 'output'

    # optimizer
    batch_size = 128
    SGD = SGDConfig(momentum=0.9, learning_rate=0.001)
    train_config_1 = TrainingConfig(SGD, epochs, batch_size)
    
    Adam = AdamConfig()
    train_config_2 = TrainingConfig(Adam, epochs, batch_size)

    # model structure / config
    model_config = model_architecture_1()

    interval_size = 2
    n_samples = 2048
    samples = [[n_samples]] * 10
    methods = [['uniform'], ['normal']] * 5
    intervals = [[(a, a + interval_size)] for a in np.linspace(-pi, pi-interval_size, 5)] * 2
    noises = [[0], [0.3]] * 5

    # TODO: different level of overlapping
    # TODO: should also iterate over different seeds, and then put them into model_name/optim/seed/runX/
    stop = False
    data_generator = SinCosData()
    for train_config in [train_config_1]:#, train_config_2, train_config_1]:
        for s, m, i, n in zip(samples, methods, intervals, noises):
            sample_config = SampleIntervals1DConfig(n_samples=s, methods=m, intervals=i, noise=n)

            data = data_generator.ellipse_2D(sample_config)
            autoencoder = EllipseMLP(
                model_config, train_config, data, output_path, model_name
                )
            autoencoder.compile()
            autoencoder.fit() # TODO: pass tensorboard as callback function
        #     if stop:
        #         break
        #     stop = True
        # break
