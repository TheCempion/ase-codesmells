# standard libraries

# third party libraries

# local libraries
from models.base_models.MnistBase import MnistBase
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from utils.DataGenerator import ImageData


__all__ = [
    'MnistWrapper',
]


class MnistWrapper(MnistBase):
    def __init__(
        self, model_config: ModelConfig, train_config: TrainingConfig, output_path: str,
        model_name: str, noise: float, dataset: str
    ) -> None:
        data_generator = ImageData()
        if dataset.lower() == 'mlp':
            data = data_generator.MNIST_mlp(noise)
        elif dataset.lower() == 'cnn':
            data = data_generator.MNIST_cnn(noise)
        elif dataset.lower() == 'mlp-fashion':
            data = data_generator.fashion_MNIST_mlp(noise)
        elif dataset.lower() == 'cnn-fashion':
            data = data_generator.fashion_MNIST_cnn(noise)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

        super().__init__(
            model_config, train_config, data, output_path, model_name
        )

    def __del__(self) -> None:
        self.save_config()
        self.plot_model()
        self.save_training_history()
        self.plot_latent_space()
