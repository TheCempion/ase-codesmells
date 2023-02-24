# standard libraries
from typing import List

# third party libraries
import matplotlib.pyplot as plt
import numpy as np

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from models.base_models.MnistBase import MnistBase
from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from utils.DataGenerator import DataGenerator
from utils.paths import get_safe_filename


__all__ = [
    'MnistBase',
    'MnistMLP',
]


class MnistMLP(MnistBase):
    def __init__(
        self, model_config: ModelConfig, train_config: TrainingConfig, output_path: str, model_name: str, noise: float
    ) -> None:
        data = DataGenerator().MNIST_mlp(noise)
        super().__init__(
            model_config, train_config, data, output_path, model_name
        )
        
    def __del__(self) -> None:
        self.save_config()
        self.plot_model()
        self.save_training_history()
        self.plot_latent_space()


class MnistCNN(MnistBase):
    def __init__(
        self, model_config: ModelConfig, train_config: TrainingConfig, output_path: str, model_name: str, noise: float
    ) -> None:
        data = DataGenerator().MNIST_cnn(noise)
        super().__init__(
            model_config, train_config, data, output_path, model_name
        )
        
    def __del__(self) -> None:
        self.save_config()
        self.plot_model()
        self.save_training_history()
        self.plot_latent_space()


class FashionMnistMLP(MnistBase):
    def __init__(
        self, model_config: ModelConfig, train_config: TrainingConfig, output_path: str, model_name: str, noise: float
    ) -> None:
        data = DataGenerator().fashion_MNIST_mlp(noise)
        super().__init__(
            model_config, train_config, data, output_path, model_name
        )
        
    def __del__(self) -> None:
        self.save_config()
        self.plot_model()
        self.save_training_history()
        self.plot_latent_space()


class FashionMnistCNN(MnistBase):
    def __init__(
        self, model_config: ModelConfig, train_config: TrainingConfig, output_path: str, model_name: str, noise: float
    ) -> None:
        data = DataGenerator().fashion_MNIST_cnn(noise)
        super().__init__(
            model_config, train_config, data, output_path, model_name
        )
        
    def __del__(self) -> None:
        self.save_config()
        self.plot_model()
        self.save_training_history()
        self.plot_latent_space()
