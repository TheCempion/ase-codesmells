# standard libraries
from typing import List

# third party libraries
import matplotlib.pyplot as plt
import numpy as np

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from configs.data_config import DataConfig
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from utils.paths import get_safe_filename


__all__ = [
    'MnistBase',
    'MnistMLP',
]


class MnistBase(AE):
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        data: DataConfig,
        output_path: str,
        model_name: str
        ) -> None:
        super().__init__(
            model_config, train_config, data, output_path, model_name
        )

    def plot_latent_space(self) -> None:
        axis = self.scatter_latent_space()
        self.plot_latent_space_interpolation(axis=axis)

    def scatter_latent_space(self) -> List[int]:
        """Create a scatter plot for the latent space, if the space has dimension 2. Saves the resulting plot.
        """
        if self.model_config.bottleneck.latent_dim != 2:
            self.logger.warning(
                f'Cannot plot latent space for latent dimension {self.model_config.bottleneck.latent_dim}'
                )
            return

        output_path = self.output_path / 'latent_space'
        tf_encoded = self.encoder.predict(self.data.test.examples, verbose=0)

        plt.figure()
        plt.scatter(
            tf_encoded[:, 0],
            tf_encoded[:, 1],
            s=1,
            c=self.data.test.labels,
            cmap='tab10'
        )
        plt.title(f'Latent space of {self.model_name}', fontsize=10)
        plt.xlabel('x')
        plt.ylabel('y')

        cbar = plt.colorbar()
        cbar.set_ticks(ticks=np.linspace(0.5, 8.5, 10))
        cbar.set_ticklabels([str(i) for i in range(10)])
        path = get_safe_filename(output_path / 'latent_space_scattered.pdf')
        plt.savefig(path, bbox_inches='tight')
        self.logger.info(f'Saved scatter plot of latent space in {path}.')

        
        x_min, x_max = tf_encoded[:, 0].min(), tf_encoded[:, 0].max()
        y_min, y_max = tf_encoded[:, 1].min(), tf_encoded[:, 1].max()
        return [x_min, x_max, y_min, y_max]

    def plot_latent_space_interpolation(
        self, axis: List[int], same_scale: bool = True, m: int = 30, n: int = 30, alpha: float = 0.1
        ) -> None:
        """Interpolates the latentspace and stores the resulting image.

        Args:
            axis (List[int]): min and max values for values in the latent space (retrieved by `scatter_latent_space()`).
            same_scale (bool): If True, the scale of both, the x and y axis are beeing equalized. Defaults to True.
            m (int, optional): Number of interpolations along the x-axis. Defaults to 30.
            n (int, optional): Number of interpolations along the y-axis. Defaults to 30.
            alpha (float, optional): Intensity of latent space plot in background. Defaults to 0.1.
        """
        if self.model_config.bottleneck.latent_dim != 2:
            self.logger.warning(
                f'Cannot plot latent space for latent dimension {self.model_config.bottleneck.latent_dim}'
                )
            return

        output_path = self.output_path / 'latent_space'

        image_size = 28
        figure = np.zeros((image_size * n, image_size * m))

        x_min, x_max, y_min, y_max = axis
        scale_x, scale_y = (x_max - x_min), (y_max - y_min) 
        max_scale = max(scale_x, scale_y)
        if same_scale:
            correction_x = max_scale / (m - 1)
            correction_y = max_scale / (n - 1)

            # update size of either axis
            if scale_x > scale_y:
                y_min -= (scale_x - scale_y) / 2
                y_max += (scale_x - scale_y) / 2
            elif scale_x < scale_y:
                x_min -= (scale_y - scale_x) / 2
                x_max += (scale_y - scale_x) / 2
        else:
            correction_x = (scale_x) / (m - 1)
            correction_y = (scale_y) / (n - 1)


        grid_x = np.linspace(x_min, x_max, m)
        grid_y = np.linspace(y_min, y_max, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample, verbose=0).clip(min=0.0, max=1.0)
                digit = x_decoded[0].reshape(image_size, image_size)
                figure[
                    i * image_size : (i + 1) * image_size,
                    j * image_size : (j + 1) * image_size,
                ] = digit

        plt.figure(figsize=(15, 15))

        encoded = self.encoder.predict(self.data.test.examples, verbose=0)
        plt.scatter(encoded[:, 0], encoded[:, 1], s=16, c=self.data.test.labels, cmap='tab10', alpha=alpha)

        x_min -= correction_x
        x_max += correction_x
        y_min -= correction_y
        y_max += correction_y
        axis = [x_min, x_max, y_min, y_max]
        plt.imshow(figure, extent=axis, cmap='gray')
        plt.axis(axis)

        path = get_safe_filename(output_path / 'latent_space_interpolated.pdf')
        plt.savefig(path)
        self.logger.info(f'Saved plot of interpolated latent space in {path}.')
