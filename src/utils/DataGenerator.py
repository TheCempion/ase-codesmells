# standard libraries
from typing import Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# third party libraries
import tensorflow as tf
import numpy as np
from numpy import pi
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

# local libraries
from configs.data_config import DataConfig, Data, Example, ColorConfig
from configs.sample_config import SampleIntervals1DConfig
from utils.plotting.plotting import plot_3d
from utils.paths import add_suffix, get_safe_filename


__all__ = [
    'DataGenerator',
]


class DataGenerator:
    # NOTE: First dimension of data must be the number of samples

    def __init__(self, split_factor: float = 0.8, output_path: Path = Path('figures'), seed=42) -> None:
        self._split_factor = split_factor
        self.output_path = output_path
        self.seed = seed
        np.random.seed(seed=seed)

    def add_noise_old(self, noise: float, train: Example, test: Example) -> Tuple[Optional[Example], Optional[Example]]:
        if not 0 < noise <= 1:
            return (None, None)
        train_noisy = train + noise * np.random.normal(size=train.shape) 
        test_noisy = test + noise * np.random.normal(size=test.shape) 
        return train_noisy, test_noisy

    def add_noise(self, noise: float, data: np.ndarray) -> np.ndarray:
        if noise <= 0:
            return data
        return data + noise * np.random.normal(size=data.shape)

    def split_train_test(self, X: ArrayLike, Y: ArrayLike) -> ArrayLike:
        split_point = round(self._split_factor * X.shape[0])
        X_train, X_test =  X[:split_point], X[split_point:]
        Y_train, Y_test =  Y[:split_point], Y[split_point:]
        return X_train, X_test, Y_train, Y_test

    def get_sample_intervals_1D(self, sample_config: SampleIntervals1DConfig) -> ArrayLike:
        latent_data = []
        for samples, method, interval, noise in sample_config:
            latent, _ = self.get_samples_1D(method, samples, interval=interval, noise=noise)
            latent_data += latent.tolist()
        x_latent = np.array(latent_data)
        y_latent = x_latent * 0
        return x_latent, y_latent

    def get_samples_1D(self, method: str, n_samples: int, interval: Tuple[float, float], noise: float) -> ArrayLike:
        low, high = interval
        if method == 'normal' or method == 'gaussian':
            mean = (high + low) / 2
            x_latent = np.random.normal(loc=mean, size=n_samples)
        elif method == 'uniform':
            x_latent = np.random.uniform(low=low, high=high, size=n_samples)

        x_latent = self.add_noise(noise, x_latent)
        y_latent = x_latent * 0
        return x_latent, y_latent
    


class SinCosData(DataGenerator):
    """Considers functions that have some kind of periodicy, i.e. sin/cos

    Args:
        DataGenerator (_type_): _description_
    """
    def get_ellipse_2D(self, x_values: ArrayLike, radius: Tuple[float, float]) -> ArrayLike:
        sin_x = radius[0] * np.sin(x_values)
        cos_x = radius[1] * np.cos(x_values)
        return sin_x, cos_x

    def plot_ellipse_2D(
        self, ambient: ArrayLike, latent: ArrayLike, radius: Tuple[float, float], plot_name: Path
        ) -> None:
        plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(121)
        ax1.scatter(ambient[0], ambient[1], s=1, c=latent[0])
        ax1.set_xlim([-radius[0]*1.1, radius[0]*1.1])
        ax1.set_ylim([-radius[1]*1.1, radius[1]*1.1])
        
        ax2 = plt.subplot(122)
        ax2.scatter(latent[0], latent[1], s=1, c=latent[0])
        ax2.set_xlim([-pi*1.1, pi*1.1])
        plt.savefig(plot_name)

    def ellipse_2D(
        self,
        sample_config: SampleIntervals1DConfig,
        plot_name: str = 'ellipse',
        radius: Union[Tuple[float, float], float] = (1.0, 1.0)
        ) -> DataConfig:

        if type(radius) != tuple:
            radius = radius, radius

        x_latent, y_latent = self.get_sample_intervals_1D(sample_config)

        sin_x, cos_x = self.get_ellipse_2D(x_latent, radius)
        X = np.array((sin_x, cos_x)).T
        X_train, X_test, Y_train, Y_test = self.split_train_test(X, x_latent)

        # plots TODO
        # n_samples = sample_config.total_num_samples
        # figname = add_suffix(
        #         get_safe_filename(self.output_path / 'ellipse' / f'{plot_name}_{n_samples}'), '.pdf')
        # self.plot_ellipse_2D((sin_x, cos_x), (x_latent, y_latent), radius, figname)

        split_point = round(self._split_factor * max(x_latent.shape))
        latent_train, latent_test =  x_latent[:split_point], x_latent[split_point:]
        colors = ColorConfig(x_latent, latent_train, latent_test)

        return DataConfig(
            Data(X_train, Y_train), Data(X_test, Y_test), name='ellipse', colors=colors, sample_config=sample_config
            )

    def smooth_functions(
        self,
        n_samples: int,
        noise: float = 0.0,
        low: float = -pi,
        high: float = pi,
        plot: bool = False,
        plot_name: str = 'Smooth_functions'
        ) -> DataConfig:

        Y = np.random.uniform(low=low, high=high, size=n_samples)
        X = np.array((np.sin(Y), np.sin(2 * Y), np.cos(Y), np.cos(Y / 2), np.exp(-(Y**2))))
        X = X.T     # get format(n_samples, 5) since 5 functions are used

        X_train, X_test, Y_train, Y_test = self.split_train_test(X, Y)
        X_train_noise, X_test_noise = self.add_noise_old(noise=noise, train=X_train, test=X_test)

        if plot:
            # TODO: give each subplot a titile => the underlying function
            # plot functions
            plt.figure(figsize=(15, 15))
            ax = None
            n = min(X_train.shape)
            for i in range(n):
                ax = plt.subplot(n, 1, i + 1, sharey=ax)
                plt.plot(Y_train, X_train[:, i], '.')
            figname = add_suffix(
                get_safe_filename(self.output_path / 'smooth_functions' / f'{plot_name}_{n_samples}'), '.pdf')
            plt.savefig(figname)
        
            if noise:
                ax = None
                plt.figure(figsize=(15, 15))
                for i in range(n):
                    ax = plt.subplot(n, 1, i + 1, sharey=ax)
                    plt.plot(Y_train, X_train_noise[:, i], '.')
                plt.title(f'{plot_name.replace("_", " ")} (noise: {noise})')
                figname = add_suffix(figname, '_noise', before_extension=True)
                plt.savefig(figname)

        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='smooth_func', noise=noise
            )

    def swiss_roll(self, n_samples: int, noise: float = 0.0, plot_name: str = 'My_swiss_roll') -> DataConfig:
        # my swiss roll
        Y1 = np.random.uniform(low=3*pi/2 , high=9*pi/2, size=n_samples)
        Y2 = np.random.uniform(low=0 , high=15, size=n_samples)
        X1, X2, X3 = Y1 * np.cos(Y1), Y2, Y1 * np.sin(Y1)
        X = np.array([X1, X2, X3]).T
        Y = np.array([Y1, Y2]).T

        X_train, X_test, Y_train, Y_test = self.split_train_test(X, Y)
        X_train_noise, X_test_noise = self.add_noise_old(noise=noise, train=X_train, test=X_test)
        
        colors = np.linalg.norm(np.array([X1, X3]), axis=0)  # get appropriate coloring of data points
        colors_train = np.linalg.norm(np.array([X_train[:, 0], X_train[:, 2]]), axis=0)
        colors_test = np.linalg.norm(np.array([X_test[:, 0], X_test[:, 2]]), axis=0)

        if plot_name:
            plot_3d(X, colors, f'{plot_name}_{n_samples}', self.output_path / 'swiss_roll')
            if noise:
                plot_3d(X_train_noise, colors_train, f'{plot_name}_{n_samples}_noise', self.output_path / 'swiss_roll')

        return DataConfig(
            Data(X_train, Y_train, X_train_noise),
            Data(X_test, Y_test, X_test_noise),
            name='Swiss_roll',
            noise=noise,
            colors=ColorConfig(colors, colors_train, colors_test)
            )



class ImageData(DataGenerator):
    def MNIST_mlp(self, noise: float = 0.0) -> Any:
        mnist = tf.keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.astype('float32').reshape(-1, 784) / 255.0
        X_test = X_test.astype('float32').reshape(-1, 784) / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='MNIST', noise=noise
            )
    
    def MNIST_cnn(self, noise: float = 0.0) -> Any:
        mnist = tf.keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='MNIST', noise=noise
            )

    def fashion_MNIST_mlp(self, noise: float = 0.0) -> Any:
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train = X_train.astype('float32').reshape(-1, 784) / 255.0
        X_test = X_test.astype('float32').reshape(-1, 784) / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='fashion_MNIST', noise=noise
            )
    
    def fashion_MNIST_cnn(self, noise: float = 0.0) -> Any:
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='fashion_MNIST', noise=noise
            )

    # TODO: check dimensions in CIFAR-10/100 and if adding noise works as expected
    def CIFAR_10(self, noise: float = 0.0) -> Any:
        cifar10 = tf.keras.datasets.cifar10
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='CIFAR_10', noise=noise
            )
    
    def CIFAR_100(self, noise: float = 0.0) -> Any:
        cifar100 = tf.keras.datasets.cifar100
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train_noise, X_test_noise = self.add_noise_old(noise, X_train, X_test)
        return DataConfig(
            Data(X_train, Y_train, X_train_noise), Data(X_test, Y_test, X_test_noise), name='CIFAR_100', noise=noise
            )











    # taken from stackoverflow:
    # https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    """
    import numpy as np

    def sample_spherical(npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec
    
    # EXAMPLE:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    xi, yi, zi = sample_spherical(100)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c='r', zorder=10)
    """

# noisy images, see: https://www.tensorflow.org/tutorials/generative/autoencoder#second_example_image_denoising
"""
noise = 0.2
x_train_noisy = x_train + noise * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise * tf.random.normal(shape=x_test.shape) 

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


# plot noisy images:
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()



n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

"""