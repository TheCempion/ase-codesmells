from experiments.mnist.test_models import test_custom_activation_functions as test
from experiments.mnist.mnist_cnn import *
from utils.DataGenerator import DataGenerator
from experiments.ellipse.first_ellipse_MLP import *
from configs.general_settings import SEED

import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    # test('test_custom_activations2')
    # mnist_cnn_test_deconv('test_model_with_deconv')
    # mnist_cnn_both_conv('third_test_use_deconv')
    # mnist_cnn_decoder_dense('second_test_use_dense_decoder')

    data_generator = DataGenerator()
    
    # n_samples = 20000
    # DataGenerator().smooth_functions(2000, noise=0.5, plot=True)
    # DataGenerator().swiss_roll(n_samples, plot_name = 'My_Swiss_roll', noise = 0.3)

    
    """ 
    # plot differnt ellipses

    n_samples = 1000
    data_generator.ellipse_2D(n_samples, noise=0, plot_name='plot1')
    data_generator.ellipse_2D(n_samples, interval=(-2,1), radius=(7.5, 2.5), noise=0, plot_name='plot2')
    data_generator.ellipse_2D(n_samples, method='normal', noise=0, plot_name='plot3')


    methods = ['uniform', 'gaussian', 'normal', 'uniform']
    n_samples = [150, 40, 75, 20]
    intervals = [(-3, -2.5), (-2.3, -0.3), (0, 2), (2.5, 2.9)]
    eps = [1e-10] * 4
    
    x_latent, y_latent = data_generator.get_sample_intervals_1D(n_samples, methods, intervals, eps)

    radius = (2.0, 1.0)
    sin_x, cos_x = data_generator.get_ellipse_2D(x_latent, radius)
    data_generator.plot_ellipse_2D((sin_x, cos_x), (x_latent, y_latent), radius, 'plot4')
    """
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    tf.random.set_seed(SEED)
    test_ellipse('ellipse_model_architecture_1', 200)