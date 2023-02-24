"""Encapsulates (static) methods that plot the data for an AE trained on "circle-data".

Explanation naming convention of the plots:
    *_training_interval: Data generated from the training interval (e.g. [-1, 0.5]). Either a part of a circle
        or directly as latent space. Actually, the input data is generated of the latent space interval.
    *_entire_circle: consider data generated from the interval [-pi, pi]. This is used for comparisons. This means
        an AE (therefore additionally encoder and decoder) trained on the training interval gets the entire circle, or
        the latent space representation (i.e. decoder) as input. From this, conclusions on the "generalization" might
        be drawn. For example: Training two AEs on the intervals [-2. -0.5] and [-1, 0.5], repectively: Are there any
        similarities when looking at the latent space or the interpolated latent space when looking on the entire
        circle and not just on the intervaals the AEs were trained on.
    *_comparison: Plots that shows a direct comparison of the results with data from the training interval against
        the results of the same AE on the entire circle.
"""

# standard libraries
import pickle
from pathlib import Path
from typing import List

# third party libraries
from numpy import pi
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from matplotlib.axes import Axes

# local libraries
from utils.plotting.plotting import *
from utils.plotting import normal_equations as norm_eq


__all__ = [
    'plots_encoder',
    'plots_decoder',
    'plot_forwardpass',
    'plots_input_domain',
    'save_figure',
]


# !!!!!!!!!!!!!!!!!!!!!!!! CRITICAL !!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: read notes from last meeting with Zemke: Keyword "Ausgleichsproblem" to find outlier -> THEN PRINT (or similar)
#       and also plot the Ausgleichsproblem line into the current plots TODO: create even more plots (with and without 
#       that line). Also note the error somewhere -> Need to know!!!!
#       WRITE some kind of report afterwards

# TODO: naming of plots and axes of plots

intercept = lambda theta: f'+ {theta[0]:.3f}' if theta[0] >= 0 else f'- {abs(theta[0]):.3f}'

def plot_forwardpass(plotting_data: PlottingData, output_path: Path) -> None:
    """Plots the forward pass of given data through the trained AE.
    
    How well can the AE actually reconstruct the input? 

    Args:
        plotting_data (PlottingData): Contains data for the plots.
        output_path (Path): Path where the plots will be saved.
    """
    # forward pass on training data interval
    sin_train_in, cos_train_in = plotting_data.autoencoder_train['inputs']
    sin_train_out, cos_train_out = plotting_data.autoencoder_train['outputs']
    latent_train = plotting_data.inputs_train_generated_from    # only needed for scatter plots

    latent_min, latent_max = round(latent_train.min(), 3), round(latent_train.max(), 3)
    latent_interval_train_str = f'[{latent_min}, {latent_max}]'

    fig = plt.figure(figsize=(10, 15))

    ax1 = plt.subplot(321)  # input 1 (sin)
    ax1.plot(latent_train, sin_train_in)
    ax1.set_xlim([-pi*1.2, pi*1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'sin(x) (input 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')

    ax2 = plt.subplot(322, sharex=ax1, sharey=ax1)  # ouput 1 (sin)
    ax2.plot(latent_train, sin_train_out)
    ax2.set_title(f'Approx. of sin(x) (output 1)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')

    ax3 = plt.subplot(323)  # input 2 (cos)
    ax3.plot(latent_train, cos_train_in)
    ax3.set_xlim([-pi*1.2, pi*1.2])
    ax3.set_ylim([-1.2, 1.2])
    ax3.set_title(f'cos(x) (input 2)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('cos(x)')

    ax4 = plt.subplot(324, sharex=ax3, sharey=ax3)  # ouput 2 (cos)
    ax4.plot(latent_train, cos_train_out)
    ax4.set_title(f'Approx. of cos(x) (output 2)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('AE(x) [1]' + r'$\approx$' + 'cos(x)')

    ax5 = plt.subplot(325)  # input circle
    ax5.set_xlim([-1.2, 1.2])
    ax5.set_ylim([-1.2, 1.2])
    ax5.set_title(f'2D Input for AE')
    ax5.set_xlabel('sin(x)')
    ax5.set_ylabel('cos(x)')
    scatter(ax5, sin_train_in, cos_train_in, c=latent_train)

    ax6 = plt.subplot(326, sharex=ax5, sharey=ax5)  # output circle
    scatter(ax6, sin_train_out, cos_train_out, c=latent_train)
    ax6.set_title(f'2D Output of AE')
    ax6.set_xlabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax6.set_ylabel('AE(x) [1] ' + r'$\approx$' + ' cos(x)')

    fig.suptitle(f'Forwardpass on training data\n(trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_training_interval')
    plt.close()

    # forward pass with latent representation
    fig = plt.figure(figsize=(15, 10))

    # input 1
    ax1 = plt.subplot(231)  # input 1 (sin)
    ax1.plot(latent_train, sin_train_in)
    ax1.set_xlim([-pi*1.2, pi*1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'sin(x) (input 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')

    # output 1
    ax2 = plt.subplot(233, sharex=ax1, sharey=ax1)  # ouput 1 (sin)
    ax2.plot(latent_train, sin_train_out)
    ax2.set_title(f'Approx. of sin(x) (output 1)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')

    # latent representation
    latent = plotting_data.encoder_train['outputs'][0]
    ax3 = plt.subplot(335, sharex=ax1)
    ax3.plot(latent_train, latent)
    ax3.set_title(f'Latent representation')
    ax3.set_xlabel('x')
    # ax3.set_ylabel('Latent space')

    # input 2
    ax4 = plt.subplot(234)  # input 2 (cos)
    ax4.plot(latent_train, cos_train_in)
    ax4.set_xlim([-pi*1.2, pi*1.2])
    ax4.set_ylim([-1.2, 1.2])
    ax4.set_title(f'cos(x) (input 2)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('cos(x)')

    # output 2
    ax5 = plt.subplot(236, sharex=ax4, sharey=ax4)  # ouput 2 (cos)
    ax5.plot(latent_train, cos_train_out)
    ax5.set_title(f'Approx. of cos(x) (output 2)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('AE(x) [1]' + r'$\approx$' + 'cos(x)')

    fig.suptitle(f'Forwardpass and latent representation\n(trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_with_LS')
    plt.close()

    # forward pass with latent representation AND sine/cosine in AE output (+ errors)
    fig = plt.figure(figsize=(15, 10))

    # input 1
    ax1 = plt.subplot(231)  # input 1 (sin)
    ax1.plot(latent_train, sin_train_in)
    ax1.set_xlim([-pi*1.2, pi*1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'sin(x) (input 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')

    # output 1
    ax2 = plt.subplot(233, sharex=ax1, sharey=ax1)  # ouput 1 (sin)
    ax2.plot(latent_train, sin_train_out, label='Output AE')
    ax2.set_ylabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    mean_err = abs(sin_train_out - sin_train_in).mean() # TODO: squared error?
    std_err = abs(sin_train_out - sin_train_in).std()
    ax2.plot(latent_train, sin_train_in, label='Sine reference\nErr: '
                  + r'$\mu =$' + f'{mean_err:.3f}, '
                  + r'$\sigma =$' + f'{std_err:.3f}')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_title(f'Approx. of sin(x) (output 1)')   
    
    # latent representation
    latent = plotting_data.encoder_train['outputs'][0]
    with (output_path / 'mep.pkl').open('rb') as f:
        mep = pickle.load(f)
    theta = mep['theta'].flatten()
    ax3 = plt.subplot(335, sharex=ax1)
    ax3.plot(latent_train, latent, label='Latent space')
    ax3.plot(mep['train_interval'], mep['latent_approx'],
             label=f'LS(x) ' + r'$\approx$' + f' {theta[1]:.3f}x {intercept(theta)}')
    ax3.legend()
    ax3.set_xlabel('x')
    # ax3.set_ylabel('Latent space')
    ax3.set_title(f'Latent representation')

    # input 2
    ax4 = plt.subplot(234)  # input 2 (cos)
    ax4.plot(latent_train, cos_train_in)
    ax4.set_xlim([-pi*1.2, pi*1.2])
    ax4.set_ylim([-1.2, 1.2])
    ax4.set_title(f'cos(x) (input 2)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('cos(x)')

    # output 2
    ax5 = plt.subplot(236, sharex=ax4, sharey=ax4)  # ouput 2 (cos)
    ax5.plot(latent_train, cos_train_out, label='Output AE')
    mean_err = abs(cos_train_out - cos_train_in).mean()
    std_err = abs(cos_train_out - cos_train_in).std()
    ax5.plot(latent_train, cos_train_in, label='Cosine reference\nErr: '
                  + r'$\mu =$' + f'{mean_err:.3f}, '
                  + r'$\sigma =$' + f'{std_err:.3f}')
    ax5.legend()
    ax5.set_xlabel('x')
    ax5.set_ylabel('AE(x) [1]' + r'$\approx$' + 'cos(x)')
    ax5.set_title(f'Approx. of cos(x) (output 2)')

    fig.suptitle(f'Forwardpass and latent representation\n(trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_with_LS_and_with_err')
    plt.show()

    # forward pass circle -> LS -> circle-out
    fig = plt.figure(figsize=(15, 6))

    # input circle
    ax1 = plt.subplot(131)  
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'2D Input for AE')
    ax1.set_xlabel('sin(x)')
    ax1.set_ylabel('cos(x)')
    ax1.scatter(sin_train_in, cos_train_in, s=1)

    # latent representation
    latent = plotting_data.encoder_train['outputs'][0]
    with (output_path / 'mep.pkl').open('rb') as f:
        mep = pickle.load(f)
    theta = mep['theta'].flatten()
    ax3 = plt.subplot(132)
    ax3.set_xlim([-pi*1.2, pi*1.2])
    ax3.plot(latent_train, latent, label='Latent space')
    ax3.plot(mep['train_interval'], mep['latent_approx'],
             label=f'LS(x) ' + r'$\approx$' + f' {theta[1]:.3f}x {intercept(theta)}')
    ax3.legend()
    ax3.set_xlabel('x')
    ax3.set_ylabel('Latent space')
    ax3.set_title(f'Latent representation')

    # output circle
    ax2 = plt.subplot(133, sharex=ax1, sharey=ax1)
    ax2.scatter(sin_train_out, cos_train_out, s=1)
    ax2.set_title(f'2D Output of AE')
    ax2.set_xlabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax2.set_ylabel('AE(x) [1] ' + r'$\approx$' + ' cos(x)')

    fig.suptitle(f'Forwardpass and latent representation\n(trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_circle-in_LS_circle-out')
    plt.close()

    # forward pass circle -> LS -> circle-out with output-reference
    fig = plt.figure(figsize=(15, 6))

    # input circle
    ax1 = plt.subplot(131)  
    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'2D Input for AE')
    ax1.set_xlabel('sin(x)')
    ax1.set_ylabel('cos(x)')
    ax1.scatter(sin_train_in, cos_train_in, s=1)

    # latent representation
    latent = plotting_data.encoder_train['outputs'][0]
    with (output_path / 'mep.pkl').open('rb') as f:
        mep = pickle.load(f)
    theta = mep['theta'].flatten()
    ax2 = plt.subplot(132)
    ax2.set_xlim([-pi*1.2, pi*1.2])
    ax2.plot(latent_train, latent, label='Latent space')
    ax2.plot(mep['train_interval'], mep['latent_approx'],
             label=f'LS(x) ' + r'$\approx$' + f' {theta[1]:.3f}x {intercept(theta)}')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('Latent space')
    ax2.set_title(f'Latent representation')

    # output circle
    ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    ax3.scatter(sin_train_out, cos_train_out, s=1, label='Output AE')
    ax3.scatter(sin_train_in, cos_train_in, s=1, label='Circle reference')
    ax3.legend()
    ax3.set_xlabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax3.set_ylabel('AE(x) [1] ' + r'$\approx$' + ' cos(x)')
    ax3.set_title(f'2D Output of AE')

    fig.suptitle(f'Forwardpass and latent representation\n(trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_circle-in_LS_circle-out_output-ref')
    plt.close()

    # forward pass WITHOUT latent representation BUT WITH sine/cosine in AE output (+ errors)
    fig = plt.figure(figsize=(10, 10))

    # input 1
    ax1 = plt.subplot(221)  # input 1 (sin)
    ax1.plot(latent_train, sin_train_in)
    ax1.set_xlim([-pi*1.2, pi*1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'sin(x) (input 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')

    # output 1
    ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)  # ouput 1 (sin)
    ax2.plot(latent_train, sin_train_out, label='Output AE')
    ax2.set_ylabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    mean_err = abs(sin_train_out - sin_train_in).mean()
    std_err = abs(sin_train_out - sin_train_in).std()
    ax2.plot(latent_train, sin_train_in, label='Sine reference\nErr: '
                  + r'$\mu =$' + f'{mean_err:.3f}, '
                  + r'$\sigma =$' + f'{std_err:.3f}')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_title(f'Approx. of sin(x) (output 1)')   

    # input 2
    ax3 = plt.subplot(223)  # input 2 (cos)
    ax3.plot(latent_train, cos_train_in)
    ax3.set_xlim([-pi*1.2, pi*1.2])
    ax3.set_ylim([-1.2, 1.2])
    ax3.set_title(f'cos(x) (input 2)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('cos(x)')

    # output 2
    ax4 = plt.subplot(224, sharex=ax3, sharey=ax3)  # ouput 2 (cos)
    ax4.plot(latent_train, cos_train_out, label='Output AE')
    mean_err = abs(cos_train_out - cos_train_in).mean()
    std_err = abs(cos_train_out - cos_train_in).std()
    ax4.plot(latent_train, cos_train_in, label='Cosine reference\nErr: '
                  + r'$\mu =$' + f'{mean_err:.3f}, '
                  + r'$\sigma =$' + f'{std_err:.3f}')
    ax4.legend()
    ax4.set_xlabel('x')
    ax4.set_ylabel('AE(x) [1]' + r'$\approx$' + 'cos(x)')
    ax4.set_title(f'Approx. of cos(x) (output 2)')

    fig.suptitle(f'Forwardpass (trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_sin-cos_with_with_err')
    plt.close()

    # forward pass WITHOUT latent representation
    fig = plt.figure(figsize=(10, 10))

    # input 1
    ax1 = plt.subplot(221)  # input 1 (sin)
    ax1.plot(latent_train, sin_train_in)
    ax1.set_xlim([-pi*1.2, pi*1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_title(f'sin(x) (input 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sin(x)')

    # output 1
    ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)  # ouput 1 (sin)
    ax2.plot(latent_train, sin_train_out)
    ax2.set_xlabel('x')
    ax2.set_ylabel('AE(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax2.set_title(f'Approx. of sin(x) (output 1)')   

    # input 2
    ax3 = plt.subplot(223)  # input 2 (cos)
    ax3.plot(latent_train, cos_train_in)
    ax3.set_xlim([-pi*1.2, pi*1.2])
    ax3.set_ylim([-1.2, 1.2])
    ax3.set_title(f'cos(x) (input 2)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('cos(x)')

    # output 2
    ax4 = plt.subplot(224, sharex=ax3, sharey=ax3)  # ouput 2 (cos)
    ax4.plot(latent_train, cos_train_out)
    ax4.set_xlabel('x')
    ax4.set_ylabel('AE(x) [1]' + r'$\approx$' + 'cos(x)')
    ax4.set_title(f'Approx. of cos(x) (output 2)')

    fig.suptitle(f'Forwardpass (trained on x = {latent_interval_train_str})')
    save_figure(output_path, 'fp_sin-cos')
    plt.close()


def plots_input_domain(plotting_data: PlottingData, output_path: Path) -> None:
    """Plots the input domain that was used for training and compares the training data with the entire circle 

    Args:
        plotting_data (PlottingData): Contains data for the plots.
        output_path (Path): Path where the plots will be saved.
    """
    # TODO: also plot input1 and input 2 (sin/cos) (separate plots) -> three more in total

    train_from_latent = plotting_data.inputs_train_generated_from
    train_inputs = plotting_data.encoder_train['inputs']
    latent_min, latent_max = round(train_from_latent.min(), 3), round(train_from_latent.max(), 3)
    latent_interval_train_str = f'[{latent_min}, {latent_max}]'
    latent_interval_circle_str = '[-' + r'$\pi$' + ',' + r'$\pi$'+ ']'

    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(121)
    ax1.set_title(f'Interval for training data \n({latent_interval_train_str})')
    ax1.set_xlabel('x')
    scatter_line(ax1, train_from_latent, train_from_latent*0, c=train_from_latent)

    ax2 = plt.subplot(122)
    ax2.set_title(f'2D Training input for AE: \n(sin(x), cos(x)) for x = {latent_interval_train_str}')
    ax2.set_xlabel('sin(x)')
    ax2.set_ylabel('cos(x)')
    scatter_circle(ax2, train_inputs[0], train_inputs[1], c=train_from_latent)  # [0]: sin; [1]: cos

    fig.suptitle('Circle-section used as training input')
    save_figure(folder=output_path, name='input_domain_training_interval')
    plt.close()

    # plots for entire circle
    circle_from_latent = plotting_data.inputs_all_generated_from
    circle_inputs = plotting_data.encoder_all['inputs']
    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(121)
    ax1.set_title(f'Interval for entire circle \n({latent_interval_circle_str})')
    ax1.set_xlabel('x')
    scatter_line(ax1, circle_from_latent, circle_from_latent*0, c=circle_from_latent)

    ax2 = plt.subplot(122)
    ax2.set_title(f'Entire circle: (sin(x), cos(x)) \nfor x = {latent_interval_circle_str}')
    ax2.set_xlabel('sin(x)')
    ax2.set_ylabel('cos(x)')
    scatter_circle(ax2, circle_inputs[0], circle_inputs[1], c=circle_from_latent)  # [0]: sin; [1]: cos

    fig.suptitle('Data of entire circle')
    save_figure(folder=output_path, name='input_domain_entire_circle')
    plt.close()

    # comparison training data and entire circle (not necessarily need, still good to have)
    fig = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(221)
    ax1.set_title(f'Interval for training data ({latent_interval_train_str})')
    ax1.set_xlabel('x')
    scatter_line(ax1, train_from_latent, train_from_latent*0, c=train_from_latent)

    ax2 = plt.subplot(222)
    ax2.set_title(f'2D Training input for AE: (sin(x), cos(x)) \nfor x = {latent_interval_train_str}')
    ax2.set_xlabel('sin(x)')
    ax2.set_ylabel('cos(x)')
    scatter_circle(ax2, train_inputs[0], train_inputs[1], c=train_from_latent)  # [0]: sin; [1]: cos

    ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
    ax3.set_title(f'Interval for entire circle \n({latent_interval_circle_str})')
    ax3.set_xlabel('x')
    scatter_line(ax3, circle_from_latent, circle_from_latent*0, c=circle_from_latent)

    ax4 = plt.subplot(224, sharex=ax2, sharey=ax2)
    ax4.set_title(f'Entire circle: (sin(x), cos(x)) \nfor x = {latent_interval_circle_str}')
    ax4.set_xlabel('sin(x)')
    ax4.set_ylabel('cos(x)')
    scatter_circle(ax4, circle_inputs[0], circle_inputs[1], c=circle_from_latent)  # [0]: sin; [1]: cos

    fig.suptitle('Training data / circle-section vs. entire circle')
    save_figure(folder=output_path, name='input_domain_comparison')
    plt.close()


def plots_encoder(plotting_data: PlottingData, output_path: Path) -> None:
    """Creates plots that are related with the encoder of the trained AE. Plots the latent space.

    Args:
        plotting_data (PlottingData): Contains data for the plots.
        output_path (Path): Path where the plots will be saved.

    """
    # plots for training data
    train_from_latent = plotting_data.inputs_train_generated_from
    latent_min, latent_max = round(train_from_latent.min(), 3), round(train_from_latent.max(), 3)
    latent_interval_train_str = f'[{latent_min}, {latent_max}]'
    latent_interval_circle_str = '[-' + r'$\pi$' + ',' + r'$\pi$'+ ']'

    train_inputs = plotting_data.encoder_train['inputs']
    train_latent_space_repr = plotting_data.encoder_train['outputs'][0] # latent dimension of 1

    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(121)
    ax1.set_title(f'Input for encoder: Circle-section')
    ax1.set_xlabel('sin(x)')
    ax1.set_ylabel('cos(x)')
    scatter_circle(ax1, train_inputs[0], train_inputs[1], c=train_from_latent)  # [0]: sin; [1]: cos

    ax2 = plt.subplot(122)
    ax2.set_title(f'Latent space for circle-section')
    ax2.set_xlabel('x')
    ax2.set_ylabel('encoder(sin(x), cos(x))')
    ax2.plot(train_from_latent, train_latent_space_repr, label='latent space')

    fig.suptitle(f'Latentspace of "circle-section" \ngenerated on x = {latent_interval_train_str}')
    save_figure(folder=output_path, name='encoder_training_interval') # TODO: need to note somewhere the interval

    # include mathematical equality problem (MEP) (parameters retrieved using normal equation)
    theta = norm_eq.linear(train_from_latent, train_latent_space_repr)
    approx_line = theta[0] + train_from_latent*theta[1]
    ax2.plot(train_from_latent, approx_line, label=f'{theta[1]}x + {theta[0]}')
    ax2.legend()
    save_figure(folder=output_path, name='encoder_training_interval_MEP')
    plt.close()

    # plot fitted into separate plot
    fig = plt.figure(figsize=(8, 8))  # larger than normally a single plot would be (instead of (5, 5) or (6.4, 4.8))
    plt.plot(train_from_latent, train_latent_space_repr, label='latent space')
    plt.plot(train_from_latent, approx_line, label=f'approximated line ({theta[1]}x + {theta[0]})')
    plt.legend()
    plt.suptitle(f'Latent space of AE trained on x = {latent_interval_train_str}')
    save_figure(folder=output_path, name='latent_space')
    squared_error = (approx_line - train_latent_space_repr)**2
    mean = squared_error.mean()
    std = squared_error.std()
    plt.close()

    # plots for entire circle
    circle_from_latent = plotting_data.inputs_all_generated_from
    circle_inputs = plotting_data.encoder_all['inputs']
    circle_latent_space_repr = plotting_data.encoder_all['outputs'][0] # latent dimension of 1

    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(121)
    ax1.set_title(f'Input for encoder: entire circle')
    ax1.set_xlabel('sin(x)')
    ax1.set_ylabel('cos(x)')
    scatter_circle(ax1, circle_inputs[0], circle_inputs[1], c=circle_from_latent)  # [0]: sin; [1]: cos

    ax2 = plt.subplot(122)
    ax2.set_title(f'Latent space for entire circle')
    ax2.set_xlabel('x')
    ax2.set_ylabel('encoder(sin(x), cos(x))')
    ax2.plot(circle_from_latent, circle_latent_space_repr)

    fig.suptitle(f'Latentspace of "entire circle" \ngenerated from {latent_interval_circle_str}')
    save_figure(folder=output_path, name='encoder_entire_circle')
    plt.close()

    # plots for comparison of training interval and entire circle
    fig = plt.figure(figsize=(10, 10))

    # training interval (upper two plots))
    ax1 = plt.subplot(221)
    ax1.set_title(f'Input for encoder: Circle-section')
    ax1.set_xlabel('sin(x)')
    ax1.set_ylabel('cos(x)')
    scatter_circle(ax1, train_inputs[0], train_inputs[1], c=train_from_latent)  # [0]: sin; [1]: cos

    ax2 = plt.subplot(222)
    ax2.set_title(f'Latent space for circle-section')
    ax2.set_xlabel('x')
    ax2.set_ylabel('encoder(sin(x), cos(x))')
    ax2.plot(train_from_latent, train_latent_space_repr)

    # entire circle (lower two plots)
    ax4 = plt.subplot(223, sharex=ax1, sharey=ax1)
    ax4.set_title(f'Input for encoder: Entire circle')
    ax4.set_xlabel('sin(x)')
    ax4.set_ylabel('cos(x)')
    scatter_circle(ax4, circle_inputs[0], circle_inputs[1], c=circle_from_latent)  # [0]: sin; [1]: cos

    ax4 = plt.subplot(224, sharex=ax2, sharey=ax2)
    ax4.set_title(f'Latent space for entire circle')
    ax4.set_xlabel('x')
    ax4.set_ylabel('encoder(sin(x), cos(x))')
    ax4.plot(circle_from_latent, circle_latent_space_repr)

    plt.suptitle(f'Comparison latent space of AE trained on x = {latent_interval_train_str}')
    save_figure(folder=output_path, name='encoder_comparison')
    plt.close()

    # save new generated data
    data = {
        'theta': theta.flatten(),
        'latent_approx': approx_line,
        'mean': mean,
        'std': std,
        'squarred_error': squared_error,
        'train_interval': train_from_latent,
    }
    with open(output_path / 'mep.pkl', 'wb') as f:
        pickle.dump(data, f)


def plots_decoder(plotting_data: PlottingData, output_path: Path) -> None:
    """Creates plots that are related with the decoder of the trained AE. Interpolates the latent space.

    Args:
        plotting_data (PlottingData): Contains data for the plots.
        output_path (Path): Path where the plots will be saved.
    """
    # interpolate on training interval (latent space)
    latent_train = plotting_data.decoder_train['inputs'][0] # 1D latent space
    latent_min, latent_max = round(latent_train.min(), 3), round(latent_train.max(), 3)
    latent_interval_train_str = f'[{latent_min}, {latent_max}]'
    latent_interval_circle_str = '[-' + r'$\pi$' + ',' + r'$\pi$'+ ']'

    outputs_train = plotting_data.decoder_train['outputs']  # 2D interpolated latent space
    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(131) 
    ax1.set_title(f'Approximation of sine (output 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('decoder(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax1.plot(latent_train, outputs_train[0])
    ax1.set_xlim(-pi*1.1,pi*1.1)

    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
    ax2.set_title(f'Approximation of cosine (output 2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('decoder(x) [1] ' + r'$\approx$' + ' cos(x)')
    ax2.plot(latent_train, outputs_train[1])

    ax3 = plt.subplot(133)
    ax3.set_title(f'Approximation of circle')
    ax3.set_xlabel('decoder(x)[0] ' + r'$\approx$' + ' sin(x)')
    ax3.set_ylabel('decoder(x)[1] ' + r'$\approx$' + ' cos(x)')
    ax3.scatter(outputs_train[0], outputs_train[1], s=1)

    fig.suptitle(f'Interpolation of latentspace: decoder(x) for x = {latent_interval_train_str}')
    save_figure(output_path, 'decoder_training_interval')
    plt.close()
    
    # interpolate on entire interval (latent space used for entire circle construction)
    latent_circle = plotting_data.decoder_all['inputs'][0] # 1D latent space
    outputs_circle = plotting_data.decoder_all['outputs']  # 2D interpolated latent space

    fig = plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(131) 
    ax1.set_title(f'Approximation of sine (output 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('decoder(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax1.plot(outputs_circle[0], outputs_circle[1])
    ax1.set_xlim(-pi*1.1,pi*1.1)

    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1) 
    ax2.set_title(f'Approximation of cosine (output 2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('decoder(x) [1] ' + r'$\approx$' + ' cos(x)')
    ax2.plot(latent_circle, outputs_circle[1])

    ax3 = plt.subplot(133)
    ax3.set_title(f'Approximation of circle')
    ax3.set_xlabel('decoder(x)[0] ' + r'$\approx$' + ' sin(x)')
    ax3.set_ylabel('decoder(x)[1] ' + r'$\approx$' + ' cos(x)')
    ax3.scatter(outputs_circle[0], outputs_circle[1], s=1)

    fig.suptitle(f'Interpolation of latentspace: decoder(x) for x = {latent_interval_circle_str}')
    save_figure(output_path, 'decoder_entire_circle')
    plt.close()

    # comparison of latent space interpolations
    fig = plt.figure(figsize=(15,10))

    # interpolation on training data (upper three plots)
    ax1 = plt.subplot(231) # output 1 ("sine")
    ax1.set_title(f'Approximation of sine (output 1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('decoder(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax1.plot(latent_train, outputs_train[0])
    ax1.set_xlim(-pi*1.1,pi*1.1)

    ax2 = plt.subplot(232, sharex=ax1, sharey=ax1)
    ax2.set_title(f'Approximation of cosine (output 2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('decoder(x) [1] ' + r'$\approx$' + ' cos(x)')
    ax2.plot(latent_train, outputs_train[1])

    ax3 = plt.subplot(233)
    ax3.set_title(f'Approximation of circle-section')
    ax3.set_xlabel('decoder(x)[0] ' + r'$\approx$' + ' sin(x)')
    ax3.set_ylabel('decoder(x)[1] ' + r'$\approx$' + ' cos(x)')
    ax3.scatter(outputs_train[0], outputs_train[1], s=1)

    # interpolation on entire interval/circle (lower three plots)
    ax4 = plt.subplot(234, sharex=ax1, sharey=ax1) 
    ax4.set_title(f'Approximation of sine (output 1)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('decoder(x) [0] ' + r'$\approx$' + ' sin(x)')
    ax4.plot(latent_circle, outputs_circle[0])
    ax4.set_xlim(-pi*1.1,pi*1.1)

    ax5 = plt.subplot(235, sharex=ax2, sharey=ax2) 
    ax5.set_title(f'Approximation of cosine (output 2)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('decoder(x) [1] ' + r'$\approx$' + ' cos(x)')
    ax5.plot(latent_circle, outputs_circle[1])

    ax6 = plt.subplot(236, sharex=ax3, sharey=ax3)
    ax6.set_title(f'Approximation of circle')
    ax6.set_xlabel('decoder(x)[0] ' + r'$\approx$' + ' sin(x)')
    ax6.set_ylabel('decoder(x)[1] ' + r'$\approx$' + ' cos(x)')
    ax6.scatter(outputs_circle[0], outputs_circle[1], s=1)

    fig.suptitle(f'Comparison interpolations on training interval (upper) vs circle-interval (lower)\n \
                {latent_interval_train_str} vs. {latent_interval_circle_str}')
    save_figure(output_path, 'decoder_comparison')
    plt.close()


def comparison_latentspace(output_path: Path, runs: List[int]) -> None:
    """Plot the latent space of multiple trained AE for the training interval into a single plot.

    Args:
        output_path (Path): The path to the model's output, i.e. "output/<model_name>"
        runs (List[int]): Load the preprocessed data that was used for creating the plots for the runs given by the
                elements of the list, e.g. [1,2,3] plots the latent space of the trained AEs of `path / <run_x>`
                with `x` being the elements of `runs`.
    """
    # load data
    rel_path_data = 'run_{}/figures/plotting_data.pkl'
    rel_path_mep = 'run_{}/figures/mep.pkl'
    data_train_interval = []    # data from training interval
    data_circle_interval = []   # data from entire circle ([-pi, pi])
    data_mep = []
    for run in runs:
        # load plotting data
        with (output_path / rel_path_data.format(run)).open('rb') as f:
            d = pickle.load(f)
            x_values = d.inputs_train_generated_from
            latent_repr = d.encoder_train['outputs'][0]
            data_train_interval.append((x_values, latent_repr))

            x_values = d.inputs_all_generated_from
            latent_repr = d.encoder_all['outputs'][0]
            data_circle_interval.append((x_values, latent_repr))
        # load parameters for mathemaical equality problem
        with (output_path / rel_path_mep.format(run)).open('rb') as f:
            data_mep.append(pickle.load(f))
    
    output_comparisons = output_path / ('compare_latent_' + '_'.join([str(r) for r in runs]))
    suptitle = f'Comparison latent spaces of test-runs: {str(runs)[1:-1]}'
    title = f'Model: {output_path.stem}'

    # create plots for encoded input of training inputs of different models (and training intervals)
    plt.figure(figsize=(10, 10))
    for run, (x, y) in zip(runs, data_train_interval):
        plt.plot(x, y, label=f'run_{run}')  
    plt.xlim(-pi*1.1, pi*1.1)
    plt.legend()
    plt.suptitle(suptitle)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Latent space')
    save_figure(output_comparisons, 'training_interval')
    plt.show()

    # Plot overlapping latent spaces (without considering training interval)
    # NOTE: Not very helpful
    plt.figure(figsize=(5, 5))     # smaller than other single plot figures
    for run, (x, y) in zip(runs, data_train_interval):
        plt.plot(y, label=f'run_{run}') 
    plt.legend()
    plt.title('Plot overlapping latent spaces (without considering training interval)')
    plt.ylabel('Latent space')
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)   # remove tick labels
    save_figure(output_comparisons, 'latent_overlapping')
    plt.show()

    # create plots for encoded input of circle inputs on AEs trained on different (sub-)intervals
    plt.figure(figsize=(8, 8))
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (run, (x_t, y_t), (x_c, y_c)) in enumerate(zip(runs, data_train_interval, data_circle_interval)):
        plt.plot(x_c, y_c, color=plot_colors[i], linestyle=':')   # dotted in extra polation
        plt.plot(x_t, y_t, color=plot_colors[i], linestyle='-', label=f'run_{run}') # solid in training interval
    plt.legend()
    plt.suptitle(suptitle)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Latent space')
    save_figure(output_comparisons, 'circle_interval')
    plt.show()

    # create subplots to compare latent space with its extrapolation
    _, axes = plt.subplots(len(runs), 2, figsize=(10, len(runs)*5 + 1), sharex=True, sharey=True)
    for i, (run, (x, y)) in enumerate(zip(runs, data_train_interval)):
        ax = axes[i][0]
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('Latent space')
        ax.title.set_text(f'Latent space for run_{run}')
    
    for i, (run, (x, y)) in enumerate(zip(runs, data_circle_interval)):
        ax = axes[i][1]
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('Latent space')
        ax.title.set_text(f'Extrapolated latent space for run_{run}')

    plt.suptitle(f'Latent space of training data vs extrapolated latent space\n{title}')
    save_figure(output_comparisons, 'training_vs_circle')
    plt.show()

    ## compare approximated 1D-latent spaces by the encoder
    # plot fitted line vs corresponding training interval (x-values)
    plt.figure(figsize=(8, 8))     
    for run, mep in zip(runs, data_mep):
        theta = mep['theta'].flatten()
        plt.plot(mep['train_interval'], mep['latent_approx'], label=f'f_{run}(x) = {theta[1]:.3f}x {intercept(theta)}')
    plt.xlim(-pi*1.1, pi*1.1)
    plt.legend()
    plt.title('Plot fitted line vs corresponding training interval (x-values)')
    plt.xlabel('x')
    plt.ylabel('approximated latent space')
    save_figure(output_comparisons, 'approx_latent')
    plt.show()

    # Plot Latent space and the fitted line for all runs
    plt.figure(figsize=(8, 8))
    for run, (x, y) in zip(runs, data_train_interval):
        plt.plot(x, y, label=f'run_{run}')

    for run, mep in zip(runs, data_mep):
        theta = mep['theta'].flatten()
        plt.plot(mep['train_interval'], mep['latent_approx'], label=f'f_{run}(x) = {theta[1]:.3f}x {intercept(theta)}')
    plt.xlim(-pi*1.1, pi*1.1)
    plt.legend()
    plt.suptitle(suptitle)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Latent space')
    save_figure(output_comparisons, 'latent_space_with_fitted')
    plt.show()

    # Plot fitted lines (without considering training interval), therefore with overlaps
    plt.figure(figsize=(8, 8))     
    for run, mep in zip(runs, data_mep):
        theta = mep['theta'].flatten()
        plt.plot(mep['latent_approx'], label=f'f_{run}(x) = {theta[1]:.3f}x {intercept(theta)}')
    plt.legend()
    plt.title('Plot fitted lines (without considering training interval)')
    plt.ylabel('approximated latent space')
    plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False)   # remove tick labels
    save_figure(output_comparisons, 'approx_latent_overlapping')
    plt.show()

    ## store each latent space by its own (with and without fitted line)
    for run, mep, (x, y) in zip(runs, data_mep, data_train_interval):
        plt.figure(figsize=(8, 8))     
        plt.title(f'Latent space for run_{run}')
        plt.xlabel('x')
        plt.ylabel('approximated latent space')
        plt.xlim(-pi*1.1, pi*1.1)
        plt.plot(x, y, label=f'Approximated latent space')
        save_figure(output_comparisons / '../single_latent_spaces', f'latent_run_{run}')
        
        theta = mep['theta'].flatten()
        mean, std = mep['mean'], mep['std']
        plt.plot(x, mep['latent_approx'], label=f'Fitted LS: f_{run}(x) = {theta[1]:.3f}x {intercept(theta)}')
        plt.legend() 
        plt.suptitle(f'Latent space with fitted line for run_{run}')
        plt.title(f'Mean error: {mean:.3f}, Std error {std:.3f}')
        save_figure(output_comparisons / '../single_latent_spaces', f'approx_latent_run_{run}')
        plt.close()


def scatter_circle(ax: Axes, x1: ArrayLike, x2: ArrayLike, c: ArrayLike, s: float = 1, r: float = 1) -> None:
    x_lim = (-r*1.1, r*1.1)
    y_lim = (-r*1.1, r*1.1)
    scatter(ax, x1, x2, c, x_lim, y_lim, s=s)


def scatter_line(ax: Axes, x1: ArrayLike, x2: ArrayLike, c: ArrayLike, s: float = 1) -> None:
    x_lim = (-pi*1.1, pi*1.1)
    scatter(ax, x1, x2, c, x_lim=x_lim, s=s)
