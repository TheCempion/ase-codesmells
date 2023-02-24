# standard libraries
import pickle

# third party libraries
import numpy as np
from numpy import pi

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from utils.plotting.plotting import PlottingData
from utils.plotting.circle_plots import *


__all__ = [
    'EllipseMLP',
]


class EllipseMLP(AE):
    
    def __del__(self) -> None:
        self.save_training_history()
        self.create_plots()
        self.plot_model()
        self.save_config()

    # TODO: every model must have this function (and also create plots)
    def preprocess(self) -> PlottingData:
        latent_train = self.data.sample_config.equidistant_latent
        latent_all = np.linspace(-pi, pi, self.data.sample_config.n_samples_latent)

        # encoder training input and output
        enc_input_1, enc_input_2 = np.sin(latent_train), np.cos(latent_train)
        input_2D = np.array((enc_input_1, enc_input_2)).T
        encoded = self.encoder.predict(input_2D, verbose=0)

        # encoder input and output for entire circle
        enc_input_1_all, enc_input_2_all = np.sin(latent_all), np.cos(latent_all)
        input_2D_all = np.array((enc_input_1_all, enc_input_2_all)).T
        encoded_all = self.encoder.predict(input_2D_all, verbose=0)

        # decoder input and outputs for training interval
        self.decoder.predict(latent_train, verbose=0)
        dec_output = self.decoder.predict(latent_train, verbose=0)
        dec_output_1, dec_output_2 = dec_output[:, 0], dec_output[:, 1]
    
        # decoder input and outputs for circle
        dec_output = self.decoder.predict(latent_all, verbose=0)
        dec_output_1_all, dec_output_2_all = dec_output[:, 0], dec_output[:, 1]
    
        # forward pass through AE on training interval
        ae_output = self.model.predict(input_2D, verbose=0)
        ae_output_1, ae_output_2 = ae_output[:, 0], ae_output[:, 1]

        # forward pass through AE on entire interval/ circle
        ae_output = self.model.predict(input_2D_all, verbose=0)
        ae_output_1_all, ae_output_2_all = ae_output[:, 0], ae_output[:, 1]

        # create data structure that will be saved
        encoder_all = PlottingData.format_as_field(inputs=[enc_input_1_all, enc_input_2_all], outputs=[encoded_all])
        encoder_train = PlottingData.format_as_field(inputs=[enc_input_1, enc_input_2], outputs=[encoded])
        decoder_all = PlottingData.format_as_field(inputs=[latent_all], outputs=[dec_output_1_all, dec_output_2_all])
        decoder_train = PlottingData.format_as_field(inputs=[latent_train], outputs=[dec_output_1, dec_output_2])
        autoencoder_all = PlottingData.format_as_field(
            inputs=[enc_input_1_all, enc_input_2_all], outputs=[ae_output_1_all, ae_output_2_all])
        autoencoder_train = PlottingData.format_as_field(
            inputs=[enc_input_1, enc_input_2], outputs=[ae_output_1, ae_output_2])
        
        plotting_data = PlottingData(
            latent_all, latent_train,
            encoder_all, encoder_train,
            decoder_all, decoder_train,
            autoencoder_all, autoencoder_train)
        return plotting_data    # TODO: save permantly in create_plots-fcunction?


    def create_plots(self) -> None:
        self.output_path_plots = self.output_path / 'figures'
        self.output_path_plots.mkdir(parents=True)
        self.plotting_data = self.preprocess()
        with (self.output_path_plots / 'plotting_data.pkl').open('wb') as f:
            pickle.dump(self.plotting_data, f)
        plots_input_domain(self.plotting_data, self.output_path_plots)
        plots_encoder(self.plotting_data, self.output_path_plots)
        plots_decoder(self.plotting_data, self.output_path_plots)
        plot_forwardpass(self.plotting_data, self.output_path_plots)  # NOTE: MUST be called AFTER plots_encoder()
