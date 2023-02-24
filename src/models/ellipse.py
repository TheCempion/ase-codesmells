# standard libraries
import pickle

# third party libraries
import numpy as np
from numpy import pi

# local libraries
from models.base_models.AutoencoderBase import AutoencoderBase as AE
from utils.plotting.plotting import PlottingData, InOutData, ConsolidatedData, ManifoldData
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
        enc_train = InOutData(input=[enc_input_1, enc_input_2], output=[encoded])
        enc_all = InOutData(input=[enc_input_1_all, enc_input_2_all], output=[encoded_all])
        dec_train = InOutData(input=[latent_train], output=[dec_output_1, dec_output_2])
        dec_all = InOutData(input=[latent_all], output=[dec_output_1_all, dec_output_2_all])
        ae_train = InOutData(
            input=[enc_input_1, enc_input_2], output=[ae_output_1, ae_output_2])
        ae_all = InOutData(
            input=[enc_input_1_all, enc_input_2_all], output=[ae_output_1_all, ae_output_2_all])
        
        data_mf = ManifoldData(latent_train, latent_all)
        data_enc = ConsolidatedData(enc_train, enc_all)
        data_dec = ConsolidatedData(dec_train, dec_all)
        data_ae = ConsolidatedData(ae_train, ae_all)
        return PlottingData(data_mf, data_enc, data_dec, data_ae)

    def create_plots(self) -> None:
        output_path_plots = self.output_path / 'figures'
        output_path_plots.mkdir(parents=True)
        plotting_data = self.preprocess()
        with (output_path_plots / 'plotting_data.pkl').open('wb') as f:
            pickle.dump(plotting_data, f)
        plots_input_domain(plotting_data, output_path_plots)
        plots_encoder(plotting_data, output_path_plots)
        plots_decoder(plotting_data, output_path_plots)
        plot_forwardpass(plotting_data, output_path_plots) # NOTE: MUST be called AFTER plots_encoder()
