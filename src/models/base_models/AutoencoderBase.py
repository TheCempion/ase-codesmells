# standard libraries
import json
import time
import pickle
import logging
from pathlib import Path
from typing import List

# third party libraries
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import callbacks 

# local libraries
from configs.training_config import TrainingConfig
from configs.model_config import ModelConfig
from configs.data_config import DataConfig
from utils.paths import add_suffix, get_safe_path, get_safe_filename
from utils.config_to_dict import data_to_dict, training_to_dict


class AutoencoderBase:
    history = None
    encoder = None
    decoder = None
    model = None
    is_compiled = False
    is_trained = False
    training_time = None

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        data: DataConfig,
        output_path: str,
        model_name: str
        ) -> None:

        self.model_config = model_config
        self.train_config = train_config
        self.optimizer = train_config.optim_config.optimizer
        self.data = data
        self.model_name = model_name

        self.output_path = get_safe_path(Path(output_path) / model_name / 'run', must_have_number=True)
        logging.basicConfig(level=logging.ERROR, filename=self.output_path / f'{model_name}.log', filemode='w')
        self.logger = logging.getLogger()

        self.create_autoencoder()

    def create_autoencoder(self) -> None:
        self.logger.debug('Constructing autoencoder.')
        self.is_compiled = False
        self.is_trained = False
        input = self.model_config.input_layer.layer
        x = Activation('linear')(input)

        # traverse through encoder
        if self.model_config.encoder:
            for layer in self.model_config.encoder:
                x = layer.layer(x)
                
        latent_space = self.model_config.bottleneck.layer(x)    # act_func is linear
        x = Activation('linear')(latent_space)
        if self.model_config.decoder:
            for layer in self.model_config.decoder:
                x = layer.layer(x)
        output = self.model_config.output_layer.layer(x)

        # create encoder and decoder by training the entire model (cf. Keras Functional API)
        self.encoder = keras.models.Model(inputs=input, outputs=latent_space, name='encoder')
        self.decoder = keras.models.Model(inputs=latent_space, outputs=output, name='decoder')
        self.model = keras.models.Model(inputs=input, outputs=output, name='autoencoder')
        self.summary()
        self.logger.info('Autoencoder constructed successfully.')

    def _load_data(self) -> None:
        raise NotImplementedError('Method must be defined in child classes.')
    
    def summary(self, encoder: bool = False, decoder: bool = False) -> None:
        if encoder:
            self.encoder.summary()
        if decoder:
            self.decoder.summary()
        self.model.summary()

    def compile(self) -> None:
        if self.is_compiled is True:
            self.logger.warning('Autoencoder was already compiled and will therefore not be re-compiled.')
            return

        self.logger.debug('Start compiling autoencoder.')
        self.model.compile(
            optimizer=self.optimizer, loss=self.train_config.loss, metrics=self.train_config.metrics
            )
        self.logger.info('Compiled autoencoder successfully.')
        self.is_compiled = True
    
    def fit(
            self, callbacks: List[callbacks.Callback] = None, val_split: float = 0.0, val_batch_size: int = None) -> None:
        if self.is_trained is True:
            self.logger.warning('Autoencoder was already trained and will therefore not be re-trained.')
            return

        self.logger.debug(
            f'Training autoencoder: batch_size = {self.train_config.batch_size}, epochs = {self.train_config.epochs}.'
            )
        training_start = time.time()
        self.history = self.model.fit(
            self.data.train.examples,
            self.data.train.examples,
            epochs=self.train_config.epochs,
            batch_size=self.train_config.batch_size,
            verbose=1,
            callbacks=callbacks,
            validation_split = val_split,
            validation_data=(self.data.test.examples, self.data.test.examples),
            validation_batch_size=val_batch_size
            )
        self.training_time = time.time() - training_start
        self.logger.info('Trained autoencoder successfully.')
        self.is_trained = True

    def save_training_history(self) -> None:
        if self.history is None:
            self.logger.warning('Training history could not be saved. No history available.')
            return

        self.logger.debug('Creating plot for trainin history')
        output_path = self.output_path / 'training_history'
        plt.figure(figsize=(15, 15))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel(f'loss: {self.train_config.loss}')
        plt.xticks(
            ticks=[i+1 for i in range(self.train_config.epochs)],
            labels=[str(i+1) for i in range(self.train_config.epochs)]
            )
        plt.savefig(get_safe_filename(output_path / 'history_plot.pdf'))
        plt.close()

        # also safe as pickle-file
        with get_safe_filename(output_path / 'history.pkl').open('wb') as f:
            pickle.dump(self.history, f)
        self.logger.info('Training history was saved successfully')

    def plot_model(self, format: str = 'pdf', ) -> None:
        if not self.is_compiled:
            self.logger.warning('Cannot save plot of model. Model was not compiled, yet.')
            return

        output_path = self.output_path / 'model_plots'
        if output_path.exists():
            self.logger.warning('Seems like there already exist plots for the models. No plot saved.')
            return
        output_path.mkdir(parents=True)

        self.logger.debug('Creating plots for compiled models (encoder | decoder | autoencoder)')
        for model in [self.model, self.encoder, self.decoder]:
            model_path = add_suffix(output_path / model.name, f'.{format}')
            path_plot = get_safe_filename(model_path)
            path_plot_verbose = get_safe_filename(add_suffix(model_path, '_verbose', before_extension=True))
            keras.utils.plot_model(model, path_plot)
            keras.utils.plot_model(model, path_plot_verbose, show_shapes=True, show_layer_activations=True)
            if model == self.model: # store entire model also as png (can then be displayed in notebook)
                model_path = add_suffix(output_path / model.name, '_verbose.png')
                keras.utils.plot_model(model, model_path, show_shapes=True, show_layer_activations=True)

        self.logger.info('Successfully created and saved plots for compiled models.')

    def save_config(self) -> None:
        '''Save models and its settings in different file formats
        '''
        # save models
        self.logger.debug('Saving trained models and configurations')
        configs = {}
        output_path_models = get_safe_path(self.output_path / 'models')
        for model in [self.encoder, self.decoder, self.model]:
            model_path = output_path_models / model.name
            h5_path = add_suffix(model_path, '.h5')
            
            # import h5py

            # with h5py.File(str(h5_path), 'a') as f:
            #     # data = None
            print(str(h5_path))
            #     name = ''
                # dset_id = f.create_dataset(name, data=data)

            model.save(str(h5_path))
            configs[model.name] = str(h5_path)   # store paths, where the models are saved

            # save model as json file
            with add_suffix(model_path, '.json').open('w') as f:
                json.dump(json.loads(model.to_json()), f, indent=4)

        # save training and data configurations
        configs['training_config'] = training_to_dict(self.train_config),
        configs['data_config'] = data_to_dict(self.data)
        configs['training_time'] = self.training_time
        with get_safe_filename(self.output_path / 'configs.json').open('w') as f:
            json.dump(configs, f, indent=4)
        self.logger.info('Saved model and configurations successfully')
