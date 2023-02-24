# standard libraries
from typing import Dict, Any, List

# third party libraries

# local libraries
from configs.data_config import *
from configs.model_config import *
from configs.optimizer_config import *
from configs.training_config import *
from utils.Layers import *


__all__ = [
    'data_to_dict',
    'optimizer_to_dict',
    'training_to_dict',
]

def data_to_dict(config: DataConfig) -> Dict[str, Any]:
    as_dict = dict(
        name=config.name,
        # TODO: store as pickle
        # compact_sampling=config.sample_config.compact,   # TODO: save further stuff?
        # equidistant_latent=config.sample_config.equidistant_latent
    )
    return as_dict


def training_to_dict(config: TrainingConfig) -> Dict[str, Any]:
    as_dict = dict()
    as_dict['optimizer'] = optimizer_to_dict(config.optim_config)
    as_dict['epochs'] = config.epochs
    as_dict['batch_size'] = config.batch_size
    as_dict['loss'] = config.loss
    as_dict['metrics'] = config.metrics
    return as_dict


def optimizer_to_dict(config: OptimizerConfig) -> Dict[str, Any]:
    as_dict = dict()
    if isinstance(config, AdamConfig):
        as_dict['name'] = 'Adam'
        as_dict['learning_rate'] = config.learning_rate
        as_dict['beta_1'] = config.beta_1
        as_dict['beta_2'] = config.beta_2
        as_dict['epsilon'] = config.epsilon
    elif isinstance(config, SGDConfig):
        as_dict['name'] = 'SGD'
        as_dict['learning_rate'] = config.learning_rate
        as_dict['momentum'] = config.momentum
        as_dict['nesterov'] = config.nesterov
    else:
        raise ValueError(f'Unknown optimizer: {config}')
    return as_dict


def model_to_dict(config: ModelConfig) -> List[Dict[str, Any]]:
    layers = []
    for layer in config.get_list_of_layers():
        if layer.type not in [LayerType.INPUT, LayerType.DENSE, LayerType.BOTTLENECK_DENSE, LayerType.OUTPUT]:
            raise NotImplementedError
        
        if layer.type == LayerType.INPUT:
            layers.append(dict(
                shape=layer.shape
            ))
        elif layer.type == LayerType.BOTTLENECK_DENSE:
            layers.append(dict(
                latent_dim=layer.latent_dim,
                layer_type=str(layer.type)
            ))
        else:
            # Dense Layers
            f_act = layer.activation if type(layer.activation) == str else layer.activation.__name__
            layers.append(dict(
                units=layer.units,
                f_act=f_act,
                layer_type=str(layer.type),
                name=layer.name,
            ))
    return layers
