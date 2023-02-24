# standard libraries
from dataclasses import dataclass, field
from typing import List

# third party libraries

# local libraries
from configs.optimizer_config import OptimizerConfig


__all__ = [
    'TrainingConfig',
]


@dataclass(frozen=True)
class TrainingConfig:
    optim_config: OptimizerConfig
    epochs: int
    batch_size: int
    loss: str = 'mse'
    metrics: List[str] = field(default_factory= lambda: ['accuracy'])
