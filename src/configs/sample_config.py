from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


__all__ = [
    'SampleIntervals1DConfig',
]

# TODO: currently kind of overkill to consider multiple sections of circle at a time
#       @Jens: Is this in the scope of this work? If not, make it more convinient
@dataclass
class SampleIntervals1DConfig:
    n_samples: List[int]
    methods: List[str]
    intervals: List[Tuple[float, float]]
    noise: List[float] = None

    def __post_init__(self) -> None:
        if self.noise is None:
            self.noise = [0] * len(self.n_samples)

        self.total_num_samples = 0
        self._idx = -1
        self.compact = []
        for s, m, i, n in zip(self.n_samples, self.methods, self.intervals, self.noise):
            self.compact.append((s, m, i, n))
            self.total_num_samples += s
        self._size = len(self.compact)
        self.n_samples_latent = 256
        self._construct_equidistant_latent()

    def _construct_equidistant_latent(self) -> None:
        interval_size = 0
        for _, _, (low, high), _ in self.compact:
            interval_size += high - low

        latent = []
        for _, _, (low, high), _ in self.compact:
            n = (high - low) / interval_size * self.n_samples_latent 
            latent += np.linspace(low, high, round(n)).tolist()
        self.equidistant_latent = np.array(latent)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, str, float, Tuple[float, float]]:
        self._idx += 1
        if self._idx >= self._size:
            self._idx = -1
            raise StopIteration
        else:
            return self.compact[self._idx]
