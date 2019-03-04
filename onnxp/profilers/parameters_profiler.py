import numpy as np

from .default_profiler import DefaultProfiler

__all__ = ['ParametersProfile']


class ParametersProfile(DefaultProfiler):
    @staticmethod
    def default(type, attributes, weights, inputs, outputs, **kwargs):
        return np.sum(np.prod(weight.size) for weight in weights)

    @staticmethod
    def batchnormalization(attributes, weights, inputs, outputs, **kwargs):
        return np.sum(np.prod(weight.size) for weight in weights[:2])
