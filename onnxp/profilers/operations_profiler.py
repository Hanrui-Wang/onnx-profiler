import numpy as np

from .default_profiler import DefaultProfiler

__all__ = ['OperationsProfiler']


class OperationsProfiler(DefaultProfiler):
    @staticmethod
    def default(type, attributes, weights, inputs, outputs, **kwargs):
        return 0

    @staticmethod
    def conv(attributes, weights, inputs, outputs, **kwargs):
        kernel_size = weights[0].size
        output_size = outputs[0].size
        return np.prod([output_size[0]] + output_size[2:] + kernel_size) // attributes['group']

    @staticmethod
    def gemm(attributes, weights, inputs, outputs, **kwargs):
        kernel_size = weights[0].size
        output_size = outputs[0].size
        return np.prod([output_size[0]] + kernel_size)

    @staticmethod
    def mul(attributes, weights, inputs, outputs, **kwargs):
        return np.max([np.prod(input.size) for input in inputs]).astype(np.int)
