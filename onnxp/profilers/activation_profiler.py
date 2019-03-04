from .default_profiler import DefaultProfiler

__all__ = ['ActivationProfiler']


class ActivationProfiler(DefaultProfiler):
    @staticmethod
    def default(type, attributes, weights, inputs, outputs, **kwargs):
        return [output.size for output in outputs] if outputs else None
