from collections import deque

import onnx

from .model import Model

__all__ = ['onnx_profile', 'torch_profile']


def onnx_profile(model, profiler, reduction=None, verbose=False):
    model = Model(model, remove_batchnorms=True, propagate_size=True)

    outputs = []
    for module in model.modules:
        output = getattr(profiler, module.type, profiler.default)(**module.__dict__)
        outputs.append(output)

    if verbose:
        strings = []

        for module in model.modules:
            string = ', '.join(['${}'.format(variable.name) for variable in module.outputs])
            string += ' = '
            string += module.type

            if module.attributes:
                string += '['
                string += ', '.join(['{}={}'.format(k, v) for k, v in module.attributes.items()])
                string += ']'

            if module.inputs:
                string += '('
                string += ', '.join(['${}'.format(variable.name) for variable in module.inputs])
                string += ')'

            strings.append(string)

        string_length = 4 + max(len(string) for string in strings)
        output_length = 4 + max(len(str(output)) for output in outputs)

        print('-' * string_length + '  ' + '-' * output_length)

        for string, output in zip(strings, outputs):
            print(('  {: <' + str(string_length) + '}  {: <' + str(output_length) + '}').format(string, str(output)))

        print('-' * string_length + '  ' + '-' * output_length)

    if reduction is not None:
        return reduction(outputs)
    else:
        return outputs


def torch_profile(model, inputs, profiler, reduction=None, verbose=False):
    import torch
    import torch.nn as nn

    queue = deque([model])

    while queue:
        x = queue.popleft()

        for module in x._modules.values():
            if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                module.ceil_mode = False

            queue.append(module)

    torch.onnx.export(model, inputs, '/tmp/model.onnx')
    model = onnx.load('/tmp/model.onnx')

    return onnx_profile(model, profiler=profiler, reduction=reduction, verbose=verbose)
