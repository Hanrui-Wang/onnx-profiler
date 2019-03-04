import numpy as np
import torch
from torchvision.models import *

from onnxp import *

if __name__ == '__main__':
    model = resnet101()
    inputs = torch.randn(1, 3, 224, 224)

    torch_profile(model, inputs, profiler=ActivationProfiler, verbose=True)

    flops = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum)
    params = torch_profile(model, inputs, profiler=ParametersProfile, reduction=np.sum)
    print(flops / 1e6, params / 1e6)
