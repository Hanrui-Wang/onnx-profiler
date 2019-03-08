import numpy as np
import torch
from torchvision.models import *
from onnxp import *

if __name__ == '__main__':
    model = resnet18()
    inputs = torch.randn(1, 3, 224, 224)

    mults = torch_profile(model, inputs, profiler=OperationsProfiler, reduction=np.sum)
    params = torch_profile(model, inputs, profiler=ParametersProfiler, reduction=np.sum)

    print(model)
    print(mults / 1e6, params / 1e6)
